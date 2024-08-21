import json
import sys
import click
import warnings

import multiprocessing as mp

from loguru import logger
from graph.graph_builder import StructuralGraphBuilder
from graph.graph_filter import GraphFilter, PearsonGraphFilter
from graph.metric_filter import MetricFilter, NonZeroMetricFilter
from ranker.circa import *
from ranker.latent_regressor import *
from evaluator import Evaluator
from typing import *


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


class CaseProcessor:
    def __init__(self, dataset: str = 'dataset_b', graph_builder: str = 'structural', metric_filters: List[str] = [],
                 graph_filters: List[str] = [], ranker: str = 'latentregressor'):
        # Initialize
        logger.info("[CaseProcessor] Initializing...")
        self.dataset = dataset

        # Initialize Graph Builder
        if graph_builder == 'structural':
            self.graph_builder = StructuralGraphBuilder(dataset=dataset)
        else:
            raise NotImplementedError(f"Unrecognized graph builder: {graph_builder}")

        # Initialize Metric Filters
        self.metric_filters: List[MetricFilter] = []
        for metric_filter in metric_filters:
            if metric_filter == 'nonzero':
                self.metric_filters.append(NonZeroMetricFilter(dataset))
            else:
                raise NotImplementedError(f"Unrecognized metric filter: {metric_filter}")

        # Initialize Graph Filter
        self.graph_filters: List[GraphFilter] = []
        for graph_filter in graph_filters:
            if graph_filter == 'pearson':
                self.graph_filters.append(PearsonGraphFilter(dataset))
            else:
                raise NotImplementedError(f"Unrecognized graph filter: {graph_filter}")

        # Initialize Ranker
        if ranker == 'latentregressor':
            self.ranker = LatentRegressorRanker(dataset)
        else:
            raise NotImplementedError(f"Unrecognized ranker: {ranker}")

        # Set processor name
        metric_filter_str = '+'.join(metric_filters) if len(metric_filters) else 'none'
        graph_filter_str = '+'.join(graph_filters) if len(graph_filters) else 'none'

        self.name = f"{graph_builder}_{metric_filter_str}_{graph_filter_str}_{ranker}"
        logger.info(f"[CaseProcessor] Case processor with name {self.name} initialized successfully!")

        # Load case info
        with open(os.path.join('data', dataset, 'labels', 'label.json'), 'rt') as f:
            self.case_info = json.load(f)

    def process_case(self, case: str):
        # # Load data
        metrics = CaseMetric.load(f"data/{self.dataset}/data/{case}/metrics.json")
        rccs, rcc_edges = RCC.load(f"data/{self.dataset}/data/{case}/rccs.json")

        for metric_filter in self.metric_filters:
            metrics, rccs = metric_filter.run_filter(self.case_info[case], metrics, rccs)

        # Build observable layer graph
        metric_edges = self.graph_builder.build(metrics)

        # Filter graph nodes and edges
        for graph_filter in self.graph_filters:
            metric_edges = graph_filter.run_filter(self.case_info[case], metrics, metric_edges)

        # Run Ranking
        metrics.value = list(metrics.value)
        rank_result = self.ranker.rank(
            case_info=self.case_info[case],
            graph_edges=metric_edges,
            metrics=metrics,
            rccs=rccs,
            rcc_edges=rcc_edges
        )

        # Dump rank_result to json
        rank_result_json = []
        for i in rank_result:
            rank_result_json.append({
                'metric': i.node.key,
                'type': i.node.kind,
                'score': i.score,
                'run_time': float(self.ranker.run_times[-1]) if hasattr(self.ranker, 'run_times') else 0.0
            })
        os.makedirs(f'results/{self.dataset}/{self.name}', exist_ok=True)
        with open(f'results/{self.dataset}/{self.name}/{case}.json', 'wt') as f:
            json.dump(rank_result_json, f)

        return rank_result, metrics


def worker(worker_id, task_queue, result_queue, *args):
    case_processor = CaseProcessor(*args)
    while not task_queue.empty():
        args = task_queue.get()
        if args is None:
            break
        label, case_idx, case, cnt = args
        logger.info(f"[Worker {worker_id}] Processing case: {case}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rank_result, metrics = case_processor.process_case(case)
        result_queue.put((case, rank_result, metrics))
        logger.info(f"[Worker {worker_id}] Finished processing: {case}")
    logger.info(f"[Worker {worker_id}] Current worker stoped.")


@click.command()
@click.option("-d", "--dataset", type=str, default='dataset_b')
@click.option("-b", "--graph-builder", type=str, default='structural')
@click.option("-mf", "--metric-filter", type=str, default=['nonzero'], multiple=True)
@click.option("-gf", "--graph-filter", type=str, default=['pearson'], multiple=True)
@click.option("-r", "--ranker", type=str, default='latentregressor')
@click.option("--cpus", type=int, default=16)
def _main(dataset: str, graph_builder: str, metric_filter: List[str], graph_filter: List[str], ranker: str, cpus: int):
    main(dataset, graph_builder, metric_filter, graph_filter, ranker, cpus)


def main(dataset: str, graph_builder: str, metric_filter: List[str], graph_filter: List[str], ranker: str, cpus: int):
    # Load all cases
    with open(os.path.join('data', dataset, 'labels', 'label.json'), 'rt') as f:
        label = json.load(f)
    evaluator = Evaluator(dataset=dataset)

    # Create queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Populate task queue
    cnt = 0
    for case_idx, case in enumerate(list(label.keys())):
        task_queue.put((label, case_idx, case, cnt))
        cnt += 1

    # Create worker processes
    processes = []
    for i in range(cpus):
        p = mp.Process(target=worker,
                       args=(i, task_queue, result_queue, dataset, graph_builder, metric_filter, graph_filter, ranker))
        processes.append(p)
        p.start()

    # Retrieve results from the result queue while workers are running
    finished_processes = 0
    while finished_processes < cpus:
        while not result_queue.empty():
            case, rank_result, metrics = result_queue.get()
            if rank_result is None:
                continue
            evaluator.evaluate_case(case, rank_result, metrics)
        for p in processes:
            if not p.is_alive():
                finished_processes += 1
                processes.remove(p)
                break

    # Collect results
    for i, p in enumerate(processes):
        logger.info(f"[main] Process {i} join.")
        p.join()
    logger.info("[main] All processes has stopped!")

    # Evalute result
    overall_mrr = evaluator.print_result()
    logger.info("[main] Finished!")

    os.makedirs(f'results/{dataset}', exist_ok=True)
    case_processor = CaseProcessor(dataset, graph_builder, metric_filter, graph_filter, ranker)
    evaluator.save_result(f'results/{dataset}/{case_processor.name}.txt'),
    return overall_mrr


if __name__ == '__main__':
    _main()

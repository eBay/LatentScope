import numpy as np
from models import *
from loguru import logger
from utils import metric_type_map
from graph.causal_methods import run_pearson


class GraphFilter:
    def __init__(self, dataset: str):
        self.dataset = dataset

    def run_filter(self, case_info, metrics: CaseMetric, graph_edges):
        return graph_edges


class PearsonGraphFilter(GraphFilter):
    def __init__(self, dataset: str):
        self.dataset = dataset

    def run_filter(self, case_info, metrics: CaseMetric, graph_edges):
        start_ts = case_info['time'] - 20 * 60
        end_ts = case_info['time'] + 20 * 60
        valid_ts = (start_ts <= metrics.ts) & (metrics.ts <= end_ts)

        change_metrics = set()
        for key, value in zip(metrics.key, metrics.value):
            if metric_type_map(self.dataset, key) in ['C', 'DE']:
                # Check if all zero
                if np.max(np.abs(value)) < 1e-10:
                    continue
                change_metrics.add(key)

        result_edges = set()
        for i in graph_edges:
            if metrics.key[i[0]] in change_metrics or metrics.key[i[1]] in change_metrics:
                result_edges.add(i)

        result_edges.update(run_pearson([i[valid_ts] for i in metrics.value], graph_edges))

        return sorted(result_edges)

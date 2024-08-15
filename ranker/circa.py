from circa.alg.ci.base import Regressor
from sklearn.linear_model._base import LinearModel
from ranker.base import Ranker
import numpy as np
from circa.alg.ci import Scorer
from typing import *
from datetime import timedelta
import networkx as nx
import os

from sklearn.linear_model import Ridge
from circa.alg.ci import RHTScorer
from circa.alg.ci.anm import ANMRegressor
from circa.alg.common import Model, zscore_conf
from circa.graph.common import StaticGraphFactory
from circa.model.case import CaseData
from circa.model.data_loader import MemoryDataLoader
from circa.model.graph import MemoryGraph, Graph
from circa.model.graph import Node as CircaNode
from circa.alg.base import Score
from models import CaseMetric, RootCause, RCC
from copy import deepcopy


class ModelWithGraph(Model):
    def analyze(
        self, data: CaseData, current: float, output_dir: str = None, return_graph: bool = False
    ) -> List[Tuple[CircaNode, Score]]:
        """
        Conduct root cause analysis
        """
        # 1. Create a graph
        if output_dir is not None:
            graph_filename = os.path.join(output_dir, f"{self._name_graph}.json")
            graph = None
            if os.path.isfile(graph_filename):
                graph = self._graph_factory.load(graph_filename)
            if graph is None:
                graph = self._graph_factory.create(data=data, current=current)
        else:
            graph = self._graph_factory.create(data=data, current=current)

        # 2. Score nodes
        scores: Dict[CircaNode, Score] = None
        for scorer in self._scorers:
            scores = scorer.score(
                graph=graph, data=data, current=current, scores=scores
            )
        if return_graph:
            return sorted(scores.items(), key=lambda item: item[1].key, reverse=True), graph
        return sorted(scores.items(), key=lambda item: item[1].key, reverse=True)


class DataRHTScorer(RHTScorer):
    def __init__(self, tau_max: int = 0, regressor: Regressor = None, use_confidence: bool = False, **kwargs):
        super().__init__(tau_max, regressor, use_confidence, **kwargs)
        self.regressor_data = {}

    @staticmethod
    def _split_train_test(
        series_x: np.ndarray,
        series_y: np.ndarray,
        train_window: int,
        test_window: int,
    ):
        train_x: np.ndarray = series_x[:train_window, :]
        train_y: np.ndarray = series_y[:train_window]
        test_x: np.ndarray = series_x[train_window:, :]
        test_y: np.ndarray = series_y[train_window:]
        return train_x, test_x, train_y, test_y

    def score_node(
        self,
        graph: Graph,
        series: Dict[CircaNode, Sequence[float]],
        node: CircaNode,
        data: CaseData,
    ) -> Score:
        parents = list(graph.parents(node))

        train_x, test_x, train_y, test_y = self.split_data(series, node, parents, data)
        z_scores = self._regressor.score(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y)

        self.regressor_data[node.metric] = {
            'train_x': train_x,
            'test_x': test_x,
            'train_y': train_y,
            'test_y': test_y,
            'parents': [i.metric for i in parents if i in series],
            'regressor': deepcopy(self._regressor._regressor)
        }

        z_score = self._aggregator(abs(z_scores))
        confidence = zscore_conf(z_score)
        if self._use_confidence:
            score = Score(confidence)
            score.key = (score.score, z_score)
        else:
            score = Score(z_score)
        score["z-score"] = z_score
        score["Confidence"] = confidence

        return score


class DataANMRegressor(ANMRegressor):
    def __init__(self, regressor: LinearModel = None, **kwargs):
        super().__init__(regressor, **kwargs)

    def _score(
        self,
        train_x: np.ndarray,
        test_x: np.ndarray,
        train_y: np.ndarray,
        test_y: np.ndarray,
    ) -> np.ndarray:
        self._regressor.fit(train_x, train_y)
        
        self.train_x = train_x
        self.test_x = test_x
        self.train_pred = self._regressor.predict(train_x)
        self.test_pred = self._regressor.predict(test_x)
        self.train_y = train_y
        self.test_y = test_y

        train_err: np.ndarray = train_y - self.train_pred
        test_err: np.ndarray = test_y - self.test_pred
        return self._zscore(train_y=train_err, test_y=test_err)


class BaseCircaRanker(Ranker):
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.scorer = DataRHTScorer(regressor=DataANMRegressor(regressor=Ridge()))
    
    def _circa_rank(self, case_info, graph_edges: List[Tuple[int, int]], metrics: CaseMetric, return_graph: bool = False) -> List[Tuple[CircaNode, Score]]:
        # Build circa_graph
        circa_graph = nx.DiGraph()
        for key, entity in zip(metrics.key, metrics.entity):
            circa_graph.add_node(CircaNode(entity=entity, metric=key))

        for (u, v) in graph_edges:
            u_node = CircaNode(entity=metrics.entity[u], metric=metrics.key[u])
            v_node = CircaNode(entity=metrics.entity[v], metric=metrics.key[v])
            circa_graph.add_edge(u_node, v_node)

        graph_factory = StaticGraphFactory(MemoryGraph(circa_graph))
        scorers = [
            self.scorer
        ]
        model = ModelWithGraph(graph_factory=graph_factory, scorers=scorers)

        # Build metric data
        metric_data = {}
        for key, entity, value in zip(metrics.key, metrics.entity, metrics.value):
            metric_data.setdefault(entity, {})
            metric_data[entity][key] = list(zip(metrics.ts, value))

        data = CaseData(
            data_loader=MemoryDataLoader(metric_data),
            sli=CircaNode(metrics.entity[metrics.key.index(case_info['trigger'][0])], case_info['trigger'][0]),
            detect_time=case_info['time'],
            interval=timedelta(seconds=60)
        )

        result = model.analyze(data, current=case_info['time'] + 10 * 60, return_graph=return_graph)
        return result

    def rank(self, case_info, graph_edges: List[Tuple[int, int]], metrics: CaseMetric, nodes: List[RCC], rcc_edges: Dict[Tuple[RCC, RCC], Set[str]]) -> List[RootCause]:
        raise NotImplementedError()

class FixedDAScorer(Scorer):
    """
    Scorer with descendant adjustment
    """

    def __init__(
        self,
        dataset: str,
        threshold: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self._threshold = max(threshold, 0.0)
        self._aggregator = lambda x: max(x, key=lambda y: y.score)

    def score(
        self,
        graph: Graph,
        data: CaseData,
        current: float,
        scores: Dict[RCC, Score] = None,
    ) -> Dict[RCC, Score]:
        sorted_nodes = [
            {node for node in nodes if node in scores}
            for nodes in graph.topological_sort
        ]
        # 0. Set topological rank
        for index, nodes in enumerate(sorted_nodes):
            for node in nodes:
                score = scores[node]
                score["index"] = index

        # 1. Gather child scores
        new_scores = deepcopy(scores)
        child_scores: Dict[RCC, Set[RCC]] = {}
        for nodes in reversed(sorted_nodes):
            for node in nodes:
                child_score: Set[RCC] = set()
                for child in graph.children(node):
                    if child in scores:
                        child_score.add(child)
                        if scores[child].score < self._threshold:
                            child_score.update(child_scores.get(child, {}))
                child_scores[node] = child_score

        # 2. Set child_score
        for node, score in scores.items():
            if score.score >= self._threshold:
                child_score = child_scores.get(node)
                if child_score:
                    new_score = self._aggregator([new_scores[ch] for ch in child_score])
                    score.score += new_score.score
                    score["child_score"] = new_score
                    new_scores[node].score = score.score

        # 3. Set key
        for score in scores.values():
            if "index" not in score._info:
                continue
            score.key = (score.score, -score["index"], score.get("z-score", 0))
        return scores

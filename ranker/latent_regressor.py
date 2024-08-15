import numpy as np
from typing import *
from ranker.circa import BaseCircaRanker, DataANMRegressor, DataRHTScorer, FixedDAScorer
from sklearn.linear_model import Ridge
from circa.model.graph import Node as CircaNode
from circa.alg.base import Score
from models import CaseMetric, RootCause, RCC


class LatentRegressorRanker(BaseCircaRanker):
    def __init__(self, dataset: str):
        super().__init__(dataset)

        # Reset scorer
        self.dataset = dataset
        self.scorer = DataRHTScorer(regressor=DataANMRegressor(regressor=Ridge()))

        # Initialize DAScorer
        self.da_scorer = FixedDAScorer(dataset, threshold=5.0)

    @staticmethod
    def _ridge_regressor(case_info, ts: np.ndarray, u_arr: np.ndarray, v_arr: np.ndarray, v_score: float) -> Tuple[float, float]:
        start_ts = case_info['time']
        end_ts = case_info['time'] + 60 * 10
        valid_ts = (start_ts <= ts) & (ts <= end_ts)
        u_arr, v_arr = u_arr[valid_ts], v_arr[valid_ts]

        # Adjust score with ridge regressor
        regressor = Ridge()
        pred_arr = regressor.fit(u_arr.reshape(-1, 1), v_arr).predict(u_arr.reshape(-1, 1))
        pred_err = np.abs(pred_arr - v_arr)

        # Calculate error score
        v_portion = np.max(pred_err) / np.max(np.abs(v_arr))
        u_score = (1.0 - v_portion) * v_score
        v_score = v_portion * v_score

        return u_score, v_score

    def latent_regressor(
            self,
            case_info,
            circa_result: Dict[CircaNode, Score],
            metrics: CaseMetric,
            rccs: List[RCC],
            rcc_edges: Dict[Tuple[RCC, RCC], Set[str]]):
        circa_result = dict(circa_result)
        key_to_score = dict([(k.metric, v.score) for (k, v) in circa_result.items()])
        key_to_index = dict([(k, i) for (i, k) in enumerate(metrics.key)])

        def error_value(key: str) -> np.ndarray:
            # Calculate residual value for metrics
            parents = self.scorer.regressor_data[key]['parents']
            series_y = metrics.value[key_to_index[key]]
            if len(parents) == 0:
                train_y = self.scorer.regressor_data[key]['train_y']
                pred_y = np.ones(len(series_y), dtype=np.float64) * np.mean(train_y)
            else:
                series_x = np.stack([metrics.value[key_to_index[i]] for i in parents], axis=1)
                pred_y = self.scorer.regressor_data[key]['regressor'].predict(series_x)

            return series_y - pred_y

        # Record metrics in rcc (for faster computation)
        metrics_in_rcc: Dict[RCC, Set[str]] = {}
        for (rcc, _), co_metrics in rcc_edges.items():
            metrics_in_rcc.setdefault(rcc, set())
            metrics_in_rcc[rcc].update(co_metrics.intersection(key_to_score))

        # Record RCC Scores
        pa_scores: Dict[RCC, float] = dict((nd, 0.0) for nd in rccs)
        ch_scores: Dict[RCC, Tuple[str, float]] = {}

        # Initialize ch_scores (metrics not shared by next(rcc))
        for node in rccs:
            ch_candidates = set(node.metrics) - metrics_in_rcc.get(node, set())
            if len(ch_candidates) == 0:
                continue
            max_ch = max(ch_candidates, key=lambda x: key_to_score.get(x, 0.0))
            ch_scores[node] = (max_ch, key_to_score.get(max_ch, 0.0))
            # ch_scores[node] = float(max([0.0] + [key_to_score.get(i, 0.0) for i in set(node.metrics) - metrics_in_rcc.get(node, set())]))

        # Iterate over rcc_edges, append a new node to parent edges
        for (a, b), metrics_in_b in rcc_edges.items():
            # (Step 1) Up Step
            metrics_in_b = set(rccs[rccs.index(b)].metrics).intersection(key_to_score)
            if len(metrics_in_b) == 0:
                continue

            a_max_candidates = metrics_in_rcc[a] - metrics_in_b
            if len(a_max_candidates) == 0:
                continue
            
            a_max = max(a_max_candidates, key=lambda x: key_to_score[x])
            b_max = max(metrics_in_b, key=lambda x: key_to_score[x])
            score_cur, ch_score = self._ridge_regressor(case_info, metrics.ts, error_value(a_max), error_value(b_max), key_to_score[b_max])
        
            # (Step 2) Down Step
            for c, b1 in rcc_edges:
                if b1 != b or c == a: continue
                c_max_candidates = metrics_in_rcc[c] - metrics_in_rcc[a]
                if len(c_max_candidates) == 0:
                    continue
                c_max = max(c_max_candidates, key=lambda x: key_to_score[x])
                c_score = self._ridge_regressor(case_info, metrics.ts, error_value(c_max), error_value(b_max), key_to_score[b_max])[1]
                score_cur = min(score_cur, c_score)
            pa_scores[a] = max(pa_scores[a], float(score_cur))
            if b in ch_scores and b_max == ch_scores[b][0]:
                ch_scores[b] = (b_max, min(ch_scores[b][1], float(ch_score)))

        # (Final Step) Merge results
        results: Dict[RCC, float] = {}
        for node in pa_scores | ch_scores:
            results[node] = max(pa_scores.get(node, 0.0), ch_scores.get(node, (None, 0.0))[1])

        return results

    def rank(self, case_info, graph_edges: List[Tuple[int, int]], metrics: CaseMetric, rccs: List[RCC], rcc_edges: Dict[Tuple[RCC, RCC], Set[str]]):
        # (STEP 1) Metric Step (with CIRCA)
        metric_layer_result, graph = self._circa_rank(case_info, graph_edges, metrics, return_graph=True)
        metric_layer_result = self.da_scorer.score(graph, None, None, dict(metric_layer_result))

        # (STEP 2) Run LatentRegressor
        result = self.latent_regressor(case_info, metric_layer_result, metrics, rccs, rcc_edges)

        # Sort result according to their scores
        result = sorted([RootCause(k, v) for (k, v) in result.items()], key=lambda x: x.score, reverse=True)
        return result

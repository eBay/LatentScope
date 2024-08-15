from models import CaseMetric
from sklearn.preprocessing import StandardScaler
from models import RCC
import numpy as np
from typing import List


class MetricFilter:
    def __init__(self, dataset: str):
        self.dataset = dataset

    def _metric_filter(self, case_info, metrics: CaseMetric) -> CaseMetric:
        raise NotImplementedError()

    def run_filter(self, case_info, metrics: CaseMetric, nodes: List[RCC]):
        filtered_metrics = self._metric_filter(case_info, metrics)

        # Filter nodes
        filtered_keys = set(filtered_metrics.key)
        result_nodes: List[RCC] = []
        
        for nd in nodes:
            result_nodes.append(RCC(
                key=nd.key,
                kind=nd.kind,
                metrics=list(filtered_keys.intersection(nd.metrics))
            ))
        return filtered_metrics, result_nodes


class NonZeroMetricFilter(MetricFilter):
    def __init__(self, 
                 dataset: str,
                 thresh: float = 0.0,
                 test_start: int=-10800,
                 test_end: int=1200):
        super().__init__(dataset)
        self.thresh = thresh

        self.test_start = test_start
        self.test_end = test_end

    def _metric_filter(self, case_info, metrics: CaseMetric):
        start_ts = case_info['time'] + self.test_start
        end_ts = case_info['time'] + self.test_end
        valid_ts = (start_ts <= metrics.ts) & (metrics.ts <= end_ts)
        # Set valid_idx
        valid_idx = []

        for i, (key, entity, value) in enumerate(zip(metrics.key, metrics.entity, metrics.value)):
            value = value[valid_ts]
            value = StandardScaler().fit_transform(value.reshape(-1, 1)).flatten()
            
            # Check if all zero
            if np.max(np.abs(value - value[0])) < 1e-10:
                continue
            valid_idx.append(i)

        new_metrics = CaseMetric(
            ts=metrics.ts,
            key=[metrics.key[i] for i in valid_idx],
            entity=[metrics.entity[i] for i in valid_idx],
            value=[metrics.value[i] for i in valid_idx]
        )

        return new_metrics

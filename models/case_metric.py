import numpy as np
from dataclasses import dataclass
from typing import *
import json


@dataclass
class CaseMetric:
    ts: np.ndarray = None
    key: List[str] = None
    entity: List[str] = None
    value: List[np.ndarray] = None

    def load(filepath: str) -> 'CaseMetric':
        with open(filepath, 'rt') as f:
            content = json.load(f)
        result = CaseMetric(None, [], [], [])
        for item in content:
            assert 'key' in item and 'service' in item and 'data' in item
            ts, values = map(np.array, zip(*item['data'].items()))
            ts = ts.astype(int)
            values = values.astype(float)
            if result.ts is not None and np.any(ts != result.ts):
                raise RuntimeError(f"[CaseMetric] ts in {filepath} should be same!")
            result.ts = ts
            result.key.append(item['key'])
            result.entity.append(item['service'])
            result.value.append(values)

        return result

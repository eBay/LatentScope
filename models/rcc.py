from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set
import json


@dataclass
class RCC:
    key: str 
    kind: str = 'default'
    metrics: List[str] = field(default_factory=list)

    def __eq__(self, __value: 'RCC') -> bool:
        return self.key == __value.key and self.kind == __value.kind

    def __hash__(self) -> int:
        return hash(self.key) * 1023 + hash(self.kind)

    def load(filepath: str) -> Tuple[List['RCC'], Dict[Tuple['RCC', 'RCC'], Set[str]]]:
        """
            Load from filepath.
            Return: List of RCCs (including RCC-Metric links) and edges between RCCs.
        """
        rccs: List['RCC'] = []
        rcc_edges: Dict[Tuple['RCC', 'RCC'], Set[str]] = {}

        with open(filepath, 'rt') as f:
            content = json.load(f)
        
        for rcc in content['rccs']:
            rccs.append(RCC(key=rcc['key'], kind=rcc['kind'], metrics=rcc['linked_metrics']))

        for e in content['rcc_edges']:
            u = [rcc for rcc in rccs if rcc.key == e['rcc1']]
            v = [rcc for rcc in rccs if rcc.key == e['rcc2']]
            rcc_edges[(u[0], v[0])] = set(e['linked_metrics'])

        return rccs, rcc_edges

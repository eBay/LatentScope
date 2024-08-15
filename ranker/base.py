from models import CaseMetric, RootCause, RCC
from typing import List, Tuple, Dict, Set


class Ranker:
    def __init__(self):
        pass

    def rank(self, case_info, graph_edges: List[Tuple[int, int]], metrics: CaseMetric, nodes: List[RCC], rcc_edges: Dict[Tuple[RCC, RCC], Set[str]]) -> List[RootCause]:
        raise NotImplementedError()

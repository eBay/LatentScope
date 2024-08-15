from typing import *
from models import CaseMetric, RootCause, RCC


class Ranker:
    def __init__(self):
        pass

    def rank(self, case_info: dict, graph_edge: List[Tuple[int, int]], metrics: CaseMetric, nodes: List[RCC]) -> List[RootCause]:
        raise NotImplementedError()

from dataclasses import dataclass
from models.rcc import RCC


@dataclass
class RootCause:
    node: RCC
    score: float

    @staticmethod
    def from_dict(data):
        return RootCause(
            node=RCC(key=data['key'], kind=data['type'], metrics=[]),
            score=0.0
        )

    def __eq__(self, __value: object) -> bool:
        return self.compare(__value)

    def compare(self, data):
        if type(data) == dict:
            data = RootCause.from_dict(data)

        return self.node.key == data.node.key

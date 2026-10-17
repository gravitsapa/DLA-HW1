import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric


class IncorrectLosses(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, loss_is_correct: bool, **kwargs) -> dict:
        return {"value": 1 - int(loss_is_correct)}

    def avg_type(self) -> str:
        return "sum"

import math
from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch import Tensor, nn


class DecodeMethod(Enum):
    GREEDY = 1
    BEAM_SEARCH = 2


class DecoderBase(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def greedy_decode(self, **kwargs) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def beam_search(self, beam_size: int | None = None, **kwargs) -> dict:
        raise NotImplementedError()

    def forward(self, method: DecodeMethod, **kwargs) -> dict:
        if method == DecodeMethod.GREEDY:
            return self.greedy_decode(**kwargs)
        elif method == DecodeMethod.BEAM_SEARCH:
            return self.beam_search(**kwargs)
        raise ValueError()


def log_add(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    high, low = max(left, right), min(left, right)
    return high + math.log1p(math.exp(low - high))

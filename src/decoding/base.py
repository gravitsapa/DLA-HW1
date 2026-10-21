import math
from abc import ABC, abstractmethod
from enum import StrEnum

import torch
from torch import Tensor, nn

from src.decoding.lm import NoTokenLM


class DecodeMethod(StrEnum):
    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"
    BEAM_SEARCH_LM = "beam_search_lm"


class DecoderBase(nn.Module):
    def __init__(self, lm: NoTokenLM | None, lm_weight: float = 0.2, **kwargs):
        super().__init__()
        self.lm = lm
        self.lm_weight = lm_weight

    @abstractmethod
    def greedy_decode(self, **kwargs) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def beam_search(self, beam_size: int | None = None, **kwargs) -> dict:
        raise NotImplementedError()

    def beam_search_lm(self, **kwargs) -> dict:
        if self.lm is None:
            raise ValueError("beam_search_lm requires a configured lm")
        self.lm.clear_cache()
        try:
            return self.beam_search(_use_lm=True, **kwargs)
        finally:
            self.lm.clear_cache()

    def _lm_scores(self, prefixes, device, n_tokens):
        assert self.lm is not None
        scores = self.lm.score_prefixes(prefixes, device=device)
        return scores

    def forward(self, method: DecodeMethod, **kwargs) -> dict:
        if method == DecodeMethod.GREEDY:
            decode_data_kw = self.greedy_decode(**kwargs)
        elif method == DecodeMethod.BEAM_SEARCH:
            decode_data_kw = self.beam_search(**kwargs)
        elif method == DecodeMethod.BEAM_SEARCH_LM:
            decode_data_kw = self.beam_search_lm(**kwargs)
        else:
            raise ValueError(f"Unknown decoding method: {method}")

        return decode_data_kw


def log_add(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    high, low = max(left, right), min(left, right)
    return high + math.log1p(math.exp(low - high))

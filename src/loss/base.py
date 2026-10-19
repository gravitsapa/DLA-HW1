from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class LossBase(nn.Module, ABC):
    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        text_encoded: Tensor,
        text_encoded_length: Tensor,
        **batch,
    ) -> dict:
        raise NotImplementedError()

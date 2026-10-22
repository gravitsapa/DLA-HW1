from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class EncoderBase(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        spectrogram: Tensor,
        spectrogram_length: Tensor,
        **batch,
    ) -> dict:
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError()

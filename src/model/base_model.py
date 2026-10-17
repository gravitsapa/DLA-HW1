from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class BaseModel(nn.Module, ABC):
    """
    Base model that defines the contract
    """

    def __init__(
        self,
        n_feats: int,
        n_tokens: int,
    ):
        super().__init__()

        self._n_feats = n_feats
        self._n_tokens = n_tokens

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info += (
            f"\nInput features: {self._n_feats}, output tokens: {self._n_tokens}"
        )
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"

        return result_info

    @abstractmethod
    def forward(
        self,
        spectrogram: Tensor,
        spectrogram_length: Tensor,
        **batch,
    ) -> dict:
        raise NotImplementedError()

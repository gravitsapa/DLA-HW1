import torch
import torchaudio.functional as AF
from torch import Tensor, nn

from src.loss.base import LossBase


class RNNTLossWrapper(LossBase):
    def __init__(
        self,
        blank_id: int,
        **kwargs,
    ):
        super().__init__()

        self.blank_token = blank_id

    def forward(
        self,
        text_encoded: Tensor,
        text_encoded_length: Tensor,
        *,
        logits: Tensor,
        logits_length: Tensor,
        **batch,
    ) -> dict:
        return {
            "loss": AF.rnnt_loss(
                logits=logits,
                targets=text_encoded.to(dtype=torch.int32),
                logit_lengths=logits_length.to(dtype=torch.int32),
                target_lengths=text_encoded_length.to(dtype=torch.int32),
                blank=self.blank_token,
            )
        }

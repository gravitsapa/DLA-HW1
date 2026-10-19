import torch
from torch import Tensor
from torch.nn import CTCLoss

from src.loss.base import LossBase


class CTCLossWrapper(LossBase):
    def __init__(self, blank_id: int, **kwargs):
        super().__init__()

        self.loss = CTCLoss(
            blank=blank_id,
            **kwargs,
        )

    def forward(
        self,
        text_encoded: Tensor,
        text_encoded_length: Tensor,
        *,
        log_probs: Tensor,
        log_probs_length: Tensor,
        **batch,
    ) -> dict:
        log_probs_t = torch.transpose(log_probs, 0, 1)

        loss = self.loss(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )

        return {"loss": loss}

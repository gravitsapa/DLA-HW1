import torch
import torchaudio.functional as AF
from torch import Tensor, nn


class RNNTLoss(nn.Module):
    def __init__(
        self,
        blank_token: int,
    ):
        super().__init__()

        self.blank_token = blank_token

    def forward(
        self,
        logits: Tensor,
        logits_length: Tensor,
        text_encoded: Tensor,
        text_encoded_length: Tensor,
    ):
        return AF.rnnt_loss(
            logits=logits,
            targets=text_encoded,
            logit_lengths=logits_length,
            target_lengths=text_encoded_length,
            blank=self.blank_token,
        )

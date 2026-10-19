import torch
import torchaudio.functional as AF
from torch import Tensor, nn


class RNNTLossWrapper(nn.Module):
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
        **batch,
    ):
        return {
            "loss": AF.rnnt_loss(
                logits=logits,
                targets=text_encoded.to(dtype=torch.int32),
                logit_lengths=logits_length.to(dtype=torch.int32),
                target_lengths=text_encoded_length.to(dtype=torch.int32),
                blank=self.blank_token,
            )
        }

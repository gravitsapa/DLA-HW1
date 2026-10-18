import torch
from torch import Tensor, nn

from src.model.head.base import Head


class LinearHead(Head):
    def __init__(
        self,
        encoder_hidden_size: int,
        n_tokens: int,
    ):
        super().__init__()

        self.linear = nn.Linear(encoder_hidden_size, n_tokens)

    def forward(
        self,
        hidden_state: Tensor,
        hidden_states_length: Tensor,
        text_encoded: Tensor,
        text_encoded_length: Tensor,
    ) -> dict:
        x = self.linear(hidden_state)
        log_probs = nn.functional.log_softmax(x, dim=-1)
        return {
            "log_probs": log_probs,
            "log_probs_length": hidden_states_length,
        }

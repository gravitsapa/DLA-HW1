from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class Predictor(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        text_encoded: Tensor,
        text_encoded_length: Tensor,
    ) -> Tensor:
        raise NotImplementedError()


class PredictorLSTM(Predictor):
    def __init__(
        self,
        n_tokens: int,
        emb_size: int,
        hidden_size: int,
        lstm_layers: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(n_tokens, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, lstm_layers)
        self.norm = nn.LayerNorm(hidden_size)

    @abstractmethod
    def forward(
        self,
        text_encoded: Tensor,
        text_encoded_length: Tensor,
    ) -> Tensor:
        embeddings = self.embedding(text_encoded)
        output, new_state = self.lstm(embeddings, None)
        output = self.norm(output)
        return output

    @property
    def output_size(self) -> int:
        return self.hidden_size

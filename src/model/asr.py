import torch
from torch import Tensor, nn

from src.model.base_model import BaseModel
from src.model.common import Factory
from src.model.encoder.base import EncoderBase
from src.model.head.base import Head


class ASR(BaseModel):
    def __init__(
        self,
        encoder_factory: Factory[EncoderBase],
        head_factory: Factory[Head],
    ):
        super().__init__()

        self.encoder = encoder_factory()
        self.head = head_factory()

    def forward(
        self,
        spectrogram: Tensor,
        spectrogram_length: Tensor,
        text_encoded: Tensor,
        text_encoded_length: Tensor,
        **batch,
    ) -> dict:
        encoder_kwargs = self.encoder(spectrogram, spectrogram_length)

        output = self.head(
            text_encoded=text_encoded,
            text_encoded_length=text_encoded_length,
            **encoder_kwargs,
        )

        return output

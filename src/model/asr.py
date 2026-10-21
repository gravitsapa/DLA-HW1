import torch
from torch import Tensor, nn

from src.decoding.base import DecodeMethod, DecoderBase
from src.model.base_model import BaseModel
from src.model.common import Factory
from src.model.encoder.base import EncoderBase
from src.model.head.base import Head


class ASR(BaseModel):
    def __init__(
        self,
        n_tokens: int,
        blank_id: int,
        encoder_factory: Factory[EncoderBase],
        head_factory: Factory[Head],
        decoder_factory: Factory[DecoderBase],
    ):
        super().__init__()

        self.encoder = encoder_factory(n_tokens=n_tokens)
        self.head = head_factory(
            blank_id=blank_id,
            encoder_hidden_size=self.encoder.output_size,
            n_tokens=n_tokens,
        )
        self.decoder = decoder_factory(blank_id=blank_id)

    def encode(
        self,
        spectrogram: Tensor,
        spectrogram_length: Tensor,
        **batch,
    ) -> dict:
        encoder_hidden = self.encoder(spectrogram, spectrogram_length)
        return encoder_hidden

    def forward(
        self,
        spectrogram: Tensor,
        spectrogram_length: Tensor,
        text_encoded: Tensor,
        text_encoded_length: Tensor,
        **batch,
    ) -> dict:
        encoder_kwargs = self.encode(spectrogram, spectrogram_length)

        output = self.head(
            text_encoded=text_encoded,
            text_encoded_length=text_encoded_length,
            **encoder_kwargs,
        )

        return encoder_kwargs | output

    def decode(
        self,
        decode_method: DecodeMethod,
        **kwargs,
    ) -> dict:
        return self.decoder(
            method=decode_method,
            head=self.head,
            **kwargs,
        )

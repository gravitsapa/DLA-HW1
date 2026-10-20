import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.model.encoder.base import EncoderBase


def _padding_mask(lengths: Tensor, time_steps: int) -> Tensor:
    return torch.arange(time_steps, device=lengths.device)[None] >= lengths[:, None]


class ConvSubsampling(nn.Module):
    def __init__(self, n_feats: int, model_dim: int, channels: int):
        super().__init__()
        self.model_dim = model_dim
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        n_freq = (n_feats + 3) // 4
        self.proj = nn.Linear(channels * n_freq, model_dim)

    def forward(self, spectrogram: Tensor, lengths: Tensor) -> Tensor:
        batch_size, _, time_steps = spectrogram.shape

        mask = _padding_mask(lengths, time_steps)
        x = spectrogram.masked_fill(mask[:, None], 0).unsqueeze(1)
        for conv in (self.conv1, self.conv2):
            x = F.relu(conv(x))
            lengths = (lengths + 1) // 2
            mask = _padding_mask(lengths, x.shape[-1])

            x = x.masked_fill(mask[:, None, None], 0)
        x = x.permute(0, 3, 1, 2).flatten(2)
        return self.proj(x).masked_fill(mask[..., None], 0)


class PosEncoder(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim

    def forward(self, x: Tensor) -> Tensor:
        pos = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
        freq = torch.exp(
            torch.arange(0, self.model_dim, 2, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) / self.model_dim)
        )
        angles = pos[:, None] * freq[None]
        encoded = angles.new_empty(pos.numel(), self.model_dim)
        encoded[:, 0::2] = angles.sin()
        encoded[:, 1::2] = angles[:, : self.model_dim // 2].cos()
        return x + encoded.to(dtype=x.dtype)[None]


class FF(nn.Module):
    def __init__(self, model_dim: int, expansion_factor: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, expansion_factor * model_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(expansion_factor * model_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class MaskedBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        values = x.transpose(1, 2)
        valid = values[~padding_mask]
        normalized = super().forward(valid)
        output = torch.zeros_like(values)
        output[~padding_mask] = normalized
        return output.transpose(1, 2)


class ConvBlock(nn.Module):
    def __init__(self, model_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.pointwise_in = nn.Conv1d(model_dim, 2 * model_dim, kernel_size=1)
        self.depthwise = nn.Conv1d(
            model_dim, model_dim, kernel_size=kernel_size, groups=model_dim
        )
        self.batch_norm = MaskedBatchNorm1d(model_dim)
        self.pointwise_out = nn.Conv1d(model_dim, model_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        left = (kernel_size - 1) // 2
        self.padding = (left, kernel_size - 1 - left)

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        x = self.norm(x).transpose(1, 2)
        x = F.glu(self.pointwise_in(x), dim=1)
        x = x.masked_fill(padding_mask[:, None], 0)
        x = self.depthwise(F.pad(x, self.padding))
        x = F.silu(self.batch_norm(x, padding_mask))
        x = self.dropout(self.pointwise_out(x).transpose(1, 2))
        return x.masked_fill(padding_mask[..., None], 0)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        ff_expansion_factor: int,
        conv_kernel_size: int,
        dropout: float,
        attention_dropout: float,
    ):
        super().__init__()
        self.ffn1 = FF(model_dim, ff_expansion_factor, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.conv = ConvBlock(model_dim, conv_kernel_size, dropout)
        self.ffn2 = FF(model_dim, ff_expansion_factor, dropout)
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        mask = padding_mask[..., None]
        x = (x + 0.5 * self.ffn1(x)).masked_fill(mask, 0)
        normalized = self.norm1(x)
        key_padding_mask = padding_mask.clone()

        key_padding_mask[:, 0] = False
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = (x + self.dropout1(attended)).masked_fill(mask, 0)
        x = (x + self.conv(x, padding_mask)).masked_fill(mask, 0)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x).masked_fill(mask, 0)


class Conformer(EncoderBase):
    def __init__(
        self,
        n_feats: int,
        model_dim: int,
        n_layers: int,
        n_heads: int,
        ff_expansion_factor: int,
        conv_kernel_size: int,
        dropout: float,
        attention_dropout: float,
        subsampling_channels: int | None = None,
        **kwargs,
    ):
        super().__init__()

        self.n_feats = n_feats
        self.model_dim = model_dim
        self.subsampling = ConvSubsampling(
            n_feats,
            model_dim,
            model_dim if subsampling_channels is None else subsampling_channels,
        )
        self.input_dropout = nn.Dropout(dropout)
        self.position_encoding = PosEncoder(model_dim)
        self.blocks = nn.ModuleList(
            ConformerBlock(
                model_dim,
                n_heads,
                ff_expansion_factor,
                conv_kernel_size,
                dropout,
                attention_dropout,
            )
            for _ in range(n_layers)
        )

    @property
    def output_size(self) -> int:
        return self.model_dim

    @staticmethod
    def transform_input_lengths(input_lengths: Tensor) -> Tensor:
        return (input_lengths + 3) // 4

    def forward(
        self,
        spectrogram: Tensor,
        spectrogram_length: Tensor,
        **batch,
    ) -> dict:
        lengths = spectrogram_length.to(device=spectrogram.device)
        x = self.subsampling(spectrogram, lengths)
        output_lengths = self.transform_input_lengths(spectrogram_length)

        padding_mask = _padding_mask(output_lengths.to(x.device), x.shape[1])
        x = self.input_dropout(self.position_encoding(x))
        x = x.masked_fill(padding_mask[..., None], 0)
        for block in self.blocks:
            x = block(x, padding_mask)
        return {"hidden_state": x, "hidden_states_length": output_lengths}

import torch
from torch import Tensor, nn

from src.model.common import ModuleFactory
from src.model.encoder.base import EncoderBase


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels_size: tuple[int, int],
        stride: tuple[int, int],
        act_fn: ModuleFactory,
    ):
        super().__init__()

        frequency_kernel, time_kernel = kernels_size
        if min(frequency_kernel, time_kernel) < 1:
            raise ValueError("Convolution kernel sizes must be positive")
        # Total padding is kernel_size - 1 on each axis. With stride s,
        # this gives ceil(input_size / s), including for even kernels.
        left = (time_kernel - 1) // 2
        top = (frequency_kernel - 1) // 2
        self.padding = (
            left,
            time_kernel - 1 - left,
            top,
            frequency_kernel - 1 - top,
        )
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(frequency_kernel, time_kernel),
                stride=stride,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            act_fn(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(nn.functional.pad(x, self.padding))


class DeepSpeech2(EncoderBase):
    """
    DeepSpeech2-like model
    """

    def __init__(
        self,
        n_feats: int,
        n_tokens: int,
        conv_channels: int,
        conv_act_fn: ModuleFactory,
        n_rnn_layers: int,
        rnn_hidden_size: int,
        conv1_kernel_size: tuple[int, int] = (41, 11),
        conv2_kernel_size: tuple[int, int] = (21, 11),
        lookahead_kernel_size: int = 81,
    ):
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels=1,
            out_channels=conv_channels,
            kernels_size=conv1_kernel_size,
            stride=(2, 2),
            act_fn=conv_act_fn,
        )
        self.conv2 = ConvBlock(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernels_size=conv2_kernel_size,
            stride=(2, 1),
            act_fn=conv_act_fn,
        )

        rnn_input_size = conv_channels * ((n_feats + 3) // 4)

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=n_rnn_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.lookahead_kernel = lookahead_kernel_size
        self.lookahead = nn.Conv1d(
            in_channels=rnn_hidden_size,
            out_channels=rnn_hidden_size,
            groups=rnn_hidden_size,
            kernel_size=self.lookahead_kernel,
            padding=0,
            bias=False,
        )

    def forward(
        self,
        spectrogram: Tensor,
        spectrogram_length: Tensor,
        **batch,
    ) -> dict:
        x = spectrogram
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x.permute((0, 3, 1, 2)).flatten(start_dim=-2, end_dim=-1)
        x, _ = self.rnn_layers(x)
        lengths_now = (1 + spectrogram_length.to(x.device)) // 2
        time_indices = torch.arange(x.shape[1], device=x.device)
        padding_mask = time_indices.unsqueeze(0) >= lengths_now.unsqueeze(1)

        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = x.permute((0, 2, 1))
        x = nn.functional.pad(x, (0, self.lookahead_kernel - 1), value=0)
        x = self.lookahead(x)

        hidden_state = x.permute((0, 2, 1))
        hidden_states_length = self.transform_input_lengths(spectrogram_length)

        return {
            "hidden_state": hidden_state,
            "hidden_states_length": hidden_states_length,
        }

    def transform_input_lengths(self, input_lengths):
        return (input_lengths + 1) // 2

    @property
    def output_size(self) -> int:
        return self.rnn_hidden_size

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
        padding: tuple[int, int],
        act_fn: ModuleFactory,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernels_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            act_fn(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


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
    ):
        super().__init__()

        if n_feats % 4 != 0:
            raise ValueError()

        self.conv1 = ConvBlock(
            in_channels=1,
            out_channels=conv_channels,
            kernels_size=(41, 11),
            stride=(2, 2),
            padding=(20, 5),
            act_fn=conv_act_fn,
        )
        self.conv2 = ConvBlock(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernels_size=(21, 11),
            stride=(2, 1),
            padding=(10, 5),
            act_fn=conv_act_fn,
        )

        rnn_input_size = conv_channels * n_feats // 4

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=n_rnn_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.lookahead_kernel = 81
        self.lookahead = nn.Conv1d(
            in_channels=rnn_hidden_size,
            out_channels=rnn_hidden_size,
            groups=rnn_hidden_size,
            kernel_size=self.lookahead_kernel,
            padding=0,
            bias=False,
        )

        self.linear_head = nn.Linear(
            in_features=rnn_hidden_size,
            out_features=n_tokens,
        )

    def forward(
        self,
        spectrogram: Tensor,
        spectrogram_length: Tensor,
        **batch,
    ) -> dict:
        x = spectrogram  # [B, F, T]
        x = x.unsqueeze(1)  # [B, C(1), F, T]

        x = self.conv1(x)  # [B, CONV_C, F / 2, T / 2]
        x = self.conv2(x)  # [B, CONV_C, F / 4, T / 2]

        x = x.permute((0, 3, 1, 2)).flatten(start_dim=-2, end_dim=-1)
        # [B, T / 2, NEW_C], NEWC = CONV_C * F / 4

        x, _ = self.rnn_layers(x)  # [B, T / 2, RNN_HIDDEN]

        lengths_now = (1 + spectrogram_length.to(x.device)) // 2
        time_indices = torch.arange(x.shape[1], device=x.device)
        padding_mask = time_indices.unsqueeze(0) >= lengths_now.unsqueeze(1)
        # [B, T / 2]

        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = x.permute((0, 2, 1))  # [B, RNN_HIDDEN, T / 2]
        x = nn.functional.pad(x, (0, self.lookahead_kernel - 1), value=0)
        # [B, RNN_HIDDEN, T / 2 + KERNEL_SIZE - 1]

        x = self.lookahead(x)
        # [B, RNN_HIDDEN, T / 2]

        hidden_state = x.permute((0, 2, 1))
        # [B, T / 2, RNN_HIDDEN]

        hidden_states_length = self.transform_input_lengths(spectrogram_length)

        return {
            "hidden_state": hidden_state,
            "hidden_states_length": hidden_states_length,
        }

    def transform_input_lengths(self, input_lengths):
        """
        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return (input_lengths + 1) // 2

    @property
    def output_size(self) -> int:
        return self.rnn_hidden_size

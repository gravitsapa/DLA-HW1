import torch
import torchaudio
from torch import Tensor, nn


class SpeedPerturbation(nn.Module):
    """Uniformly sample speed per recording; keep the playback sample rate."""

    def __init__(self, factor_min: float, factor_max: float, sample_rate: int):
        super().__init__()

        self.factor_min = factor_min
        self.factor_max = factor_max
        self.sample_rate = sample_rate

    def forward(self, data: Tensor) -> Tensor:
        factor = (
            self.factor_min
            + (self.factor_max - self.factor_min) * torch.rand(()).item()
        )

        augmentation = torchaudio.transforms.SpeedPerturbation(
            orig_freq=self.sample_rate,
            factors=[factor],
        ).to(device=data.device, dtype=data.dtype)
        waveform, _ = augmentation(data)
        return waveform

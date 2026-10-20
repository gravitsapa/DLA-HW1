from collections.abc import Sequence

import torchaudio
from torch import Tensor, nn


class SpeedPerturbation(nn.Module):
    def __init__(self, factors: Sequence[float], sample_rate: int):
        super().__init__()

        self.factors = tuple(float(factor) for factor in factors)
        self.sample_rate = sample_rate

        self._aug = torchaudio.transforms.SpeedPerturbation(
            orig_freq=self.sample_rate,
            factors=self.factors,
        )

    def forward(self, data: Tensor) -> Tensor:
        self._aug.to(device=data.device, dtype=data.dtype)
        waveform, _ = self._aug(data)
        return waveform

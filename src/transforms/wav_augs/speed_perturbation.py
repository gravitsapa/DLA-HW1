import numpy as np
import torchaudio
from torch import Tensor, nn


class SpeedPerturbation(nn.Module):
    def __init__(
        self, factor_min: float, factor_max: float, n_factors: int, *args, **kwargs
    ):
        super().__init__()
        self._factors = list(map(float, np.linspace(factor_min, factor_max, n_factors)))
        self._aug = torchaudio.transforms.SpeedPerturbation(
            factors=self._factors, *args, **kwargs
        )

    def forward(self, data: Tensor):
        waveform, _ = self._aug(data)
        return waveform

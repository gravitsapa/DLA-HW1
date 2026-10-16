import torch
from torch import Tensor, nn
from torchaudio.transforms import MelSpectrogram


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        eps: float = 1.0e-5,
        *args,
        **kwargs,
    ):
        super().__init__()
        self._spec_transform = MelSpectrogram(*args, **kwargs)
        self.eps = eps

    def forward(self, data: Tensor):
        mel_spec = self._spec_transform(data)
        log_mel = torch.log(mel_spec.clamp_min(self.eps))
        return log_mel

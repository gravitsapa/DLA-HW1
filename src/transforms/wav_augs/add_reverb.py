import torch
from pedalboard import Reverb
from torch import Tensor, nn


class AddReverb(nn.Module):
    def __init__(self, sample_rate: int, *args, **kwargs):
        super().__init__()
        self.sample_rate = sample_rate
        self._aug = Reverb(
            *args,
            **kwargs,
        )

    def forward(self, data: Tensor):
        result = self._aug(data.cpu().numpy(), self.sample_rate)
        return torch.from_numpy(result).to(data.device)

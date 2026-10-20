import torch
from torch import Tensor, nn


class NormalizeSpectrogram(nn.Module):
    def __init__(self, eps: float = 1.0e-5):
        super().__init__()

        self.eps = eps

    def forward(self, data: Tensor) -> Tensor:
        values = data
        variance, mean = torch.var_mean(values, dim=-1, correction=0, keepdim=True)

        std = variance.clamp_min(self.eps**2).sqrt()
        normalized = (values - mean) / std
        return normalized.to(dtype=data.dtype)

import math

import torch
from pedalboard import Reverb
from torch import Tensor, nn


class AddReverb(nn.Module):
    """Sample independent reverb parameters once per unpadded audio recording."""

    def __init__(
        self,
        sample_rate: int,
        room_size_min: float,
        room_size_max: float,
        damping_min: float,
        damping_max: float,
        wet_level_min: float,
        wet_level_max: float,
        dry_level_min: float,
        dry_level_max: float,
        width_min: float,
        width_max: float,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.parameter_ranges = {
            "room_size": (room_size_min, room_size_max),
            "damping": (damping_min, damping_max),
            "wet_level": (wet_level_min, wet_level_max),
            "dry_level": (dry_level_min, dry_level_max),
            "width": (width_min, width_max),
        }

    def forward(self, data: Tensor) -> Tensor:
        parameters = {
            name: low + (high - low) * torch.rand(()).item()
            for name, (low, high) in self.parameter_ranges.items()
        }

        effect = Reverb(**parameters)
        result = effect(
            data.detach().to(device="cpu").numpy(),
            self.sample_rate,
        )
        return torch.from_numpy(result).to(device=data.device, dtype=data.dtype)

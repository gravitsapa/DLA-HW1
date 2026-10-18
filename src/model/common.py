from typing import Callable

from torch import nn

ModuleFactory = Callable[[], nn.Module]
type Factory[T] = Callable[..., T]

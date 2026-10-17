from typing import Callable

from torch import nn

ModuleFabric = Callable[[], nn.Module]

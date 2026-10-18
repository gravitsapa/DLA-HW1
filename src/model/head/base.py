from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from src.model.common import Factoty, ModuleFactory
from src.model.joint_network import JointNetwork
from src.model.predictor import Predictor


class Head(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        hidden_state: Tensor,
        hidden_states_length: Tensor,
        text_encoded: Tensor,
        text_encoded_length: Tensor,
    ) -> dict:
        raise NotImplementedError()

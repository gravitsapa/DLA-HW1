import torch
from torch import Tensor, nn

from src.model.common import Factoty, ModuleFactory
from src.model.head.base import Head
from src.model.joint_network import JointNetwork
from src.model.predictor import Predictor


class RNNTHead(Head):
    def __init__(
        self,
        encoder_hidden_size: int,
        n_tokens: int,
        predictor_factory: Factoty[Predictor],
        joint_hidden_size: int,
        joint_act_fn: ModuleFactory,
    ):
        super().__init__()

        self.predictor = predictor_factory(n_tokens=n_tokens)
        self.joint_net = JointNetwork(
            encoder_output_dim=encoder_hidden_size,
            predictor_output_dim=self.predictor.output_size,
            n_tokens=n_tokens,
            hidden_dim=joint_hidden_size,
            act_fn=joint_act_fn,
        )

    def forward(
        self,
        hidden_state: Tensor,
        hidden_states_length: Tensor,
        text_encoded: Tensor,
        text_encoded_length: Tensor,
    ) -> dict:
        predictor_output = self.predictor(
            text_encoded,
            text_encoded_length,
        )

        logits = self.joint_net(
            hidden_state,
            predictor_output,
        )

        return {"logits": logits, "logits_length": hidden_states_length}

import torch
from torch import Tensor, nn

from src.model.common import ModuleFactory


class JointNetwork(nn.Module):
    def __init__(
        self,
        encoder_output_dim: int,
        predictor_output_dim: int,
        n_tokens: int,
        hidden_dim: int,
        act_fn: ModuleFactory,
    ):
        super().__init__()

        self.enc_proj = nn.Linear(
            encoder_output_dim,
            hidden_dim,
        )
        self.predictor_proj = nn.Linear(
            predictor_output_dim,
            hidden_dim,
        )

        self.act = act_fn()
        self.output_proj = nn.Linear(hidden_dim, n_tokens)

    def forward(
        self,
        enc_output: Tensor,
        predictor_output: Tensor,
    ):
        enc_context = self.project_encoder(enc_output).unsqueeze(2)
        pred_context = self.project_predictor(predictor_output).unsqueeze(1)

        return self.joint_projected(enc_context, pred_context)

    def project_encoder(self, enc_output: Tensor) -> Tensor:
        return self.enc_proj(enc_output)

    def project_predictor(self, predictor_output: Tensor) -> Tensor:
        return self.predictor_proj(predictor_output)

    def joint_projected(self, enc_context: Tensor, pred_context: Tensor) -> Tensor:
        return self.output_proj(self.act(enc_context + pred_context))

    def step(self, enc_output: Tensor, predictor_output: Tensor) -> Tensor:
        return self.joint_projected(
            self.project_encoder(enc_output), self.project_predictor(predictor_output)
        )

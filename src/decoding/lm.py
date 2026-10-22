import torch
from torch import Tensor

from src.text_encoder import TextEncoder

RNN_STATE = tuple[Tensor, Tensor]
LM_STATE = tuple[Tensor, RNN_STATE]


class FlairCharLM:
    """Character LM with explicit LSTM states.

    Scores are normalized over the non-blank ASR vocabulary.
    """

    def __init__(
        self,
        text_encoder: TextEncoder,
        model_name: str = "news-forward-fast",
        device: str | torch.device = "auto",
    ):
        from flair.embeddings import FlairEmbeddings

        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        self.model = FlairEmbeddings(model_name, has_decoder=True).lm
        self.model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        if not self.model.is_forward_lm or self.model.decoder is None:
            raise ValueError(
                "A forward character LM with a prediction head is required"
            )

        self.blank_id = text_encoder.blank_id
        dictionary = self.model.dictionary
        characters = set(dictionary.get_items())
        missing = [
            char
            for i, char in enumerate(text_encoder.vocab)
            if i != self.blank_id and char not in characters
        ]
        if missing:
            raise ValueError(f"Characters missing from the LM vocabulary: {missing}")
        self.id_to_lm_token = [
            dictionary.get_idx_for_item(char) if i != self.blank_id else 0
            for i, char in enumerate(text_encoder.vocab)
        ]

    def _forward(self, tokens: list[int], state: RNN_STATE) -> LM_STATE:
        inputs = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        logits, _, next_state = self.model(inputs, state)
        scores = logits[-1, 0, self.id_to_lm_token]
        scores[self.blank_id] = -torch.inf
        return scores.log_softmax(-1).cpu(), next_state

    @torch.inference_mode()
    def get_initial_state(self) -> LM_STATE:
        delimiter = self.model.document_delimiter
        tokens = [self.model.dictionary.get_idx_for_item(char) for char in delimiter]
        return self._forward(tokens, self.model.init_hidden(1))

    @torch.inference_mode()
    def step(self, state: RNN_STATE, token: int) -> LM_STATE:
        if not 0 <= token < len(self.id_to_lm_token) or token == self.blank_id:
            raise ValueError(f"Expected a non-blank ASR token, got {token}")
        return self._forward([self.id_to_lm_token[token]], state)

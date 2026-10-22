from collections import OrderedDict

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
        cache_size: int = 4096,
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

        self.blank_id = text_encoder.blank_id
        dictionary = self.model.dictionary
        characters = set(dictionary.get_items())
        missing = [
            char
            for i, char in enumerate(text_encoder.vocab)
            if i != self.blank_id and char not in characters
        ]
        if missing:
            raise ValueError(f"characters missing: {missing}")
        self.id_to_lm_token = [
            dictionary.get_idx_for_item(char) if i != self.blank_id else 0
            for i, char in enumerate(text_encoder.vocab)
        ]

        self.cache_size = cache_size
        self._states = OrderedDict()

    def clear_cache(self):
        self._states.clear()

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
        return self._forward([self.id_to_lm_token[token]], state)

    @torch.inference_mode()
    def batch_step(self, states: list[RNN_STATE], tokens: list[int]) -> list[LM_STATE]:
        if not tokens:
            return []
        for token in tokens:
            self._validate_token(token)
        inputs = torch.tensor(
            [[self.id_to_lm_token[token] for token in tokens]],
            dtype=torch.long,
            device=self.device,
        )
        state = tuple(torch.cat([s[part] for s in states], dim=1) for part in range(2))
        logits, _, next_state = self.model(inputs, state)
        scores = logits[-1, :, self.id_to_lm_token]
        scores[:, self.blank_id] = -torch.inf
        scores = scores.log_softmax(-1).cpu()
        parts = [part.unbind(dim=1) for part in next_state]

        return [
            (scores[i], tuple(part[i].unsqueeze(1) for part in parts))
            for i in range(len(tokens))
        ]

    def get_states(
        self, states: list[RNN_STATE], prefixes: list[tuple[int, ...]]
    ) -> list[LM_STATE]:
        missing = {}
        result = [None] * len(prefixes)
        for i, prefix in enumerate(prefixes):
            if prefix in self._states:
                result[i] = self._states[prefix]
                self._states.move_to_end(prefix)
            else:
                missing.setdefault(prefix, []).append(i)
        computed = self.batch_step(
            [states[indices[0]] for indices in missing.values()],
            [prefix[-1] for prefix in missing],
        )
        for (prefix, indices), state in zip(missing.items(), computed):
            for i in indices:
                result[i] = state
            if self.cache_size:
                self._states[prefix] = state
                while len(self._states) > self.cache_size:
                    self._states.popitem(last=False)
        return result


def get_lm_states(model, states, prefixes):
    if hasattr(model, "get_states"):
        return model.get_states(states, prefixes)
    return [model.step(state, prefix[-1]) for state, prefix in zip(states, prefixes)]

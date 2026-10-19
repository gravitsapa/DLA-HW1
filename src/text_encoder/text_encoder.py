import re
from string import ascii_lowercase

import torch
from torch import Tensor


class TextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.blank_id = 0

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        return self.ind2char[item]

    def encode(self, text: str) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return (
                torch.Tensor([self.char2ind[char] for char in text])
                .unsqueeze(0)
                .to(torch.long)
            )
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode_text(
        self,
        tokens: Tensor,
        length: int,
    ) -> str:
        return "".join(self[int(token_id.item())] for token_id in tokens[:length])

    def decode_batch(
        self,
        tokens: Tensor,
        lengths: Tensor,
    ) -> list[str]:
        return [
            self.decode_text(tokens_row, int(length.item()))
            for tokens_row, length in zip(tokens, lengths, strict=True)
        ]

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

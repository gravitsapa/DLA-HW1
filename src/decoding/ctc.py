import math

import torch
from torch import Tensor

from src.decoding.base import DecoderBase, log_add
from src.decoding.lm import get_lm_states
from src.model.head.linear_head import LinearHead


class DecoderCTC(DecoderBase):
    def __init__(
        self,
        blank_id: int = 0,
        beam_size: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.blank_id = blank_id
        self.beam_size = beam_size

    @torch.inference_mode()
    def greedy_decode(
        self,
        *,
        head: LinearHead,
        hidden_state: Tensor,
        hidden_states_length: Tensor,
        **kwargs,
    ) -> dict:
        output = head(
            hidden_state=hidden_state,
            hidden_states_length=hidden_states_length,
        )
        log_probs = output["log_probs"]
        log_probs_length = output["log_probs_length"].to(device=log_probs.device)
        batch_size, time_steps, n_tokens = log_probs.shape

        predictions = log_probs.argmax(dim=-1)
        valid = torch.arange(time_steps, device=log_probs.device).unsqueeze(0)
        valid = valid < log_probs_length.unsqueeze(1)

        changed = torch.ones_like(predictions, dtype=torch.bool)
        changed[:, 1:] = predictions[:, 1:] != predictions[:, :-1]
        keep = valid & changed & (predictions != self.blank_id)
        lengths = keep.sum(dim=1)
        positions = (keep.cumsum(dim=1) - 1).clamp_min(0)

        tokens = torch.zeros_like(predictions)
        tokens.scatter_add_(1, positions, predictions.masked_fill(~keep, 0))
        padding = torch.arange(time_steps, device=log_probs.device).unsqueeze(0)
        tokens.masked_fill_(padding >= lengths.unsqueeze(1), self.blank_id)

        return {"tokens_predicted": tokens, "lengths_predicted": lengths}

    @torch.inference_mode()
    def beam_search(
        self,
        beam_size: int | None = None,
        *,
        head: LinearHead,
        hidden_state: Tensor,
        hidden_states_length: Tensor,
        _use_lm: bool = False,
        **kwargs,
    ) -> dict:
        use_lm = _use_lm and self.lm_weight != 0

        if use_lm and self.lm is None:
            raise ValueError("beam search with lm needs lm initialization")

        output = head(
            hidden_state=hidden_state,
            hidden_states_length=hidden_states_length,
        )
        log_probs = output["log_probs"]
        log_probs_length = output["log_probs_length"]
        beam_size = self.beam_size if beam_size is None else beam_size

        batch_size, time_steps, n_tokens = log_probs.shape

        input_lengths = log_probs_length.detach().cpu().tolist()

        probabilities = log_probs.detach().to(device="cpu", dtype=torch.float64)
        tokens = torch.full((batch_size, time_steps), self.blank_id, dtype=torch.long)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        negative_inf = -math.inf

        for batch_idx, length in enumerate(input_lengths):
            beam = {(): (0.0, negative_inf)}
            if use_lm:
                lm_states = {(): (self.lm.get_initial_state() if use_lm else None)}
            for frame in probabilities[batch_idx, : int(length)].tolist():
                next_beam = {}
                new_states = {}
                for prefix, (p_blank, p_nonblank) in beam.items():
                    total = log_add(p_blank, p_nonblank)
                    old_blank, old_nonblank = next_beam.get(
                        prefix, (negative_inf, negative_inf)
                    )
                    next_beam[prefix] = (
                        log_add(old_blank, total + frame[self.blank_id]),
                        old_nonblank,
                    )

                    if use_lm:
                        lm_probs, dynamic_cache = lm_states[prefix]
                        lm_probs = lm_probs.tolist()

                    for token, token_prob in enumerate(frame):
                        if token == self.blank_id or token_prob == negative_inf:
                            continue
                        if prefix and token == prefix[-1]:
                            old_blank, old_nonblank = next_beam[prefix]
                            next_beam[prefix] = (
                                old_blank,
                                log_add(old_nonblank, p_nonblank + token_prob),
                            )
                            extension_prob = p_blank + token_prob
                        else:
                            extension_prob = total + token_prob

                        if extension_prob == negative_inf:
                            continue
                        if use_lm:
                            extension_prob += self.lm_weight * lm_probs[token]
                        extension = prefix + (token,)
                        old_blank, old_nonblank = next_beam.get(
                            extension, (negative_inf, negative_inf)
                        )
                        next_beam[extension] = (
                            old_blank,
                            log_add(old_nonblank, extension_prob),
                        )
                        if use_lm and extension not in beam:
                            new_states[extension] = (dynamic_cache, token)

                beam = dict(
                    sorted(
                        next_beam.items(),
                        key=lambda item: (-log_add(*item[1]), item[0]),
                    )[:beam_size]
                )
                if use_lm:
                    missing = [prefix for prefix in beam if prefix not in lm_states]
                    states = get_lm_states(
                        self.lm, [new_states[prefix][0] for prefix in missing], missing
                    )
                    new_lm_states = dict(zip(missing, states))
                    lm_states = {
                        prefix: (
                            lm_states[prefix]
                            if prefix in lm_states
                            else new_lm_states[prefix]
                        )
                        for prefix in beam
                    }

            best_prefix = next(iter(beam))
            lengths[batch_idx] = len(best_prefix)
            tokens[batch_idx, : len(best_prefix)] = torch.tensor(
                best_prefix, dtype=torch.long
            )

        return {
            "tokens_predicted": tokens.to(device=log_probs.device),
            "lengths_predicted": lengths.to(device=log_probs.device),
        }

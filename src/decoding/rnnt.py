import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import Tensor

from src.decoding.base import DecoderBase, log_add
from src.decoding.lm import get_lm_states
from src.model.head.rnnt_head import RNNTHead


@dataclass
class _Prediction:
    context: Tensor
    state: tuple[Tensor, Tensor]


class DecoderRNNT(DecoderBase):
    def __init__(
        self,
        blank_id: int = 0,
        beam_size: int = 4,
        max_symbols_per_step: int = 5,
        max_cache_size: int = 4096,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.blank_id = blank_id
        self.beam_size = beam_size
        self.max_symbols_per_step = max_symbols_per_step
        self.max_cache_size = max_cache_size

    def _empty_result(self, hidden_state: Tensor, limit: int):
        batch_size, time_steps, _ = hidden_state.shape
        return {
            "tokens_predicted": torch.full(
                (batch_size, time_steps * limit),
                self.blank_id,
                dtype=torch.long,
                device=hidden_state.device,
            ),
            "lengths_predicted": torch.zeros(
                batch_size, dtype=torch.long, device=hidden_state.device
            ),
        }

    @torch.inference_mode()
    def greedy_decode(
        self,
        *,
        head: RNNTHead,
        hidden_state: Tensor,
        hidden_states_length: Tensor,
        max_symbols_per_step: int | None = None,
        **kwargs,
    ) -> dict:
        limit = (
            self.max_symbols_per_step
            if max_symbols_per_step is None
            else max_symbols_per_step
        )

        result = self._empty_result(hidden_state, limit)

        device = hidden_state.device
        batch_size = hidden_state.shape[0]
        input_length = hidden_states_length.to(device=device)

        enc_context = head.joint_net.project_encoder(hidden_state)
        start = torch.full(
            (batch_size,), self.blank_id, dtype=torch.long, device=device
        )
        prediction, state = head.predictor.step(start)
        pred_context = head.joint_net.project_predictor(prediction)
        time = torch.zeros(batch_size, dtype=torch.long, device=device)
        emitted_at_frame = torch.zeros_like(time)
        active = torch.arange(batch_size, device=device)[input_length > 0]

        while active.numel():
            logits = head.joint_net.joint_projected(
                enc_context[active, time[active]], pred_context[active]
            )
            selected = logits.argmax(dim=-1)
            emit = selected != self.blank_id
            rows = active[emit]
            if rows.numel():
                emitted = selected[emit]
                result["tokens_predicted"][
                    rows, result["lengths_predicted"][rows]
                ] = emitted
                result["lengths_predicted"][rows] += 1
                emitted_at_frame[rows] += 1
                previous_state = tuple(part.index_select(1, rows) for part in state)
                prediction, next_state = head.predictor.step(emitted, previous_state)
                pred_context.index_copy_(
                    0, rows, head.joint_net.project_predictor(prediction)
                )
                for previous, updated in zip(state, next_state):
                    previous.index_copy_(1, rows, updated)

            advance = active[(~emit) | (emitted_at_frame[active] >= limit)]
            time[advance] += 1
            emitted_at_frame[advance] = 0
            active = active[time[active] < input_length[active]]

        return result

    def _predictions(self, prefixes, parents, head, cache):
        predictions = [None] * len(prefixes)
        missing = []
        for index, prefix in enumerate(prefixes):
            cached = cache.get(prefix)
            if cached is None:
                missing.append(index)
            else:
                predictions[index] = cached
                cache.move_to_end(prefix)

        if missing:
            device = parents[missing[0]].context.device
            tokens = torch.tensor([prefixes[i][-1] for i in missing], device=device)
            states = tuple(
                torch.cat([parents[i].state[part] for i in missing], dim=1)
                for part in range(2)
            )
            contexts, states = head.predictor.step(tokens, states)
            contexts = head.joint_net.project_predictor(contexts)
            for batch_index, index in enumerate(missing):
                prediction = _Prediction(
                    contexts[batch_index : batch_index + 1].clone(),
                    tuple(
                        part[:, batch_index : batch_index + 1].clone()
                        for part in states
                    ),
                )
                predictions[index] = prediction
                if self.max_cache_size:
                    cache[prefixes[index]] = prediction
                    cache.move_to_end(prefixes[index])
                    while len(cache) > self.max_cache_size:
                        cache.popitem(last=False)
        return predictions

    @torch.inference_mode()
    def beam_search(
        self,
        beam_size: int | None = None,
        *,
        hidden_state: Tensor,
        hidden_states_length: Tensor,
        head: "RNNTHead",
        max_symbols_per_step: int | None = None,
        _use_lm: bool = False,
        **kwargs,
    ) -> dict:
        width = self.beam_size if beam_size is None else beam_size
        limit = (
            self.max_symbols_per_step
            if max_symbols_per_step is None
            else max_symbols_per_step
        )

        result = self._empty_result(hidden_state, limit)
        use_lm = _use_lm and self.lm_weight != 0
        if use_lm and self.lm is None:
            raise ValueError("beam search with lm needs lm initialization")

        device = hidden_state.device
        input_length = hidden_states_length.to(device)
        enc_context = head.joint_net.project_encoder(hidden_state)
        start = torch.tensor([self.blank_id], device=device)
        context, state = head.predictor.step(start)
        root = _Prediction(head.joint_net.project_predictor(context), state)

        cache = OrderedDict()
        if self.max_cache_size:
            cache[()] = root
        transcripts = []

        for batch_index, length in enumerate(input_length):
            beam = {(): (0.0, root)}
            lm_states = (
                {(): self.lm.get_initial_state()} if use_lm and int(length) > 0 else {}
            )
            for time in range(length):
                active = beam
                completed = {}
                completed_lm_states = {}
                for depth in range(limit + 1):
                    if depth == limit:
                        for prefix, (score, prediction) in active.items():
                            old_score = completed.get(prefix, (-math.inf, prediction))[
                                0
                            ]
                            completed[prefix] = (log_add(old_score, score), prediction)
                            if use_lm:
                                completed_lm_states[prefix] = lm_states[prefix]
                        break
                    prefixes = list(active)
                    contexts = torch.cat(
                        [active[prefix][1].context for prefix in prefixes]
                    )
                    logits = head.joint_net.joint_projected(
                        enc_context[batch_index, time].unsqueeze(0), contexts
                    )
                    probabilities = logits.float().log_softmax(-1).cpu().tolist()
                    candidates = {}
                    for prefix, frame in zip(prefixes, probabilities):
                        score, prediction = active[prefix]
                        lm_row = (
                            lm_states[prefix][0].detach().cpu().tolist()
                            if use_lm
                            else None
                        )
                        blank_score = score + frame[self.blank_id]
                        if blank_score != -math.inf:
                            old_score = completed.get(prefix, (-math.inf, prediction))[
                                0
                            ]
                            completed[prefix] = (
                                log_add(old_score, blank_score),
                                prediction,
                            )
                            if use_lm:
                                completed_lm_states[prefix] = lm_states[prefix]
                        for token, token_score in enumerate(frame):
                            if token == self.blank_id or token_score == -math.inf:
                                continue
                            if use_lm:
                                token_score += self.lm_weight * lm_row[token]
                            extension = prefix + (token,)
                            candidates[extension] = (score + token_score, prediction)

                    if not candidates:
                        break
                    retained = sorted(
                        candidates, key=lambda prefix: (-candidates[prefix][0], prefix)
                    )[:width]
                    predictions = self._predictions(
                        retained,
                        [candidates[prefix][1] for prefix in retained],
                        head,
                        cache,
                    )
                    if use_lm:
                        next_lm_states = {}
                        missing = []
                        for prefix in retained:
                            if prefix in lm_states:
                                next_lm_states[prefix] = lm_states[prefix]
                            elif prefix in completed_lm_states:
                                next_lm_states[prefix] = completed_lm_states[prefix]
                            else:
                                missing.append(prefix)
                        states = get_lm_states(
                            self.lm,
                            [lm_states[prefix[:-1]][1] for prefix in missing],
                            missing,
                        )
                        next_lm_states.update(zip(missing, states))
                        lm_states = next_lm_states
                    active = {
                        prefix: (candidates[prefix][0], prediction)
                        for prefix, prediction in zip(retained, predictions)
                    }

                if not completed:
                    raise RuntimeError("Cannot continue")

                retained = sorted(
                    completed, key=lambda prefix: (-completed[prefix][0], prefix)
                )[:width]
                beam = {prefix: completed[prefix] for prefix in retained}
                if use_lm:
                    lm_states = {
                        prefix: completed_lm_states[prefix] for prefix in retained
                    }
            transcripts.append(next(iter(beam)))

        tokens = torch.full(
            result["tokens_predicted"].shape, self.blank_id, dtype=torch.long
        )
        lengths = torch.tensor(
            [len(prefix) for prefix in transcripts], dtype=torch.long
        )
        for index, prefix in enumerate(transcripts):
            tokens[index, : len(prefix)] = torch.tensor(prefix, dtype=torch.long)
        return {
            "tokens_predicted": tokens.to(device),
            "lengths_predicted": lengths.to(device),
        }

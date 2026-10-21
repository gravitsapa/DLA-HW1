from contextlib import nullcontext
from string import ascii_lowercase

import torch


class NoTokenLM:
    def __init__(
        self,
        repo_id="omurberaisik/NoTokenLM-Gen-3.5",
        revision="0dda19f7b518eda876fe62b1b9c58685bc9742a9",
        cache_dir=None,
        local_files_only=False,
        vocabulary=None,
        blank_id=0,
        space_id=ord(" "),
        device=None,
        batch_size=16,
        use_cuda_graphs=True,
    ):
        from transformers import AutoModelForCausalLM

        self.vocabulary = (
            list(vocabulary)
            if vocabulary is not None
            else [""] + list(ascii_lowercase + " ")
        )
        self.blank_id = blank_id
        self.space_id = space_id

        self.token_ids = [
            0 if i == blank_id else ord(char) for i, char in enumerate(self.vocabulary)
        ]
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                repo_id,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
            .float()
            .requires_grad_(False)
            .eval()
        )

        self.context_size = self.model.config.block_size
        self.device = device
        self.batch_size = batch_size
        self.use_cuda_graphs = use_cuda_graphs
        self._model_device = next(self.model.parameters()).device
        self._graphs = {}
        self._scores = {}

    def clear_cache(self):
        self._scores.clear()

    def _graph_logits(self, rows, target):
        batch = min(self.batch_size, 1 << (len(rows) - 1).bit_length())
        length = min(
            self.context_size, 1 << max(3, (max(map(len, rows)) - 1).bit_length())
        )
        key = (batch, length)
        if key not in self._graphs:
            inputs = torch.zeros(batch, length, dtype=torch.long, device=target)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                logits = self.model(input_ids=inputs).logits
            self._graphs[key] = (inputs, graph, logits)

        inputs, graph, logits = self._graphs[key]
        packed = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(row, dtype=torch.long) for row in rows], batch_first=True
        )
        inputs.zero_()
        inputs[: len(rows), : packed.shape[1]].copy_(packed)
        graph.replay()
        return logits

    @torch.inference_mode()
    def score_prefixes(self, prefixes, *, device):
        target = torch.device(self.device or device)
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        if target != self._model_device:
            self._graphs.clear()
            self.clear_cache()
            self.model.to(target)
            self._model_device = target

        contexts = [
            tuple(
                ([self.space_id] + [self.token_ids[token] for token in prefix])[
                    -self.context_size :
                ]
            )
            for prefix in prefixes
        ]
        missing = list(
            dict.fromkeys(
                context for context in contexts if context not in self._scores
            )
        )
        with torch.cuda.device(target) if target.type == "cuda" else nullcontext():
            for start in range(0, len(missing), self.batch_size):
                rows = missing[start : start + self.batch_size]
                lengths = torch.tensor([len(row) for row in rows], device=target)
                with torch.autocast(device_type=target.type, enabled=False):
                    if self.use_cuda_graphs and target.type == "cuda":
                        logits = self._graph_logits(rows, target)
                    else:
                        ids = torch.nn.utils.rnn.pad_sequence(
                            [torch.tensor(row, dtype=torch.long) for row in rows],
                            batch_first=True,
                        ).to(target)
                        logits = self.model(input_ids=ids).logits
                    last = logits[torch.arange(len(rows), device=target), lengths - 1]
                    scores = last.float().log_softmax(-1)[:, self.token_ids]
                scores[:, self.blank_id] = 0.0
                self._scores.update(zip(rows, scores.cpu().tolist()))
        return [self._scores[context] for context in contexts]

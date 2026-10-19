from abc import ABC, abstractmethod
from typing import List

import torch

from src.decoding.base import DecodeMethod
from src.metrics.base_metric import BaseMetric
from src.text_encoder.text_encoder import TextEncoder


class TextMetricBase(BaseMetric, ABC):
    def __init__(
        self,
        decode_method: DecodeMethod,
        text_encoder: TextEncoder,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.decode_method = decode_method
        self.text_encoder = text_encoder

    @abstractmethod
    def metric_by_pair(
        self,
        predicted_text: str,
        target_text: str,
    ) -> dict:
        raise NotImplementedError()

    def __call__(
        self,
        *,
        text: List[str],
        **kwargs,
    ) -> dict:
        numerator = 0
        denominator = 0

        texts_predicted = kwargs[f"text_predicted_{str(self.decode_method)}"]

        for predicted, target in zip(texts_predicted, text, strict=True):
            target = self.text_encoder.normalize_text(target)
            fraction = self.metric_by_pair(predicted, target)
            numerator += fraction["numerator"]
            denominator += fraction["denominator"]

        return {
            "numerator": numerator,
            "denominator": denominator,
        }

    def inference_only(self) -> bool:
        return True

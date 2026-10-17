from torch import Tensor

from src.metrics.base_metric import BaseMetric


class PaddingRatioMetric(BaseMetric):
    @staticmethod
    def _fraction(data: Tensor, lengths: Tensor) -> dict:
        total = data.shape[0] * data.shape[-1]
        valid = lengths.sum().item()
        return {"numerator": total - valid, "denominator": total}

    def avg_type(self) -> str:
        return "mean"


class AudioPaddingRatioMetric(PaddingRatioMetric):
    def __call__(self, audio: Tensor, audio_length: Tensor, **kwargs) -> dict:
        return self._fraction(audio, audio_length)


class SpectrogramPaddingRatioMetric(PaddingRatioMetric):
    def __call__(
        self, spectrogram: Tensor, spectrogram_length: Tensor, **kwargs
    ) -> dict:
        return self._fraction(spectrogram, spectrogram_length)


class TextPaddingRatioMetric(PaddingRatioMetric):
    def __call__(
        self, text_encoded: Tensor, text_encoded_length: Tensor, **kwargs
    ) -> dict:
        return self._fraction(text_encoded, text_encoded_length)

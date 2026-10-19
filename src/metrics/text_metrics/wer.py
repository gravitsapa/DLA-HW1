from src.metrics.text_metrics.base import TextMetricBase
from src.metrics.utils import calc_wer_fraction


class WERMetric(TextMetricBase):
    def metric_by_pair(
        self,
        predicted_text: str,
        target_text: str,
    ) -> dict:
        return calc_wer_fraction(target_text, predicted_text)

    def avg_type(self) -> str:
        return "mean"

from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """
    Base class for all metrics
    """

    def __init__(self, name=None, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __call__(self, **batch) -> dict:
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        raise NotImplementedError()

    @abstractmethod
    def avg_type(self) -> str:
        """
        returns the way to average this metric: mean/sum
        """
        raise NotImplementedError()

    def inference_only(self) -> bool:
        return False

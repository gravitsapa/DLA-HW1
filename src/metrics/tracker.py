from typing import Iterable, Sequence

import pandas as pd

from src.metrics.utils import refined_division


class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(
        self,
        mean_keys: Iterable[str],
        sum_keys: Iterable[str],
    ):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer (WandBWriter | CometMLWriter | None): experiment tracker.
                Not used in this code version. Can be used to log metrics
                from each batch.
        """

        self._mean_data = pd.DataFrame(
            index=pd.Index(mean_keys),
            columns=["numerator", "denominator", "average"],
            dtype=float,
        )

        self._sum_data = pd.DataFrame(
            index=pd.Index(sum_keys), columns=["sum"], dtype=float
        )
        self.reset()

    def reset(self):
        """
        Reset all metrics after epoch end.
        """
        self._mean_data.loc[:, :] = 0.0
        self._sum_data.loc[:, :] = 0.0

    def update_mean(self, key, numerator, denominator=1):
        """
        Update metrics DataFrame with new value.

        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): how many times to count this value.
        """

        self._mean_data.loc[key, "numerator"] += numerator
        self._mean_data.loc[key, "denominator"] += denominator
        self._mean_data.loc[key, "average"] = refined_division(
            self._mean_data.numerator[key],
            self._mean_data.denominator[key],
        )

    def update_sum(self, key, value):
        self._sum_data.loc[key, "sum"] += value

    def update_by_type(self, avg_type, key, *args, **kwargs):
        if avg_type == "mean":
            self.update_mean(key, *args, **kwargs)
        elif avg_type == "sum":
            self.update_sum(key, *args, **kwargs)
        else:
            raise ValueError(f"Unknown metric avg type: {avg_type}")

    # def avg(self, key):
    #     """
    #     Return average value for a given metric.

    #     Args:
    #         key (str): metric name.
    #     Returns:
    #         average_value (float): average value for the metric.
    #     """
    #     return self._data.average[key]

    def result(self):
        """
        Return average value of each metric.

        Returns:
            average_metrics (dict): dict, containing average metrics
                for each metric name.
        """
        return dict(self._mean_data.average) | dict(self._sum_data["sum"])

    # def keys(self):
    #     """
    #     Return all metric names defined in the MetricTracker.

    #     Returns:
    #         metric_keys (Index): all metric names in the table.
    #     """
    #     return self._data.average.keys()

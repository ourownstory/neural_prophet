# https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#make-a-custom-logger
import collections
from typing import Any, Mapping, Optional

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only


class MetricsLogger(TensorBoardLogger):
    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.history = collections.defaultdict(list)
        self.checkpoint_path = None

    def after_save_checkpoint(self, checkpoint_callback) -> None:
        """Called after model checkpoint callback saves a new checkpoint.

        Args:
            checkpoint_callback: the model checkpoint callback instance
        """
        self.checkpoint_path = checkpoint_callback.best_model_path

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        super(MetricsLogger, self).log_metrics(metrics, step)
        # metrics is a dictionary of metric names and values
        for metric_name, metric_value in metrics.items():
            if metric_name == "hp_metric":
                pass
            elif metric_name != "epoch":
                self.history[metric_name].append(metric_value)
            else:  # case epoch. We want to avoid adding multiple times the same. It happens for multiple losses.
                if (
                    not len(self.history["epoch"]) or not self.history["epoch"][-1] == metric_value  # len == 0:
                ):  # the last values of epochs is not the one we are currently trying to add.
                    self.history["epoch"].append(metric_value)
                else:
                    pass
        return

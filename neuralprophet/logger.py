# https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#make-a-custom-logger
import collections

from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.loggers.logger import rank_zero_experiment


class MetricsLogger(Logger):
    def __init__(self):
        super().__init__()

        self.history = collections.defaultdict(list)
        self.checkpoint_path = None

    def after_save_checkpoint(self, checkpoint_callback: "ReferenceType[Checkpoint]") -> None:
        """Called after model checkpoint callback saves a new checkpoint.

        Args:
            checkpoint_callback: the model checkpoint callback instance
        """
        self.checkpoint_path = checkpoint_callback.best_model_path

    @property
    def name(self):
        return "logs"  # name of the folder to store the logs to

    @property
    def version(self):
        return ""

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        for metric_name, metric_value in metrics.items():
            if metric_name != "epoch":
                self.history[metric_name].append(metric_value)
            else:  # case epoch. We want to avoid adding multiple times the same. It happens for multiple losses.
                if (
                    not len(self.history["epoch"]) or not self.history["epoch"][-1] == metric_value  # len == 0:
                ):  # the last values of epochs is not the one we are currently trying to add.
                    self.history["epoch"].append(metric_value)
                else:
                    pass
        return

    def log_hyperparams(self, params):
        pass

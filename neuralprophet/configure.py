from collections import OrderedDict
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import logging
import inspect
import torch
import math

from neuralprophet import utils_torch

log = logging.getLogger("NP.config")


def from_kwargs(cls, kwargs):
    return cls(**{k: v for k, v in kwargs.items() if k in inspect.signature(cls).parameters})


@dataclass
class Trend:
    growth: str
    changepoints: list
    n_changepoints: int
    changepoints_range: float
    trend_reg: float
    trend_reg_threshold: (bool, float)

    def __post_init__(self):
        if self.growth not in ["off", "linear", "discontinuous"]:
            log.error("Invalid trend growth '{}'. Set to 'linear'".format(self.growth))
            self.growth = "linear"

        if self.growth == "off":
            self.changepoints = None
            self.n_changepoints = 0

        if self.changepoints is not None:
            self.n_changepoints = len(self.changepoints)
            self.changepoints = pd.to_datetime(self.changepoints).values

        if type(self.trend_reg_threshold) == bool:
            if self.trend_reg_threshold:
                self.trend_reg_threshold = 3.0 / (3.0 + (1.0 + self.trend_reg) * np.sqrt(self.n_changepoints))
                log.debug("Trend reg threshold automatically set to: {}".format(self.trend_reg_threshold))
            else:
                self.trend_reg_threshold = None
        elif self.trend_reg_threshold < 0:
            log.warning("Negative trend reg threshold set to zero.")
            self.trend_reg_threshold = None
        elif math.isclose(self.trend_reg_threshold, 0):
            self.trend_reg_threshold = None

        if self.trend_reg < 0:
            log.warning("Negative trend reg lambda set to zero.")
            self.trend_reg = 0
        if self.trend_reg > 0:
            if self.n_changepoints > 0:
                log.info("Note: Trend changepoint regularization is experimental.")
                self.trend_reg = 0.001 * self.trend_reg
            else:
                log.info("Trend reg lambda ignored due to no changepoints.")
                self.trend_reg = 0
                if self.trend_reg_threshold > 0:
                    log.info("Trend reg threshold ignored due to no changepoints.")
        else:
            if self.trend_reg_threshold is not None and self.trend_reg_threshold > 0:
                log.info("Trend reg threshold ignored due to reg lambda <= 0.")


@dataclass
class Season:
    resolution: int
    period: float
    arg: str


@dataclass
class AllSeason:
    mode: str = "additive"
    computation: str = "fourier"
    reg_lambda: float = 0
    yearly_arg: (str, bool, int) = "auto"
    weekly_arg: (str, bool, int) = "auto"
    daily_arg: (str, bool, int) = "auto"
    periods: OrderedDict = field(init=False)  # contains SeasonConfig objects

    def __post_init__(self):
        if self.reg_lambda > 0 and self.computation == "fourier":
            log.info("Note: Fourier-based seasonality regularization is experimental.")
            self.reg_lambda = 0.01 * self.reg_lambda
        self.periods = OrderedDict(
            {
                "yearly": Season(resolution=6, period=365.25, arg=self.yearly_arg),
                "weekly": Season(resolution=3, period=7, arg=self.weekly_arg),
                "daily": Season(resolution=6, period=1, arg=self.daily_arg),
            }
        )

    def append(self, name, period, resolution, arg):
        self.periods[name] = Season(resolution=resolution, period=period, arg=arg)


@dataclass
class Train:
    learning_rate: (float, None)
    epochs: (int, None)
    batch_size: (int, None)
    loss_func: (str, torch.nn.modules.loss._Loss, "typing.Callable")
    optimizer: (str, torch.optim.Optimizer)
    train_speed: (int, float, None)
    ar_sparsity: (float, None)
    reg_lambda_trend: float = None
    trend_reg_threshold: (bool, float) = None
    reg_lambda_season: float = None
    n_data: int = field(init=False)

    def __post_init__(self):
        if type(self.loss_func) == str:
            if self.loss_func.lower() in ["huber", "smoothl1", "smoothl1loss"]:
                self.loss_func = torch.nn.SmoothL1Loss()
            elif self.loss_func.lower() in ["mae", "l1", "l1loss"]:
                self.loss_func = torch.nn.L1Loss()
            elif self.loss_func.lower() in ["mse", "mseloss", "l2", "l2loss"]:
                self.loss_func = torch.nn.MSELoss()
            else:
                raise NotImplementedError("Loss function {} name not defined".format(self.loss_func))
        elif callable(self.loss_func):
            pass
        elif hasattr(torch.nn.modules.loss, self.loss_func.__class__.__name__):
            pass
        else:
            raise NotImplementedError("Loss function {} not found".format(self.loss_func))

    def set_auto_batch_epoch(
        self,
        n_data: int,
        min_batch: int = 16,
        max_batch: int = 256,
        min_epoch: int = 40,
        max_epoch: int = 400,
    ):
        assert n_data >= 1
        self.n_data = n_data
        log_data = np.log10(n_data)
        if self.batch_size is None:
            self.batch_size = 2 ** int(2 + log_data)
            self.batch_size = min(max_batch, max(min_batch, self.batch_size))
            self.batch_size = min(self.n_data, self.batch_size)
            log.info("Auto-set batch_size to {}".format(self.batch_size))
        if self.epochs is None:
            self.epochs = int(max_epoch / (1 + np.log(max(1.0, n_data / 100.0))))
            self.epochs = min(max_epoch, max(min_epoch, self.epochs))
            log.info("Auto-set epochs to {}".format(self.epochs))

    def apply_train_speed(self, batch=False, epoch=False, lr=False):
        if self.train_speed is not None and not math.isclose(self.train_speed, 0):
            if batch:
                self.batch_size = max(1, int(self.batch_size * 2 ** self.train_speed))
                self.batch_size = min(self.n_data, self.batch_size)
                log.info(
                    "train_speed-{} {}creased batch_size to {}".format(
                        self.train_speed, ["in", "de"][int(self.train_speed < 0)], self.batch_size
                    )
                )
            if epoch:
                self.epochs = max(1, int(self.epochs * 2 ** -self.train_speed))
                log.info(
                    "train_speed-{} {}creased epochs to {}".format(
                        self.train_speed, ["in", "de"][int(self.train_speed > 0)], self.epochs
                    )
                )
            if lr:
                self.learning_rate = self.learning_rate * 2 ** self.train_speed
                log.info(
                    "train_speed-{} {}creased learning_rate to {}".format(
                        self.train_speed, ["in", "de"][int(self.train_speed < 0)], self.learning_rate
                    )
                )

    def apply_train_speed_all(self):
        if self.train_speed is not None and not math.isclose(self.train_speed, 0):
            self.apply_train_speed(batch=True, epoch=True, lr=True)

    def get_optimizer(self, model_parameters):
        return utils_torch.create_optimizer(self.optimizer, model_parameters, self.learning_rate)

    def get_scheduler(self, optimizer, steps_per_epoch):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3333,
            anneal_strategy="cos",
            div_factor=100.0,
            final_div_factor=4000.0,
        )

    def get_reg_delay_weight(self, e, iter_progress, reg_start_pct: float = 0.6666, reg_full_pct: float = 1):
        progress = (e + iter_progress) / float(self.epochs)
        if reg_start_pct == reg_full_pct:
            reg_progress = float(progress > reg_start_pct)
        else:
            reg_progress = (progress - reg_start_pct) / (reg_full_pct - reg_start_pct)
        if reg_progress <= 0:
            delay_weight = 0
        elif reg_progress < 1:
            delay_weight = 1 - (1 + np.cos(np.pi * reg_progress)) / 2.0
        else:
            delay_weight = 1
        return delay_weight


@dataclass
class Model:
    num_hidden_layers: int
    d_hidden: int


@dataclass
class Covar:
    reg_lambda: float
    as_scalar: bool
    normalize: (bool, str)

    def __post_init__(self):
        if self.reg_lambda is not None:
            if self.reg_lambda < 0:
                raise ValueError("regularization must be >= 0")


@dataclass
class AR:
    n_lags: int
    ar_sparsity: float

    def __post_init__(self):
        if self.ar_sparsity is not None and self.ar_sparsity < 1:
            assert self.ar_sparsity > 0
            self.reg_lambda = 0.001 * (1.0 / (1e-6 + self.ar_sparsity) - 1.00)
        else:
            self.reg_lambda = None

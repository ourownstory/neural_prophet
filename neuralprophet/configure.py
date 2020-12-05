from collections import OrderedDict
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import logging
import inspect
import torch
import math

log = logging.getLogger("nprophet.config")


def from_kwargs(cls, kwargs):
    return cls(**{k: v for k, v in kwargs.items() if k in inspect.signature(cls).parameters})


@dataclass
class Trend:
    growth: str
    changepoints: (list, np.array)
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
            if self.trend_reg_threshold > 0 or self.trend_reg_threshold is True:
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
    learning_rate: float
    epochs: int
    batch_size: int
    loss_func: (str, torch.nn.modules.loss._Loss)
    ar_sparsity: float
    reg_delay_pct: float = 0.5
    lambda_delay: int = field(init=False)
    reg_lambda_trend: float = None
    trend_reg_threshold: (bool, float) = None
    reg_lambda_season: float = None

    def __post_init__(self):
        if self.epochs is not None:
            self.lambda_delay = int(self.reg_delay_pct * self.epochs)
        if type(self.loss_func) == str:
            if self.loss_func.lower() in ["huber", "smoothl1", "smoothl1loss"]:
                self.loss_func = torch.nn.SmoothL1Loss()
            elif self.loss_func.lower() in ["mae", "l1", "l1loss"]:
                self.loss_func = torch.nn.L1Loss()
            elif self.loss_func.lower() in ["mse", "mseloss", "l2", "l2loss"]:
                self.loss_func = torch.nn.MSELoss()
            else:
                raise NotImplementedError("Loss function {} name not defined".format(self.loss_func))
        elif hasattr(torch.nn.modules.loss, self.loss_func.__class__.__name__):
            pass
        else:
            raise NotImplementedError("Loss function {} not found".format(self.loss_func))

    def set_auto_batch_epoch(self, n_data):
        pass

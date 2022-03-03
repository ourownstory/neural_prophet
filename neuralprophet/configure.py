from collections import OrderedDict
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import logging
import inspect
import torch
import math
import types

from neuralprophet import utils_torch, utils, df_utils

log = logging.getLogger("NP.config")


def from_kwargs(cls, kwargs):
    return cls(**{k: v for k, v in kwargs.items() if k in inspect.signature(cls).parameters})


@dataclass
class Model:
    num_hidden_layers: int
    d_hidden: int


@dataclass
class Normalization:
    normalize: str
    global_normalization: bool
    global_time_normalization: bool
    unknown_data_normalization: bool
    local_data_params: dict = None  # nested dict (key1: name of dataset, key2: name of variable)
    global_data_params: dict = None  # dict where keys are names of variables

    def init_data_params(self, df_dict, covariates_config=None, regressor_config=None, events_config=None):
        if len(df_dict) == 1:
            if not self.global_normalization:
                log.info("Setting normalization to global as only one dataframe provided for training.")
                self.global_normalization = True
        self.local_data_params, self.global_data_params = df_utils.init_data_params(
            df_dict=df_dict,
            normalize=self.normalize,
            covariates_config=covariates_config,
            regressor_config=regressor_config,
            events_config=events_config,
            global_normalization=self.global_normalization,
            global_time_normalization=self.global_normalization,
        )

    def get_data_params(self, df_name):
        if self.global_normalization:
            data_params = self.global_data_params
        else:
            if df_name in self.local_data_params.keys() and df_name != "__df__":
                log.debug("Dataset name {name!r} found in training data_params".format(name=df_name))
                data_params = self.local_data_params[df_name]
            elif self.unknown_data_normalization:
                log.debug(
                    "Dataset name {name!r} is not present in valid data_params but unknown_data_normalization is True. Using global_data_params".format(
                        name=df_name
                    )
                )
                data_params = self.global_data_params
            else:
                raise ValueError(
                    "Dataset name {name!r} missing from training data params. Set unkown_data_normalization to use global (average) normalization parameters.".format(
                        name=df_name
                    )
                )
        return data_params


@dataclass
class Train:
    learning_rate: (float, None)
    epochs: (int, None)
    batch_size: (int, None)
    loss_func: (str, torch.nn.modules.loss._Loss, "typing.Callable")
    optimizer: (str, torch.optim.Optimizer)
    newer_samples_weight: float = 1.0
    newer_samples_start: float = 0.0
    ar_sparsity: (float, None) = None
    reg_delay_pct: float = 0.5
    reg_lambda_trend: float = None
    trend_reg_threshold: (bool, float) = None
    reg_lambda_season: float = None
    n_data: int = field(init=False)
    loss_func_name: str = field(init=False)

    def __post_init__(self):
        assert self.newer_samples_weight >= 1.0
        assert self.newer_samples_start >= 0.0
        assert self.newer_samples_start < 1.0
        if type(self.loss_func) == str:
            if self.loss_func.lower() in ["huber", "smoothl1", "smoothl1loss"]:
                self.loss_func = torch.nn.SmoothL1Loss(reduction="none")
            elif self.loss_func.lower() in ["mae", "l1", "l1loss"]:
                self.loss_func = torch.nn.L1Loss(reduction="none")
            elif self.loss_func.lower() in ["mse", "mseloss", "l2", "l2loss"]:
                self.loss_func = torch.nn.MSELoss(reduction="none")
            else:
                raise NotImplementedError("Loss function {} name not defined".format(self.loss_func))
            self.loss_func_name = type(self.loss_func).__name__
        else:
            if callable(self.loss_func) and isinstance(self.loss_func, types.FunctionType):
                self.loss_func_name = self.loss_func.__name__
            elif issubclass(self.loss_func().__class__, torch.nn.modules.loss._Loss):
                self.loss_func = self.loss_func(reduction="none")
                self.loss_func_name = type(self.loss_func).__name__
            else:
                raise NotImplementedError("Loss function {} not found".format(self.loss_func))

    def set_auto_batch_epoch(
        self,
        n_data: int,
        min_batch: int = 16,
        max_batch: int = 512,
        min_epoch: int = 10,
        max_epoch: int = 1000,
    ):
        assert n_data >= 1
        self.n_data = n_data
        if self.batch_size is None:
            self.batch_size = int(2 ** (2 + int(np.log10(n_data))))
            self.batch_size = min(max_batch, max(min_batch, self.batch_size))
            self.batch_size = min(self.n_data, self.batch_size)
            log.info("Auto-set batch_size to {}".format(self.batch_size))
        if self.epochs is None:
            # this should (with auto batch size) yield about 1000 steps minimum and 100,000 steps at upper cutoff
            self.epochs = int(2 ** (2.3 * np.log10(100 + n_data)) / (n_data / 1000.0))
            self.epochs = min(max_epoch, max(min_epoch, self.epochs))
            log.info("Auto-set epochs to {}".format(self.epochs))
        # also set lambda_delay:
        self.lambda_delay = int(self.reg_delay_pct * self.epochs)

    def get_optimizer(self, model_parameters):
        return utils_torch.create_optimizer_from_config(self.optimizer, model_parameters, self.learning_rate)

    def get_scheduler(self, optimizer, steps_per_epoch):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=100.0,
            final_div_factor=5000.0,
        )

    def get_reg_delay_weight(self, e, iter_progress, reg_start_pct: float = 0.5, reg_full_pct: float = 1.0):
        progress = (e + iter_progress) / float(self.epochs)
        if reg_start_pct == reg_full_pct:
            reg_progress = float(progress > reg_start_pct)
        else:
            reg_progress = (progress - reg_start_pct) / (reg_full_pct - reg_start_pct)
        if reg_progress <= 0:
            delay_weight = 0
        elif reg_progress < 1:
            delay_weight = 1 - (1 + np.cos(np.pi * float(reg_progress))) / 2.0
        else:
            delay_weight = 1
        return delay_weight

    def find_learning_rate(self, model, dataset, repeat: int = 3):
        # return 0.1
        if issubclass(self.loss_func.__class__, torch.nn.modules.loss._Loss):
            try:
                loss_func = getattr(torch.nn.modules.loss, self.loss_func_name)()
            except AttributeError:
                raise ValueError("automatic learning rate only supported for regular torch loss functions.")
        else:
            raise ValueError("automatic learning rate only supported for regular torch loss functions.")
        lrs = []
        for i in range(repeat):
            lr = utils_torch.lr_range_test(
                model,
                dataset,
                loss_func=loss_func,
                optimizer=self.optimizer,
                batch_size=self.batch_size,
            )
            lrs.append(lr)
        lrs_log10_mean = sum([np.log10(x) for x in lrs]) / repeat
        learning_rate = 10 ** lrs_log10_mean
        return learning_rate


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
class AR:
    n_lags: int
    ar_sparsity: float

    def __post_init__(self):
        if self.ar_sparsity is not None and self.ar_sparsity < 1:
            assert self.ar_sparsity > 0
            self.reg_lambda = 0.001 * (1.0 / (1e-6 + self.ar_sparsity) - 1.00)
        else:
            self.reg_lambda = None

    def regularize(self, weights, original=False):
        """Regularization of AR coefficients
        Args:
            weights (torch tensor): Model weights to be regularized towards zero
        Returns:
            regularization loss, scalar
        """
        if original:
            reg = torch.div(2.0, 1.0 + torch.exp(-2 * (1e-9 + torch.abs(weights)).pow(1 / 2.0))) - 1.0
        else:
            reg = utils_torch.penalize_nonzero(weights, eagerness=3, acceptance=1.0)
        return reg


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
class Regressor:
    reg_lambda: float
    normalize: str
    mode: str


@dataclass
class Event:
    lower_window: int
    upper_window: int
    reg_lambda: float
    mode: str


@dataclass
class Holidays:
    country: str
    lower_window: int
    upper_window: int
    mode: str = "additive"
    reg_lambda: float = None
    holiday_names: set = field(init=False)

    def init_holidays(self, df=None):
        self.holiday_names = utils.get_holidays_from_country(self.country, df)

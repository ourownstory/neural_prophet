from __future__ import annotations

import logging
import math
import types
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Type, Union

import numpy as np
import pandas as pd
import torch

from neuralprophet import df_utils, np_types, utils, utils_torch
from neuralprophet.custom_loss_metrics import PinballLoss

log = logging.getLogger("NP.config")


@dataclass
class Model:
    num_hidden_layers: int
    d_hidden: Optional[int]


@dataclass
class Normalization:
    normalize: str
    global_normalization: bool
    global_time_normalization: bool
    unknown_data_normalization: bool
    local_data_params: dict = field(default_factory=dict)  # nested dict (key1: name of dataset, key2: name of variable)
    global_data_params: dict = field(default_factory=dict)  # dict where keys are names of variables

    def init_data_params(
        self,
        df,
        config_lagged_regressors: Optional[ConfigLaggedRegressors] = None,
        config_regressors=None,
        config_events: Optional[ConfigEvents] = None,
    ):
        if len(df["ID"].unique()) == 1:
            if not self.global_normalization:
                log.info("Setting normalization to global as only one dataframe provided for training.")
                self.global_normalization = True
        self.local_data_params, self.global_data_params = df_utils.init_data_params(
            df=df,
            normalize=self.normalize,
            config_lagged_regressors=config_lagged_regressors,
            config_regressors=config_regressors,
            config_events=config_events,
            global_normalization=self.global_normalization,
            global_time_normalization=self.global_normalization,
        )

    def get_data_params(self, df_name):
        if self.global_normalization:
            data_params = self.global_data_params
        else:
            if df_name in self.local_data_params.keys() and df_name != "__df__":
                log.debug(f"Dataset name {df_name!r} found in training data_params")
                data_params = self.local_data_params[df_name]
            elif self.unknown_data_normalization:
                log.debug(
                    f"Dataset name {df_name!r} is not present in valid data_params but unknown_data_normalization is True. Using global_data_params"
                )
                data_params = self.global_data_params
            else:
                raise ValueError(
                    f"Dataset name {df_name!r} missing from training data params. Set unknown_data_normalization to use global (average) normalization parameters."
                )
        return data_params


@dataclass
class MissingDataHandling:
    impute_missing: bool = True
    impute_linear: int = 10
    impute_rolling: int = 10
    drop_missing: bool = False


@dataclass
class Train:
    learning_rate: Optional[float]
    epochs: Optional[int]
    batch_size: Optional[int]
    loss_func: Union[str, torch.nn.modules.loss._Loss, Callable]
    optimizer: Union[str, Type[torch.optim.Optimizer]]
    quantiles: List[float] = field(default_factory=list)
    optimizer_args: dict = field(default_factory=dict)
    scheduler: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None
    scheduler_args: dict = field(default_factory=dict)
    newer_samples_weight: float = 1.0
    newer_samples_start: float = 0.0
    reg_delay_pct: float = 0.5
    reg_lambda_trend: Optional[float] = None
    trend_reg_threshold: Optional[Union[bool, float]] = None
    reg_lambda_season: Optional[float] = None
    n_data: int = field(init=False)
    loss_func_name: str = field(init=False)
    early_stopping: bool = False
    lr_finder_args: dict = field(default_factory=dict)

    def __post_init__(self):
        # assert the uncertainty estimation params and then finalize the quantiles
        self.set_quantiles()
        assert self.newer_samples_weight >= 1.0
        assert self.newer_samples_start >= 0.0
        assert self.newer_samples_start < 1.0
        self.set_loss_func()
        self.set_optimizer()
        self.set_scheduler()

    def set_loss_func(self):
        if isinstance(self.loss_func, str):
            if self.loss_func.lower() in ["huber", "smoothl1", "smoothl1loss"]:
                self.loss_func = torch.nn.SmoothL1Loss(reduction="none")
            elif self.loss_func.lower() in ["mae", "l1", "l1loss"]:
                self.loss_func = torch.nn.L1Loss(reduction="none")
            elif self.loss_func.lower() in ["mse", "mseloss", "l2", "l2loss"]:
                self.loss_func = torch.nn.MSELoss(reduction="none")
            else:
                raise NotImplementedError(f"Loss function {self.loss_func} name not defined")
            self.loss_func_name = type(self.loss_func).__name__
        else:
            if callable(self.loss_func) and isinstance(self.loss_func, types.FunctionType):
                self.loss_func_name = self.loss_func.__name__
            elif issubclass(self.loss_func().__class__, torch.nn.modules.loss._Loss):
                self.loss_func = self.loss_func(reduction="none")
                self.loss_func_name = type(self.loss_func).__name__
            else:
                raise NotImplementedError(f"Loss function {self.loss_func} not found")
        if len(self.quantiles) > 1:
            self.loss_func = PinballLoss(loss_func=self.loss_func, quantiles=self.quantiles)

    def set_quantiles(self):
        # convert quantiles to empty list [] if None
        if self.quantiles is None:
            self.quantiles = []
        # assert quantiles is a list type
        assert isinstance(self.quantiles, list), "Quantiles must be in a list format, not None or scalar."
        # check if quantiles contain 0.5 or close to 0.5, remove if so as 0.5 will be inserted again as first index
        self.quantiles = [quantile for quantile in self.quantiles if not math.isclose(0.5, quantile)]
        # check if quantiles are float values in (0, 1)
        assert all(
            0 < quantile < 1 for quantile in self.quantiles
        ), "The quantiles specified need to be floats in-between (0, 1)."
        # sort the quantiles
        self.quantiles.sort()
        # 0 is the median quantile index
        self.quantiles.insert(0, 0.5)

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
            log.info(f"Auto-set batch_size to {self.batch_size}")
        if self.epochs is None:
            # this should (with auto batch size) yield about 1000 steps minimum and 100,000 steps at upper cutoff
            self.epochs = int(2 ** (2.5 * np.log10(100 + n_data)) / (n_data / 1000.0))
            self.epochs = min(max_epoch, max(min_epoch, self.epochs))
            log.info(f"Auto-set epochs to {self.epochs}")
        # also set lambda_delay:
        self.lambda_delay = int(self.reg_delay_pct * self.epochs)

    def set_optimizer(self):
        """
        Set the optimizer and optimizer args. If optimizer is a string, then it will be converted to the corresponding torch optimizer.
        The optimizer is not initialized yet as this is done in configure_optimizers in TimeNet.
        """
        self.optimizer, self.optimizer_args = utils_torch.create_optimizer_from_config(
            self.optimizer, self.optimizer_args
        )

    def set_scheduler(self):
        """
        Set the scheduler and scheduler args.
        The scheduler is not initialized yet as this is done in configure_optimizers in TimeNet.
        """
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR
        self.scheduler_args.update(
            {
                "pct_start": 0.3,
                "anneal_strategy": "cos",
                "div_factor": 100.0,
                "final_div_factor": 5000.0,
            }
        )

    def set_lr_finder_args(self, dataset_size, num_batches):
        """
        Set the lr_finder_args.
        This is the range of learning rates to test.
        """
        num_training = 150 + int(np.log10(100 + dataset_size) * 25)
        if num_batches < num_training:
            log.warning(
                f"Learning rate finder: The number of batches ({num_batches}) is too small than the required number for the learning rate finder ({num_training}). The results might not be optimal."
            )
            # num_training = num_batches
        self.lr_finder_args.update(
            {
                "min_lr": 1e-6,
                "max_lr": 10,
                "num_training": num_training,
                "early_stop_threshold": None,
            }
        )

    def get_reg_delay_weight(self, e, iter_progress, reg_start_pct: float = 0.66, reg_full_pct: float = 1.0):
        # Ignore type warning of epochs possibly being None (does not work with dataclasses)
        progress = (e + iter_progress) / float(self.epochs)  # type: ignore
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


@dataclass
class Trend:
    growth: np_types.GrowthMode
    changepoints: Optional[list]
    n_changepoints: int
    changepoints_range: float
    trend_reg: float
    trend_reg_threshold: Optional[Union[bool, float]]
    trend_global_local: str

    def __post_init__(self):
        if self.growth not in ["off", "linear", "discontinuous"]:
            log.error(f"Invalid trend growth '{self.growth}'. Set to 'linear'")
            self.growth = "linear"

        if self.growth == "off":
            self.changepoints = None
            self.n_changepoints = 0

        if self.changepoints is not None:
            self.n_changepoints = len(self.changepoints)
            self.changepoints = pd.to_datetime(self.changepoints).sort_values().values

        if self.trend_reg_threshold is None:
            pass
        elif isinstance(self.trend_reg_threshold, bool):
            if self.trend_reg_threshold:
                self.trend_reg_threshold = 3.0 / (3.0 + (1.0 + self.trend_reg) * np.sqrt(self.n_changepoints))
                log.debug(f"Trend reg threshold automatically set to: {self.trend_reg_threshold}")
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
                if self.trend_reg_threshold and self.trend_reg_threshold > 0:
                    log.info("Trend reg threshold ignored due to no changepoints.")
        else:
            if self.trend_reg_threshold is not None and self.trend_reg_threshold > 0:
                log.info("Trend reg threshold ignored due to reg lambda <= 0.")

        # If trend_global_local is not in the expected set, set to "global"
        if self.trend_global_local not in ["global", "local"]:
            log.error("Invalid global_local mode '{}'. Set to 'global'".format(self.trend_global_local))
            self.trend_global_local = "global"

        # If growth is off we want set to "global"
        if (self.growth == "off") and (self.trend_global_local == "local"):
            log.error("Invalid growth for global_local mode '{}'. Set to 'global'".format(self.trend_global_local))
            self.trend_global_local = "global"


@dataclass
class Season:
    resolution: int
    period: float
    arg: np_types.SeasonalityArgument


@dataclass
class AllSeason:
    mode: np_types.SeasonalityMode = "additive"
    computation: str = "fourier"
    reg_lambda: float = 0
    yearly_arg: np_types.SeasonalityArgument = "auto"
    weekly_arg: np_types.SeasonalityArgument = "auto"
    daily_arg: np_types.SeasonalityArgument = "auto"
    periods: OrderedDict = field(init=False)  # contains SeasonConfig objects
    global_local: np_types.SeasonGlobalLocalMode = "local"

    def __post_init__(self):
        if self.reg_lambda > 0 and self.computation == "fourier":
            log.info("Note: Fourier-based seasonality regularization is experimental.")
            self.reg_lambda = 0.001 * self.reg_lambda
        self.periods = OrderedDict(
            {
                "yearly": Season(resolution=6, period=365.25, arg=self.yearly_arg),
                "weekly": Season(resolution=3, period=7, arg=self.weekly_arg),
                "daily": Season(resolution=6, period=1, arg=self.daily_arg),
            }
        )

        # If global_local is not in the expected set, set to "global"
        if self.global_local not in ["global", "local"]:
            log.error("Invalid global_local mode '{}'. Set to 'global'".format(self.global_local))
            self.global_local = "global"

    def append(self, name, period, resolution, arg):
        self.periods[name] = Season(resolution=resolution, period=period, arg=arg)


@dataclass
class AR:
    n_lags: int
    ar_reg: Optional[float] = None

    def __post_init__(self):
        if self.ar_reg is not None and self.ar_reg > 0:
            if self.ar_reg < 0:
                raise ValueError("regularization must be >= 0")
            self.reg_lambda = 0.0001 * self.ar_reg
        else:
            self.reg_lambda = None

    def regularize(self, weights, original=False):
        """Regularization of AR coefficients

        Parameters
        ----------
            weights : torch.Tensor
                Model weights to be regularized towards zero
            original : bool
                Do not penalize non-zeros

        Returns
        -------
            numeric
                Regularization loss
        """

        if original:
            reg = torch.div(2.0, 1.0 + torch.exp(-2 * (1e-9 + torch.abs(weights)).pow(1 / 2.0))) - 1.0
        else:
            reg = utils_torch.penalize_nonzero(weights, eagerness=3, acceptance=1.0)
        return reg


@dataclass
class LaggedRegressor:
    reg_lambda: Optional[float]
    as_scalar: bool
    normalize: Union[bool, str]
    n_lags: int

    def __post_init__(self):
        if self.reg_lambda is not None:
            if self.reg_lambda < 0:
                raise ValueError("regularization must be >= 0")


ConfigLaggedRegressors = OrderedDictType[str, LaggedRegressor]


@dataclass
class Regressor:
    reg_lambda: Optional[float]
    normalize: str
    mode: str


ConfigFutureRegressors = OrderedDictType[str, Regressor]


@dataclass
class Event:
    lower_window: int
    upper_window: int
    reg_lambda: Optional[float]
    mode: str


ConfigEvents = OrderedDictType[str, Event]


@dataclass
class Holidays:
    country: str
    lower_window: int
    upper_window: int
    mode: str = "additive"
    reg_lambda: Optional[float] = None
    holiday_names: set = field(init=False)

    def init_holidays(self, df=None):
        self.holiday_names = utils.get_holidays_from_country(self.country, df)


ConfigCountryHolidays = Holidays


@dataclass
class Conformal:
    method: str
    q_hats: list


ConfigConformalPrediction = Conformal

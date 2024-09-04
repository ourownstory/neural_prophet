from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Union

import numpy as np
import pandas as pd
import torch

from neuralprophet import np_types, utils_torch
from neuralprophet.event_utils import get_holiday_names

log = logging.getLogger("NP.config_components")


@dataclass
class Trend:
    growth: np_types.GrowthMode
    changepoints: Optional[list]
    n_changepoints: int
    changepoints_range: float
    trend_reg: float
    trend_reg_threshold: Optional[Union[bool, float]]
    trend_global_local: str
    trend_local_reg: Optional[Union[bool, float]] = None

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

        if self.trend_local_reg < 0:
            log.error("Invalid  negative trend_local_reg '{}'. Set to False".format(self.trend_local_reg))
            self.trend_local_reg = False

        if self.trend_local_reg is True:
            log.error("trend_local_reg = True. Default trend_local_reg value set to 1")
            self.trend_local_reg = 1

        # If Trend modelling is global but local regularization is set.
        if self.trend_global_local == "global" and self.trend_local_reg:
            log.error("Trend modeling is '{}'. Setting the trend_local_reg to False".format(self.trend_global_local))
            self.trend_local_reg = False


@dataclass
class SingleSeasonality:
    resolution: int
    period: float
    arg: np_types.SeasonalityArgument
    condition_name: Optional[str]
    global_local: np_types.SeasonGlobalLocalMode = "local"


@dataclass
class Seasonalities:
    mode: np_types.SeasonalityMode = "additive"
    computation: str = "fourier"
    reg_lambda: float = 0
    yearly_arg: np_types.SeasonalityArgument = "auto"
    weekly_arg: np_types.SeasonalityArgument = "auto"
    daily_arg: np_types.SeasonalityArgument = "auto"
    periods: OrderedDict = field(init=False)  # contains SeasonConfig objects
    global_local: np_types.SeasonGlobalLocalMode = "global"
    seasonality_local_reg: Optional[Union[bool, float]] = None
    yearly_global_local: np_types.SeasonalityArgument = "auto"
    weekly_global_local: np_types.SeasonalityArgument = "auto"
    daily_global_local: np_types.SeasonalityArgument = "auto"
    condition_name: Optional[str] = None

    def __post_init__(self):
        if self.reg_lambda > 0 and self.computation == "fourier":
            log.info("Note: Fourier-based seasonality regularization is experimental.")
            self.reg_lambda = 0.001 * self.reg_lambda

        # If global_local is not in the expected set, set to "global"
        if self.global_local not in ["global", "local"]:
            log.error("Invalid global_local mode '{}'. Set to 'global'".format(self.global_local))
            self.global_local = "global"

        self.periods = OrderedDict(
            {
                "yearly": SingleSeasonality(
                    resolution=6,
                    period=365.25,
                    arg=self.yearly_arg,
                    global_local=(
                        self.yearly_global_local
                        if self.yearly_global_local in ["global", "local"]
                        else self.global_local
                    ),
                    condition_name=None,
                ),
                "weekly": SingleSeasonality(
                    resolution=3,
                    period=7,
                    arg=self.weekly_arg,
                    global_local=(
                        self.weekly_global_local
                        if self.weekly_global_local in ["global", "local"]
                        else self.global_local
                    ),
                    condition_name=None,
                ),
                "daily": SingleSeasonality(
                    resolution=6,
                    period=1,
                    arg=self.daily_arg,
                    global_local=(
                        self.daily_global_local if self.daily_global_local in ["global", "local"] else self.global_local
                    ),
                    condition_name=None,
                ),
            }
        )

        assert self.seasonality_local_reg >= 0, "Invalid seasonality_local_reg '{}'.".format(self.seasonality_local_reg)

        if self.seasonality_local_reg is True:
            log.warning("seasonality_local_reg = True. Default seasonality_local_reg value set to 1")
            self.seasonality_local_reg = 1

        # If Season modelling is global but local regularization is set.
        if self.global_local == "global" and self.seasonality_local_reg:
            log.error(
                "Seasonality modeling is '{}'. Setting the seasonality_local_reg to False".format(self.global_local)
            )
            self.seasonality_local_reg = False

    def append(self, name, period, resolution, arg, condition_name, global_local="auto"):
        self.periods[name] = SingleSeasonality(
            resolution=resolution,
            period=period,
            arg=arg,
            global_local=global_local if global_local in ["global", "local"] else self.global_local,
            condition_name=condition_name,
        )


@dataclass
class AutoregRession:
    n_lags: int
    ar_reg: Optional[float] = None
    ar_layers: Optional[List[int]] = None

    def __post_init__(self):
        if self.ar_reg is not None and self.n_lags == 0:
            raise ValueError("AR regularization is set, but n_lags is 0. Please set n_lags to a positive integer.")
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
class SingleLaggedRegressor:
    reg_lambda: Optional[float]
    as_scalar: bool
    normalize: Union[bool, str]
    n_lags: int

    def __post_init__(self):
        if self.reg_lambda is not None:
            if self.reg_lambda < 0:
                raise ValueError("regularization must be >= 0")


@dataclass
class LaggedRegressors:
    layers: Optional[List[int]] = field(default_factory=list)
    # List of hidden layers for shared NN across LaggedReg. The default value is ``[]``, which initializes no hidden layers.
    regressors: OrderedDict[SingleLaggedRegressor] = field(init=False)

    def __post_init__(self):
        self.regressors = OrderedDict()


@dataclass
class SingleFutureRegressor:
    reg_lambda: Optional[float]
    normalize: Union[str, bool]
    mode: str


@dataclass
class FutureRegressors:
    model: str
    regressors_layers: Optional[List[int]]
    regressors: OrderedDict[SingleFutureRegressor] = field(init=False)  # contains Regressor objects

    def __post_init__(self):
        self.regressors = OrderedDict()


@dataclass
class SingleEvent:
    lower_window: int
    upper_window: int
    reg_lambda: Optional[float]
    mode: str


# TODO: convert to dataclass
Events = OrderedDictType[str, SingleEvent]


@dataclass
class Holidays:
    country: Union[str, List[str], dict]
    lower_window: int
    upper_window: int
    mode: str = "additive"
    reg_lambda: Optional[float] = None
    holiday_names: set = field(init=False)

    def init_holidays(self, df=None):
        self.holiday_names = get_holiday_names(self.country, df)

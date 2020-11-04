from collections import OrderedDict
from dataclasses import dataclass, field
import numpy as np
import logging

log = logging.getLogger("nprophet.config")


@dataclass
class Trend:
    growth: str = "linear"
    changepoints: np.array = None
    n_changepoints: int = 5
    cp_range: float = 0.8
    reg_lambda: float = 0
    reg_threshold: (bool, float) = False

    def __post_init__(self):
        if self.growth not in ["off", "linear", "discontinuous"]:
            log.error("Invalid trend growth '{}'. Set to 'linear'".format(self.growth))
            self.growth = "linear"

        if self.reg_threshold is False:
            self.reg_threshold = 0
        elif self.reg_threshold is True:
            self.reg_threshold = 3.0 / (3.0 + (1.0 + self.reg_lambda) * np.sqrt(self.n_changepoints))
            log.debug("Trend reg threshold automatically set to: {}".format(self.reg_threshold))
        elif self.reg_threshold < 0:
            log.warning("Negative trend reg threshold set to zero.")
            self.reg_threshold = 0

        if self.reg_lambda < 0:
            log.warning("Negative trend reg lambda set to zero.")
            self.reg_lambda = 0
        if self.reg_lambda > 0:
            if self.n_changepoints > 0:
                log.info("Note: Trend changepoint regularization is experimental.")
                self.reg_lambda = 0.01 * self.reg_lambda
            else:
                log.info("Trend reg lambda ignored due to no changepoints.")
                self.reg_lambda = 0
                if self.reg_threshold > 0:
                    log.info("Trend reg threshold ignored due to no changepoints.")
        else:
            if self.reg_threshold > 0 or self.reg_threshold is True:
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
            log.warning("Note: Fourier-based seasonality regularization is experimental.")
            self.reg_lambda = 0.1 * self.reg_lambda
        self.periods = OrderedDict(
            {
                "yearly": Season(resolution=6, period=365.25, arg=self.yearly_arg),
                "weekly": Season(resolution=4, period=7, arg=self.weekly_arg),
                "daily": Season(resolution=6, period=1, arg=self.daily_arg),
            }
        )

    def append(self, name, period, resolution, arg):
        self.periods[name] = Season(resolution=resolution, period=period, arg=arg)

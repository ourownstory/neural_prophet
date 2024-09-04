from __future__ import annotations

import logging
import math
import types
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch

from neuralprophet import configure_components, df_utils
from neuralprophet.custom_loss_metrics import PinballLoss

log = logging.getLogger("NP.config")


@dataclass
class Model:
    n_forecasts: int
    quantiles: Optional[List[float]] = None
    prediction_frequency: Optional[Dict[str]] = None
    max_lags: Optional[int] = field(init=False)

    def setup_quantiles(self):
        # convert quantiles to empty list [] if None
        if self.quantiles is None:
            self.quantiles = []
        # assert quantiles is a list type
        assert isinstance(self.quantiles, list), "Quantiles must be provided as list."
        # check if quantiles are float values in (0, 1)
        assert all(
            0 < quantile < 1 for quantile in self.quantiles
        ), "The quantiles specified need to be floats in-between (0, 1)."
        # sort the quantiles
        self.quantiles.sort()
        # check if quantiles contain 0.5 or close to 0.5, remove if so as 0.5 will be inserted again as first index
        self.quantiles = [quantile for quantile in self.quantiles if not math.isclose(0.5, quantile)]
        # 0 is the median quantile index
        self.quantiles.insert(0, 0.5)

    def set_max_num_lags(self, n_lags: int, config_lagged_regressors: Optional[LaggedRegressors] = None) -> int:
        """Get the greatest number of lags between the autoregression lags and the covariates lags.

        Parameters
        ----------
            n_lags : int
                number of autoregressive lagged values of series to include as model inputs
            config_lagged_regressors : configure_components.LaggedRegressors
                Configurations for lagged regressors

        Returns
        -------
            int
                Maximum number of lags between the autoregression lags and the covariates lags.
        """
        if (
            config_lagged_regressors is not None
            and config_lagged_regressors.regressors is not None
            and len(config_lagged_regressors.regressors) > 0
        ):
            lagged_regressor_lags = [val.n_lags for key, val in config_lagged_regressors.regressors.items()]
            max_lagged_regressor_lags = max(lagged_regressor_lags)
            self.max_lags = max(n_lags, max_lagged_regressor_lags)
        else:
            self.max_lags = n_lags


ConfigModel = Model


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
        config_lagged_regressors: Optional[configure_components.LaggedRegressors] = None,
        config_regressors=None,
        config_events: Optional[configure_components.Events] = None,
        config_seasonality: Optional[configure_components.Seasonalities] = None,
    ):
        if len(df["ID"].unique()) == 1 and not self.global_normalization:
            log.info("Setting normalization to global as only one dataframe provided for training.")
            self.global_normalization = True
        self.local_data_params, self.global_data_params = df_utils.init_data_params(
            df=df,
            normalize=self.normalize,
            config_lagged_regressors=config_lagged_regressors,
            config_regressors=config_regressors,
            config_events=config_events,
            config_seasonality=config_seasonality,
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
                    f"Dataset name {df_name!r} is not present in valid data_params but unknown_data_normalization is \
                        True. Using global_data_params"
                )
                data_params = self.global_data_params
            else:
                raise ValueError(
                    f"Dataset name {df_name!r} missing from training data params. Set unknown_data_normalization to \
                        use global (average) normalization parameters."
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
    optimizer_args: dict = field(default_factory=dict)
    scheduler: Optional[Union[str, Type[torch.optim.lr_scheduler.LRScheduler]]] = None
    scheduler_args: dict = field(default_factory=dict)
    early_stopping: Optional[bool] = False
    newer_samples_weight: float = 1.0
    newer_samples_start: float = 0.0
    reg_delay_pct: float = 0.5
    reg_lambda_trend: Optional[float] = None
    trend_reg_threshold: Optional[Union[bool, float]] = None
    n_data: int = field(init=False)
    loss_func_name: str = field(init=False)
    pl_trainer_config: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.newer_samples_weight >= 1.0
        assert self.newer_samples_start >= 0.0
        assert self.newer_samples_start < 1.0
        # self.set_loss_func(self.quantiles)

        # called in TimeNet configure_optimizers:
        # self.set_optimizer()
        # self.set_scheduler()

    def set_loss_func(self, quantiles: List[float]):
        if isinstance(self.loss_func, str):
            if self.loss_func.lower() in ["smoothl1", "smoothl1loss", "huber"]:
                # keeping 'huber' for backwards compatiblility, though not identical
                self.loss_func = torch.nn.SmoothL1Loss(reduction="none", beta=0.3)
            elif self.loss_func.lower() in ["mae", "maeloss", "l1", "l1loss"]:
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
        if len(quantiles) > 1:
            self.loss_func = PinballLoss(loss_func=self.loss_func, quantiles=quantiles)

    def set_auto_batch_epoch(
        self,
        n_data: int,
        min_batch: int = 8,
        max_batch: int = 2048,
        min_epoch: int = 20,
        max_epoch: int = 500,
    ):
        assert n_data >= 1
        self.n_data = n_data
        if self.batch_size is None:
            self.batch_size = int(2 ** (1 + int(1.5 * np.log10(int(n_data)))))
            self.batch_size = min(max_batch, max(min_batch, self.batch_size))
            self.batch_size = min(self.n_data, self.batch_size)
            log.info(f"Auto-set batch_size to {self.batch_size}")
        if self.epochs is None:
            # this should (with auto batch size) yield about 1000 steps minimum and 100,000 steps at upper cutoff
            self.epochs = 10 * int(np.ceil(100 / n_data * 2 ** (2.25 * np.log10(10 + n_data))))
            self.epochs = min(max_epoch, max(min_epoch, self.epochs))
            log.info(f"Auto-set epochs to {self.epochs}")
        # also set lambda_delay:
        self.lambda_delay = int(self.reg_delay_pct * self.epochs)

    def set_optimizer(self):
        """
        Set the optimizer and optimizer args. If optimizer is a string, then it will be converted to the corresponding
        torch optimizer. The optimizer is not initialized yet as this is done in configure_optimizers in TimeNet.

        Parameters
            ----------
                optimizer_name : int
                    Object provided to NeuralProphet as optimizer.
                optimizer_args : dict
                    Arguments for the optimizer.

        """
        if isinstance(self.optimizer, str):
            if self.optimizer.lower() == "adamw":
                # Tends to overfit, but reliable
                self.optimizer = torch.optim.AdamW
                self.optimizer_args["weight_decay"] = 1e-3
            elif self.optimizer.lower() == "sgd":
                # better validation performance, but diverges sometimes
                self.optimizer = torch.optim.SGD
                self.optimizer_args["momentum"] = 0.9
                self.optimizer_args["weight_decay"] = 1e-4
            else:
                raise ValueError(
                    f"The optimizer name {self.optimizer} is not supported. Please pass the optimizer class."
                )
        elif not issubclass(self.optimizer, torch.optim.Optimizer):
            raise ValueError("The provided optimizer is not supported.")

    def set_scheduler(self):
        """
        Set the scheduler and scheduler arg depending on the user selection.
        The scheduler is not initialized yet as this is done in configure_optimizers in TimeNet.
        """

        if self.scheduler is None:
            log.warning("No scheduler specified. Falling back to ExponentialLR scheduler.")
            self.scheduler = "exponentiallr"

        if isinstance(self.scheduler, str):
            if self.scheduler.lower() in ["onecycle", "onecyclelr"]:
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR
                defaults = {
                    "pct_start": 0.3,
                    "anneal_strategy": "cos",
                    "div_factor": 10.0,
                    "final_div_factor": 10.0,
                    "three_phase": True,
                }
            elif self.scheduler.lower() == "steplr":
                self.scheduler = torch.optim.lr_scheduler.StepLR
                defaults = {
                    "step_size": 10,
                    "gamma": 0.1,
                }
            elif self.scheduler.lower() == "exponentiallr":
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR
                defaults = {
                    "gamma": 0.9,
                }
            elif self.scheduler.lower() == "cosineannealinglr":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
                defaults = {
                    "T_max": 50,
                }
            elif self.scheduler.lower() == "cosineannealingwarmrestarts":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
                defaults = {
                    "T_0": 5,
                    "T_mult": 2,
                }
            else:
                raise NotImplementedError(
                    f"Scheduler {self.scheduler} is not supported from string. Please pass the scheduler class."
                )
            if self.scheduler_args is not None:
                defaults.update(self.scheduler_args)
            self.scheduler_args = defaults
        else:
            assert issubclass(
                self.scheduler, torch.optim.lr_scheduler.LRScheduler
            ), "Scheduler must be a subclass of torch.optim.lr_scheduler.LRScheduler"

    def get_reg_delay_weight(self, progress, reg_start_pct: float = 0.66, reg_full_pct: float = 1.0):
        # Ignore type warning of epochs possibly being None (does not work with dataclasses)
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

    def set_batches_per_epoch(self, batches_per_epoch: int):
        self.batches_per_epoch = batches_per_epoch

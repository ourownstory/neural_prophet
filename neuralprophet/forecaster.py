import logging
import os
import time
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Type, Union

import matplotlib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot
from matplotlib.axes import Axes
from torch.utils.data import DataLoader

from neuralprophet import configure, df_utils, np_types, time_dataset, time_net, utils, utils_metrics
from neuralprophet.data.process import (
    _check_dataframe,
    _convert_raw_predictions_to_raw_df,
    _create_dataset,
    _handle_missing_data,
    _prepare_dataframe_to_predict,
    _reshape_raw_predictions_to_forecst_df,
    _validate_column_name,
)
from neuralprophet.data.split import _make_future_dataframe, _maybe_extend_df
from neuralprophet.data.transform import _normalize
from neuralprophet.logger import MetricsLogger
from neuralprophet.plot_forecast_matplotlib import plot, plot_components
from neuralprophet.plot_forecast_plotly import plot as plot_plotly
from neuralprophet.plot_forecast_plotly import plot_components as plot_components_plotly
from neuralprophet.plot_model_parameters_matplotlib import plot_parameters
from neuralprophet.plot_model_parameters_plotly import plot_parameters as plot_parameters_plotly
from neuralprophet.plot_utils import get_valid_configuration, log_warning_deprecation_plotly, select_plotting_backend
from neuralprophet.uncertainty import Conformal

log = logging.getLogger("NP.forecaster")


class NeuralProphet:
    """NeuralProphet forecaster.

    A simple yet powerful forecaster that models:
    Trend, seasonality, events, holidays, auto-regression, lagged covariates, and future-known regressors.
    Can be regularized and configured to model nonlinear relationships.

    Parameters
    ----------
        COMMENT
        Trend Config
        COMMENT
        growth : {'off' or 'linear'}, default 'linear'
            Set use of trend growth type.

            Options:
                * ``off``: no trend.
                * (default) ``linear``: fits a piece-wise linear trend with ``n_changepoints + 1`` segments
                * ``discontinuous``: For advanced users only - not a conventional trend,
                allows arbitrary jumps at each trend changepoint

        changepoints : {list of str, list of np.datetimes or np.array of np.datetimes}, optional
            Manually set dates at which to include potential changepoints.

            Note
            ----
            Does not accept ``np.array`` of ``np.str``. If not specified, potential changepoints are selected automatically.

        n_changepoints : int
            Number of potential trend changepoints to include.

            Note
            ----
            Changepoints are selected uniformly from the first ``changepoint_range`` proportion of the history.
            Ignored if manual ``changepoints`` list is supplied.
        changepoints_range : float
            Proportion of history in which trend changepoints will be estimated.

            e.g. set to 0.8 to allow changepoints only in the first 80% of training data.
            Ignored if  manual ``changepoints`` list is supplied.
        trend_reg : float, optional
            Parameter modulating the flexibility of the automatic changepoint selection.

            Note
            ----
            Large values (~1-100) will limit the variability of changepoints.
            Small values (~0.001-1.0) will allow changepoints to change faster.
            default: 0 will fully fit a trend to each segment.

        trend_reg_threshold : bool, optional
            Allowance for trend to change without regularization.

            Options
                * ``True``: Automatically set to a value that leads to a smooth trend.
                * (default) ``False``: All changes in changepoints are regularized

        trend_global_local : str, default 'global'
            Modelling strategy of the trend when multiple time series are present.

            Options:
                * ``global``: All the elements are modelled with the same trend.
                * ``local``: Each element is modelled with a different trend.

            Note
            ----
            When only one time series is input, this parameter should not be provided.
            Internally it will be set to ``global``, meaning that all the elements(only one in this case)
            are modelled with the same trend.

        COMMENT
        Seasonality Config
        COMMENT
        yearly_seasonality : bool, int
            Fit yearly seasonality.

            Options
                * ``True`` or ``False``
                * ``auto``: set automatically
                * ``value``: number of Fourier/linear terms to generate
        weekly_seasonality : bool, int
            Fit monthly seasonality.

            Options
                * ``True`` or ``False``
                * ``auto``: set automatically
                * ``value``: number of Fourier/linear terms to generate
        daily_seasonality : bool, int
            Fit daily seasonality.

            Options
                * ``True`` or ``False``
                * ``auto``: set automatically
                * ``value``: number of Fourier/linear terms to generate
        seasonality_mode : str
            Specifies mode of seasonality

            Options
                * (default) ``additive``
                * ``multiplicative``
        seasonality_reg : float, optional
            Parameter modulating the strength of the seasonality model.

            Note
            ----
            Smaller values (~0.1-1) allow the model to fit larger seasonal fluctuations,
            larger values (~1-100) dampen the seasonality.
            default: None, no regularization
        season_global_local : str, default 'global'
            Modelling strategy of the seasonality when multiple time series are present.
            Options:
                * ``global``: All the elements are modelled with the same seasonality.
                * ``local``: Each element is modelled with a different seasonality.
            Note
            ----
            When only one time series is input, this parameter should not be provided.
            Internally it will be set to ``global``, meaning that all the elements(only one in this case)
            are modelled with the same seasonality.

        COMMENT
        AR Config
        COMMENT
        n_lags : int
            Previous time series steps to include in auto-regression. Aka AR-order
        ar_reg : float, optional
            how much sparsity to induce in the AR-coefficients

            Note
            ----
            Large values (~1-100) will limit the number of nonzero coefficients dramatically.
            Small values (~0.001-1.0) will allow more non-zero coefficients.
            default: 0 no regularization of coefficients.

        COMMENT
        Model Config
        COMMENT
        n_forecasts : int
            Number of steps ahead of prediction time step to forecast.
        num_hidden_layers : int, optional
            number of hidden layer to include in AR-Net (defaults to 0)
        d_hidden : int, optional
            dimension of hidden layers of the AR-Net. Ignored if ``num_hidden_layers`` == 0.

        COMMENT
        Train Config
        COMMENT
        learning_rate : float
            Maximum learning rate setting for 1cycle policy scheduler.

            Note
            ----
            Default ``None``: Automatically sets the ``learning_rate`` based on a learning rate range test.
            For manual user input, (try values ~0.001-10).
        epochs : int
            Number of epochs (complete iterations over dataset) to train model.

            Note
            ----
            Default ``None``: Automatically sets the number of epochs based on dataset size.
            For best results also leave batch_size to None. For manual values, try ~5-500.
        batch_size : int
            Number of samples per mini-batch.

            If not provided, ``batch_size`` is approximated based on dataset size.
            For manual values, try ~8-1024.
            For best results also leave ``epochs`` to ``None``.
        newer_samples_weight: float, default 2.0
            Sets factor by which the model fit is skewed towards more recent observations.

            Controls the factor by which final samples are weighted more compared to initial samples.
            Applies a positional weighting to each sample's loss value.

            e.g. ``newer_samples_weight = 2``: final samples are weighted twice as much as initial samples.
        newer_samples_start: float, default 0.0
            Sets beginning of 'newer' samples as fraction of training data.

            Throughout the range of 'newer' samples, the weight is increased
            from ``1.0/newer_samples_weight`` initially to 1.0 at the end,
            in a monotonously increasing function (cosine from pi to 2*pi).
        loss_func : str, torch.nn.functional.loss
            Type of loss to use:

            Options
                * (default) ``Huber``: Huber loss function
                * ``MSE``: Mean Squared Error loss function
                * ``MAE``: Mean Absolute Error loss function
                * ``torch.nn.functional.loss.``: loss or callable for custom loss, eg. L1-Loss

            Examples
            --------
            >>> from neuralprophet import NeuralProphet
            >>> import torch
            >>> import torch.nn as nn
            >>> m = NeuralProphet(loss_func=torch.nn.L1Loss)

        collect_metrics : list of str, dict, bool
            Set metrics to compute.

            Options
                * (default) ``True``: [``mae``, ``rmse``]
                * ``False``: No metrics
                * ``list``:  Valid options: [``mae``, ``rmse``, ``mse``]
                * ``dict``:  Collection of torchmetrics.Metric objects

            Examples
            --------
            >>> from neuralprophet import NeuralProphet
            >>> m = NeuralProphet(collect_metrics=["MSE", "MAE", "RMSE"])

        COMMENT
        Uncertainty Estimation
        COMMENT
        quantiles : list, default None
            A list of float values between (0, 1) which indicate the set of quantiles to be estimated.

        COMMENT
        Missing Data
        COMMENT
        impute_missing : bool
            whether to automatically impute missing dates/values

            Note
            ----
            imputation follows a linear method up to 20 missing values, more are filled with trend.
        impute_linear : int
            maximal number of missing dates/values to be imputed linearly (default: ``10``)
        impute_rolling : int
            maximal number of missing dates/values to be imputed
            using rolling average (default: ``10``)
        drop_missing : bool
            whether to automatically drop missing samples from the data

            Options
                * (default) ``False``: Samples containing NaN values are not dropped.
                * ``True``: Any sample containing at least one NaN value will be dropped.

        COMMENT
        Data Normalization
        COMMENT
        normalize : str
            Type of normalization to apply to the time series.

            Options
                * ``off`` bypasses data normalization
                * (default, binary timeseries) ``minmax`` scales the minimum value to 0.0 and the maximum value to 1.0
                * ``standardize`` zero-centers and divides by the standard deviation
                * (default) ``soft`` scales the minimum value to 0.0 and the 95th quantile to 1.0
                * ``soft1`` scales the minimum value to 0.1 and the 90th quantile to 0.9
        global_normalization : bool
            Activation of global normalization

            Options
                * ``True``: dict of dataframes is used as global_time_normalization
                * (default) ``False``: local normalization
        global_time_normalization : bool
            Specifies global time normalization

            Options
                * (default) ``True``: only valid in case of global modeling local normalization
                * ``False``: set time data_params locally
        unknown_data_normalization : bool
            Specifies unknown data normalization

            Options
                * ``True``: test data is normalized with global data params even if trained with local data params (global modeling with local normalization)
                * (default) ``False``: no global modeling with local normalization
        accelerator: str
            Name of accelerator from pytorch_lightning.accelerators to use for training. Use "auto" to automatically select an available accelerator.
            Provide `None` to deactivate the use of accelerators.
        trainer_config: dict
            Dictionary of additional trainer configuration parameters.
        prediction_frequency: dict
            periodic interval in which forecasts should be made.
            More than one item only allowed for {"daily-hour": x, "weekly-day": y"} to forecast on a specific hour of a specific day of week.

            Key: str
                periodicity of the predictions to be made.
            value: int
                forecast origin of the predictions to be made, e.g. 7 for 7am in case of 'daily-hour'.

            Options
                * ``'hourly-minute'``: forecast once per hour at a specified minute
                * ``'daily-hour'``: forecast once per day at a specified hour
                * ``'weekly-day'``: forecast once per week at a specified day
                * ``'monthly-day'``: forecast once per month at a specified day
                * ``'yearly-month'``: forecast once per year at a specified month
    """

    model: time_net.TimeNet
    trainer: pl.Trainer

    def __init__(
        self,
        growth: np_types.GrowthMode = "linear",
        changepoints: Optional[list] = None,
        n_changepoints: int = 10,
        changepoints_range: float = 0.8,
        trend_reg: float = 0,
        trend_reg_threshold: Optional[Union[bool, float]] = False,
        trend_global_local: str = "global",
        yearly_seasonality: np_types.SeasonalityArgument = "auto",
        weekly_seasonality: np_types.SeasonalityArgument = "auto",
        daily_seasonality: np_types.SeasonalityArgument = "auto",
        seasonality_mode: np_types.SeasonalityMode = "additive",
        seasonality_reg: float = 0,
        season_global_local: np_types.SeasonGlobalLocalMode = "global",
        n_forecasts: int = 1,
        n_lags: int = 0,
        num_hidden_layers: int = 0,
        d_hidden: Optional[int] = None,
        ar_reg: Optional[float] = None,
        learning_rate: Optional[float] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        loss_func: Union[str, torch.nn.modules.loss._Loss, Callable] = "Huber",
        optimizer: Union[str, Type[torch.optim.Optimizer]] = "AdamW",
        newer_samples_weight: float = 2,
        newer_samples_start: float = 0.0,
        quantiles: List[float] = [],
        impute_missing: bool = True,
        impute_linear: int = 10,
        impute_rolling: int = 10,
        drop_missing: bool = False,
        collect_metrics: np_types.CollectMetricsMode = True,
        normalize: np_types.NormalizeMode = "auto",
        global_normalization: bool = False,
        global_time_normalization: bool = True,
        unknown_data_normalization: bool = False,
        accelerator: Optional[str] = None,
        trainer_config: dict = {},
        prediction_frequency: Optional[dict] = None,
    ):
        self.config = locals()
        self.config.pop("self")

        # General
        self.name = "NeuralProphet"
        self.n_forecasts = n_forecasts
        self.prediction_frequency = prediction_frequency

        # Data Normalization settings
        self.config_normalization = configure.Normalization(
            normalize=normalize,
            global_normalization=global_normalization,
            global_time_normalization=global_time_normalization,
            unknown_data_normalization=unknown_data_normalization,
        )

        # Missing Data Preprocessing
        self.config_missing = configure.MissingDataHandling(
            impute_missing=impute_missing,
            impute_linear=impute_linear,
            impute_rolling=impute_rolling,
            drop_missing=drop_missing,
        )

        # Training
        self.config_train = configure.Train(
            quantiles=quantiles,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            loss_func=loss_func,
            optimizer=optimizer,
            newer_samples_weight=newer_samples_weight,
            newer_samples_start=newer_samples_start,
            trend_reg_threshold=trend_reg_threshold,
        )

        if isinstance(collect_metrics, list):
            log.info(
                DeprecationWarning(
                    "Providing metrics to collect via `collect_metrics` in NeuralProphet is deprecated and will be removed in a future version. The metrics are now configure in the `fit()` method via `metrics`."
                )
            )
        self.metrics = utils_metrics.get_metrics(collect_metrics)

        # AR
        self.config_ar = configure.AR(
            n_lags=n_lags,
            ar_reg=ar_reg,
        )
        self.n_lags = self.config_ar.n_lags
        self.max_lags = self.n_lags

        # Model
        self.config_model = configure.Model(
            num_hidden_layers=num_hidden_layers,
            d_hidden=d_hidden,
        )

        # Trend
        self.config_trend = configure.Trend(
            growth=growth,
            changepoints=changepoints,
            n_changepoints=n_changepoints,
            changepoints_range=changepoints_range,
            trend_reg=trend_reg,
            trend_reg_threshold=trend_reg_threshold,
            trend_global_local=trend_global_local,
        )

        # Seasonality
        self.config_seasonality = configure.ConfigSeasonality(
            mode=seasonality_mode,
            reg_lambda=seasonality_reg,
            yearly_arg=yearly_seasonality,
            weekly_arg=weekly_seasonality,
            daily_arg=daily_seasonality,
            global_local=season_global_local,
            condition_name=None,
        )

        # Events
        self.config_events: Optional[configure.ConfigEvents] = None
        self.config_country_holidays: Optional[configure.ConfigCountryHolidays] = None

        # Extra Regressors
        self.config_lagged_regressors: Optional[configure.ConfigLaggedRegressors] = None
        self.config_regressors: Optional[configure.ConfigFutureRegressors] = None

        # set during fit()
        self.data_freq = None

        # Set during _train()
        self.fitted = False
        self.data_params = None

        # Pytorch Lightning Trainer
        self.metrics_logger = MetricsLogger(save_dir=os.getcwd())
        self.accelerator = accelerator
        self.trainer_config = trainer_config

        # set during prediction
        self.future_periods = None
        self.predict_steps = self.n_forecasts
        # later set by user (optional)
        self.highlight_forecast_step_n = None
        self.true_ar_weights = None

    def add_lagged_regressor(
        self,
        names: Union[str, List[str]],
        n_lags: Union[int, np_types.Literal["auto", "scalar"]] = "auto",
        num_hidden_layers: Optional[int] = None,
        d_hidden: Optional[int] = None,
        regularization: Optional[float] = None,
        normalize: Union[bool, str] = "auto",
    ):
        """Add a covariate or list of covariate time series as additional lagged regressors to be used for fitting and predicting.
        The dataframe passed to ``fit`` and ``predict`` will have the column with the specified name to be used as
        lagged regressor. When normalize=True, the covariate will be normalized unless it is binary.

        Parameters
        ----------
            names : string or list
                name of the regressor/list of regressors.
            n_lags : int
                previous regressors time steps to use as input in the predictor (covar order)
                if ``auto``, time steps will be equivalent to the AR order (default)
                if ``scalar``, all the regressors will only use last known value as input
            num_hidden_layers : int
                number of hidden layers to include in Lagged-Regressor-Net (defaults to same configuration as AR-Net)
            d_hidden : int
                dimension of hidden layers of the Lagged-Regressor-Net. Ignored if ``num_hidden_layers`` == 0.
            regularization : float
                optional  scale for regularization strength
            normalize : bool
                optional, specify whether this regressor will benormalized prior to fitting.
                if ``auto``, binary regressors will not be normalized.
        """
        if num_hidden_layers is None:
            num_hidden_layers = self.config_model.num_hidden_layers

        if d_hidden is None:
            d_hidden = self.config_model.d_hidden
        if n_lags == 0 or n_lags is None:
            n_lags = 0
            log.warning(
                "Please, set n_lags to a value greater than 0 or to the options 'scalar' or 'auto'. No lags will be added to regressors when n_lags = 0 or n_lags is None"
            )
        if n_lags == "auto":
            if self.n_lags is not None and self.n_lags > 0:
                n_lags = self.n_lags
                log.info(
                    f"n_lags = 'auto', number of lags for regressor is set to Autoregression number of lags ({self.n_lags})"
                )
            else:
                n_lags = 1
                log.info(
                    "n_lags = 'auto', but there is no lags for Autoregression. Number of lags for regressor is automatically set to 1"
                )
        if n_lags == "scalar":
            n_lags = 1
            log.info("n_lags = 'scalar', number of lags for regressor is set to 1")
        only_last_value = False if n_lags > 1 else True
        if self.fitted:
            raise Exception("Regressors must be added prior to model fitting.")
        if not isinstance(names, list):
            names = [names]
        for name in names:
            _validate_column_name(self, name)
            if self.config_lagged_regressors is None:
                self.config_lagged_regressors = OrderedDict()
            self.config_lagged_regressors[name] = configure.LaggedRegressor(
                reg_lambda=regularization,
                normalize=normalize,
                as_scalar=only_last_value,
                n_lags=n_lags,
                num_hidden_layers=num_hidden_layers,
                d_hidden=d_hidden,
            )
        return self

    def parameters(self):
        return self.config

    def state_dict(self):
        return {
            "data_freq": self.data_freq,
            "fitted": self.fitted,
            "data_params": self.data_params,
            "optimizer": self.config_train.optimizer,
            "scheduler": self.config_train.scheduler,
            "model": self.model,
            "future_periods": self.future_periods,
            "predict_steps": self.predict_steps,
            "highlight_forecast_step_n": self.highlight_forecast_step_n,
            "true_ar_weights": self.true_ar_weights,
        }

    def add_future_regressor(
        self,
        name: str,
        regularization: Optional[float] = None,
        normalize: Union[str, bool] = "auto",
        mode: str = "additive",
    ):
        """Add a regressor as lagged covariate with order 1 (scalar) or as known in advance (also scalar).

        The dataframe passed to :meth:`fit`  and :meth:`predict` will have a column with the specified name to be used as
        a regressor. When normalize=True, the regressor will be normalized unless it is binary.

        Note
        ----
        Future Regressors have to be known for the entire forecast horizon, e.g. ``n_forecasts`` into the future.

        Parameters
        ----------
            name : string
                name of the regressor.
            regularization : float
                optional  scale for regularization strength
            normalize : bool
                optional, specify whether this regressor will be normalized prior to fitting.

                Note
                ----
                if ``auto``, binary regressors will not be normalized.
            mode : str
                ``additive`` (default) or ``multiplicative``.
        """
        if self.fitted:
            raise Exception("Regressors must be added prior to model fitting.")
        if regularization is not None:
            if regularization < 0:
                raise ValueError("regularization must be >= 0")
            if regularization == 0:
                regularization = None
        _validate_column_name(self, name)

        if self.config_regressors is None:
            self.config_regressors = OrderedDict()
        self.config_regressors[name] = configure.Regressor(reg_lambda=regularization, normalize=normalize, mode=mode)
        return self

    def add_events(
        self,
        events: Union[str, List[str]],
        lower_window: int = 0,
        upper_window: int = 0,
        regularization: Optional[float] = None,
        mode: str = "additive",
    ):
        """
        Add user specified events and their corresponding lower, upper windows and the
        regularization parameters into the NeuralProphet object

        Parameters
        ----------
            events : str, list
                name or list of names of user specified events
            lower_window : int
                the lower window for the events in the list of events
            upper_window : int
                the upper window for the events in the list of events
            regularization : float
                optional  scale for regularization strength
            mode : str
                ``additive`` (default) or ``multiplicative``.

        """
        if self.fitted:
            raise Exception("Events must be added prior to model fitting.")

        if self.config_events is None:
            self.config_events = OrderedDict({})

        if regularization is not None:
            if regularization < 0:
                raise ValueError("regularization must be >= 0")
            if regularization == 0:
                regularization = None

        if not isinstance(events, list):
            events = [events]

        for event_name in events:
            _validate_column_name(self, event_name)
            self.config_events[event_name] = configure.Event(
                lower_window=lower_window, upper_window=upper_window, reg_lambda=regularization, mode=mode
            )
        return self

    def add_country_holidays(
        self,
        country_name: Union[str, list],
        lower_window: int = 0,
        upper_window: int = 0,
        regularization: Optional[float] = None,
        mode: str = "additive",
    ):
        """
        Add a country into the NeuralProphet object to include country specific holidays
        and create the corresponding configs such as lower, upper windows and the regularization
        parameters

        Holidays can only be added for a single country. Calling the function
        multiple times will override already added country holidays.

        Parameters
        ----------
            country_name : str, list
                name or list of names of the country
            lower_window : int
                the lower window for all the country holidays
            upper_window : int
                the upper window for all the country holidays
            regularization : float
                optional  scale for regularization strength
            mode : str
                ``additive`` (default) or ``multiplicative``.
        """
        if self.fitted:
            raise Exception("Country must be specified prior to model fitting.")
        if self.config_country_holidays:
            log.warning(
                "Country holidays can only be added for a single country. Previous country holidays were overridden."
            )

        if regularization is not None:
            if regularization < 0:
                raise ValueError("regularization must be >= 0")
            if regularization == 0:
                regularization = None
        self.config_country_holidays = configure.Holidays(
            country=country_name,
            lower_window=lower_window,
            upper_window=upper_window,
            reg_lambda=regularization,
            mode=mode,
        )
        self.config_country_holidays.init_holidays()
        return self

    def add_seasonality(self, name: str, period: float, fourier_order: int, condition_name: Optional[str] = None):
        """Add a seasonal component with specified period, number of Fourier components, and regularization.

        Increasing the number of Fourier components allows the seasonality to change more quickly
        (at risk of overfitting).
        Note: regularization and mode (additive/multiplicative) are set in the main init.

        If condition_name is provided, the dataframe passed to `fit` and
        `predict` should have a column with the specified condition_name
        containing only zeros and ones, deciding when to apply seasonality.
        Floats between 0 and 1 can be used to apply seasonality partially.

        Parameters
        ----------
            name : string
                name of the seasonality component.
            period : float
                number of days in one period.
            fourier_order : int
                number of Fourier components to use.
            condition_name : string
                string name of the seasonality condition.

        Examples
        --------
        Adding a quarterly changing weekly seasonality to the model. First, add columns to df.
        The columns should contain only zeros and ones (or floats), deciding when to apply seasonality.
            >>> df["summer"] = df["ds"].apply(lambda x: x.month in [6, 7, 8])
            >>> df["fall"] = df["ds"].apply(lambda x: x.month in [9, 10, 11])
            >>> df["winter"] = df["ds"].apply(lambda x: x.month in [12, 1, 2])
            >>> df["spring"] = df["ds"].apply(lambda x: x.month in [3, 4, 5])
            >>> df.head()
                ds	        y       summer_week     fall_week   winter_week   spring_week
            0	2022-12-01  9.59    0               0            1            0
            1	2022-12-02	8.52    0               0            1            0
            2	2022-12-03	8.18    0               0            1            0
            3	2022-12-04	8.07    0               0            1            0

        As a next step, add the seasonality to the model. With period=7, we specify that the seasonality changes weekly.
            >>> m = NeuralProphet(weekly_seasonality=False)
            >>> m.add_seasonality(name="weekly_summer", period=7, fourier_order=4, condition_name="summer")
            >>> m.add_seasonality(name="weekly_winter", period=7, fourier_order=4, condition_name="winter")
            >>> m.add_seasonality(name="weekly_spring", period=7, fourier_order=4, condition_name="spring")
            >>> m.add_seasonality(name="weekly_fall", period=7, fourier_order=4, condition_name="fall")
        """
        if self.fitted:
            raise Exception("Seasonality must be added prior to model fitting.")
        if name in ["daily", "weekly", "yearly"]:
            log.error("Please use inbuilt daily, weekly, or yearly seasonality or set another name.")
        # Do not Allow overwriting built-in seasonalities
        _validate_column_name(self, name, seasons=True)
        if condition_name is not None:
            _validate_column_name(self, condition_name)
        if fourier_order <= 0:
            raise ValueError("Fourier Order must be > 0")
        self.config_seasonality.append(
            name=name, period=period, resolution=fourier_order, condition_name=condition_name, arg="custom"
        )
        return self

    def fit(
        self,
        df: pd.DataFrame,
        freq: str = "auto",
        validation_df: Optional[pd.DataFrame] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        early_stopping: bool = False,
        minimal: bool = False,
        metrics: Optional[np_types.CollectMetricsMode] = None,
        progress: Optional[str] = "bar",
        checkpointing: bool = False,
        continue_training: bool = False,
        num_workers: int = 0,
    ):
        """Train, and potentially evaluate model.

        Training/validation metrics may be distorted in case of auto-regression,
        if a large number of NaN values are present in df and/or validation_df.

        Parameters
        ----------
            df : pd.DataFrame
                containing column ``ds``, ``y``, and optionally``ID`` with all data
            freq : str
                Data step sizes. Frequency of data recording,

                Note
                ----
                Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto`` (default) to automatically set frequency.
            validation_df : pd.DataFrame, dict
                If provided, model with performance  will be evaluated after each training epoch over this data.
            epochs : int
                Number of epochs to train for. If None, uses the number of epochs specified in the model config.
            batch_size : int
                Batch size for training. If None, uses the batch size specified in the model config.
            learning_rate : float
                Learning rate for training. If None, uses the learning rate specified in the model config.
            early_stopping : bool
                Flag whether to use early stopping to stop training when training / validation loss is no longer improving.
            minimal : bool
                Minimal mode deactivates metrics, the progress bar and checkpointing. Control more granular by using the `metrics`, `progress` and `checkpointing` parameters.
            metrics : bool
                Flag whether to collect metrics during training. If None, uses the metrics specified in the model config.
            progress : str
                Flag whether to show a progress bar during training. If None, uses the progress specified in the model config.

                Options
                * (default) ``bar``
                * ``plot``
                * `None`
            checkpointing : bool
                Flag whether to save checkpoints during training
            continue_training : bool
                Flag whether to continue training from the last checkpoint
            num_workers : int
                Number of workers for data loading. If 0, data will be loaded in the main process.
                Note: using multiple workers and therefore distributed training might significantly increase
                the training time since each batch needs to be copied to each worker for each epoch. Keeping
                all data on the main process might be faster for most datasets.

        Returns
        -------
            pd.DataFrame
                metrics with training and potentially evaluation metrics
        """
        # Configuration
        if epochs is not None:
            self.config_train.epochs = epochs

        if batch_size is not None:
            self.config_train.batch_size = batch_size

        if learning_rate is not None:
            self.config_train.learning_rate = learning_rate

        if early_stopping is not None:
            self.early_stopping = early_stopping

        if metrics is not None:
            self.metrics = utils_metrics.get_metrics(metrics)

        # Warnings
        if early_stopping:
            reg_enabled = utils.check_for_regularization(
                [
                    self.config_seasonality,
                    self.config_regressors,
                    self.config_ar,
                    self.config_events,
                    self.config_country_holidays,
                    self.config_trend,
                    self.config_lagged_regressors,
                ]
            )
            if reg_enabled:
                log.warning(
                    "Early stopping is enabled, but regularization only starts after half the number of configured epochs. \
                    If you see no impact of the regularization, turn off the early_stopping or reduce the number of epochs to train for."
                )

        if progress == "plot" and metrics is False:
            log.warning("Progress plot requires metrics to be enabled. Enabling the default metrics.")
            metrics = utils_metrics.get_metrics(True)

        if not self.config_normalization.global_normalization:
            log.warning("When Global modeling with local normalization, metrics are displayed in normalized scale.")

        if minimal:
            checkpointing = False
            self.metrics = False
            progress = None

        # Pre-processing
        # Copy df and save list of unique time series IDs (the latter for global-local modelling if enabled)
        df, _, _, self.id_list = df_utils.prep_or_copy_df(df)
        df = _check_dataframe(self, df, check_y=True, exogenous=True)
        self.data_freq = df_utils.infer_frequency(df, n_lags=self.max_lags, freq=freq)
        df = _handle_missing_data(self, df, freq=self.data_freq)

        # Setup for global-local modelling: If there is only a single time series, then self.id_list = ['__df__']
        self.num_trends_modelled = len(self.id_list) if self.config_trend.trend_global_local == "local" else 1
        self.num_seasonalities_modelled = len(self.id_list) if self.config_seasonality.global_local == "local" else 1
        self.meta_used_in_model = self.num_trends_modelled != 1 or self.num_seasonalities_modelled != 1

        if self.fitted is True and not continue_training:
            log.error("Model has already been fitted. Re-fitting may break or produce different results.")
        self.max_lags = df_utils.get_max_num_lags(self.config_lagged_regressors, self.n_lags)
        if self.max_lags == 0 and self.n_forecasts > 1:
            self.n_forecasts = 1
            self.predict_steps = 1
            log.warning(
                "Changing n_forecasts to 1. Without lags, the forecast can be "
                "computed for any future time, independent of lagged values"
            )

        # Training
        if validation_df is None:
            metrics_df = self._train(
                df,
                progress_bar_enabled=bool(progress),
                metrics_enabled=bool(self.metrics),
                checkpointing_enabled=checkpointing,
                continue_training=continue_training,
                num_workers=num_workers,
            )
        else:
            df_val, _, _, _ = df_utils.prep_or_copy_df(validation_df)
            df_val = _check_dataframe(self, df_val, check_y=False, exogenous=False)
            df_val = _handle_missing_data(self, df_val, freq=self.data_freq)
            metrics_df = self._train(
                df,
                df_val=df_val,
                progress_bar_enabled=bool(progress),
                metrics_enabled=bool(self.metrics),
                checkpointing_enabled=checkpointing,
                continue_training=continue_training,
                num_workers=num_workers,
            )

        # Show training plot
        if progress == "plot":
            assert metrics_df is not None
            if validation_df is None:
                fig = pyplot.plot(metrics_df[["Loss"]])
            else:
                fig = pyplot.plot(metrics_df[["Loss", "Loss_val"]])
            # Only display the plot if the session is interactive, eg. do not show in github actions since it
            # causes an error in the Windows and MacOS environment
            if matplotlib.is_interactive():
                fig

        self.fitted = True
        return metrics_df

    def predict(self, df: pd.DataFrame, decompose: bool = True, raw: bool = False):
        """Runs the model to make predictions.

        Expects all data needed to be present in dataframe.
        If you are predicting into the unknown future and need to add future regressors or events,
        please prepare data with make_future_dataframe.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with data
            decompose : bool
                whether to add individual components of forecast to the dataframe
            raw : bool
                specifies raw data

                Options
                    * (default) ``False``: returns forecasts sorted by target (highlighting forecast age)
                    * ``True``: return the raw forecasts sorted by forecast start date

        Returns
        -------
            pd.DataFrame
                dependent on ``raw``

                Note
                ----

                ``raw == True``: columns ``ds``, ``y``, and [``step<i>``] where step<i> refers to the i-step-ahead
                prediction *made at* this row's datetime, e.g. step3 is the prediction for 3 steps into the future,
                predicted using information up to (excluding) this datetime.

                ``raw == False``: columns ``ds``, ``y``, ``trend`` and [``yhat<i>``] where yhat<i> refers to
                the i-step-ahead prediction for this row's datetime,
                e.g. yhat3 is the prediction for this datetime, predicted 3 steps ago, "3 steps old".
        """
        if raw:
            log.warning("Raw forecasts are incompatible with plotting utilities")
        if self.fitted is False:
            raise ValueError("Model has not been fitted. Predictions will be random.")
        df, received_ID_col, received_single_time_series, _ = df_utils.prep_or_copy_df(df)
        # to get all forecasteable values with df given, maybe extend into future:
        df, periods_added = _maybe_extend_df(self, df)
        df = _prepare_dataframe_to_predict(self, df)
        # normalize
        df = _normalize(self, df)
        forecast = pd.DataFrame()
        for df_name, df_i in df.groupby("ID"):
            dates, predicted, components = self._predict_raw(
                df_i, df_name, include_components=decompose, prediction_frequency=self.prediction_frequency
            )
            df_i = df_utils.drop_missing_from_df(
                df_i, self.config_missing.drop_missing, self.predict_steps, self.n_lags
            )
            if raw:
                fcst = _convert_raw_predictions_to_raw_df(self, dates, predicted, components)
                if periods_added[df_name] > 0:
                    fcst = fcst[:-1]
            else:
                fcst = _reshape_raw_predictions_to_forecst_df(
                    self, df_i, predicted, components, self.prediction_frequency, dates
                )
                if periods_added[df_name] > 0:
                    fcst = fcst[: -periods_added[df_name]]
            forecast = pd.concat((forecast, fcst), ignore_index=True)
        df = df_utils.return_df_in_original_format(forecast, received_ID_col, received_single_time_series)
        self.predict_steps = self.n_forecasts
        return df

    def test(self, df: pd.DataFrame):
        """Evaluate model on holdout data.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with with holdout data
        Returns
        -------
            pd.DataFrame
                evaluation metrics
        """
        df, _, _, _ = df_utils.prep_or_copy_df(df)
        if self.fitted is False:
            log.warning("Model has not been fitted. Test results will be random.")
        df = _check_dataframe(self, df, check_y=True, exogenous=True)
        _ = df_utils.infer_frequency(df, n_lags=self.max_lags, freq=self.data_freq)
        df = _handle_missing_data(self, df, freq=self.data_freq)
        loader = self._init_val_loader(df)
        # Use Lightning to calculate metrics
        val_metrics = self.trainer.test(self.model, dataloaders=loader)
        val_metrics_df = pd.DataFrame(val_metrics)
        # TODO Check whether supported by Lightning
        if not self.config_normalization.global_normalization:
            log.warning("Note that the metrics are displayed in normalized scale because of local normalization.")
        return val_metrics_df

    def split_df(self, df: pd.DataFrame, freq: str = "auto", valid_p: float = 0.2, local_split: bool = False):
        """Splits timeseries df into train and validation sets.
        Prevents leakage of targets. Sharing/Overbleed of inputs can be configured.
        Also performs basic data checks and fills in missing data, unless impute_missing is set to ``False``.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
            freq : str
                data step sizes. Frequency of data recording,

                Note
                ----
                Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto`` (default) to automatically set frequency.
            valid_p : float
                fraction of data to use for holdout validation set, targets will still never be shared.
            local_split : bool
                Each dataframe will be split according to valid_p locally (in case of dict of dataframes

        Returns
        -------
            tuple of two pd.DataFrames

                training data

                validation data

        See Also
        --------
            crossvalidation_split_df : Splits timeseries data in k folds for crossvalidation.
            double_crossvalidation_split_df : Splits timeseries data in two sets of k folds for crossvalidation on training and testing data.

        Examples
        --------
            >>> df1 = pd.DataFrame({'ds': pd.date_range(start = '2022-12-01', periods = 5,
            ...                     freq='D'), 'y': [9.59, 8.52, 8.18, 8.07, 7.89]})
            >>> df2 = pd.DataFrame({'ds': pd.date_range(start = '2022-12-09', periods = 5,
            ...                     freq='D'), 'y': [8.71, 8.09, 7.84, 7.65, 8.02]})
            >>> df3 = pd.DataFrame({'ds': pd.date_range(start = '2022-12-09', periods = 5,
            ...                     freq='D'), 'y': [7.67, 7.64, 7.55, 8.25, 8.3]})
            >>> df3
                ds	        y
            0	2022-12-09	7.67
            1	2022-12-10	7.64
            2	2022-12-11	7.55
            3	2022-12-12	8.25
            4	2022-12-13	8.30

        You can split a single dataframe, which also may contain NaN values.
        Please be aware this may affect training/validation performance.
            >>> (df_train, df_val) = m.split_df(df3, valid_p = 0.2)
            >>> df_train
                ds	        y
            0	2022-12-09	7.67
            1	2022-12-10	7.64
            2	2022-12-11	7.55
            3	2022-12-12	8.25
            >>> df_val
                ds	        y
            0	2022-12-13	8.3

        One can define a single df with many time series identified by an 'ID' column.
            >>> df1['ID'] = 'data1'
            >>> df2['ID'] = 'data2'
            >>> df3['ID'] = 'data3'
            >>> df = pd.concat((df1, df2, df3))

        You can use a df with many IDs (especially useful for global modeling), which will account for the time range of the whole group of time series as default.
            >>> (df_train, df_val) = m.split_df(df, valid_p = 0.2)
            >>> df_train
                ds	y	ID
            0	2022-12-01	9.59	data1
            1	2022-12-02	8.52	data1
            2	2022-12-03	8.18	data1
            3	2022-12-04	8.07	data1
            4	2022-12-05	7.89	data1
            5	2022-12-09	8.71	data2
            6	2022-12-10	8.09	data2
            7	2022-12-11	7.84	data2
            8	2022-12-09	7.67	data3
            9	2022-12-10	7.64	data3
            10	2022-12-11	7.55	data3
            >>> df_val
                ds	y	ID
            0	2022-12-12	7.65	data2
            1	2022-12-13	8.02	data2
            2	2022-12-12	8.25	data3
            3	2022-12-13	8.30	data3

        In some applications, splitting locally each time series may be helpful. In this case, one should set `local_split` to True.
            >>> (df_train, df_val) = m.split_df(df, valid_p = 0.2, local_split = True)
            >>> df_train
                ds	y	ID
            0	2022-12-01	9.59	data1
            1	2022-12-02	8.52	data1
            2	2022-12-03	8.18	data1
            3	2022-12-04	8.07	data1
            4	2022-12-09	8.71	data2
            5	2022-12-10	8.09	data2
            6	2022-12-11	7.84	data2
            7	2022-12-12	7.65	data2
            8	2022-12-09	7.67	data3
            9	2022-12-10	7.64	data3
            10	2022-12-11	7.55	data3
            11	2022-12-12	8.25	data3
            >>> df_val
                ds	y	ID
            0	2022-12-05	7.89	data1
            1	2022-12-13	8.02	data2
            2	2022-12-13	8.30	data3
        """
        df, received_ID_col, received_single_time_series, _ = df_utils.prep_or_copy_df(df)
        df = _check_dataframe(self, df, check_y=False, exogenous=False)
        freq = df_utils.infer_frequency(df, n_lags=self.max_lags, freq=freq)
        df = _handle_missing_data(self, df, freq=freq, predicting=False)
        df_train, df_val = df_utils.split_df(
            df,
            n_lags=self.max_lags,
            n_forecasts=self.n_forecasts,
            valid_p=valid_p,
            inputs_overbleed=True,
            local_split=local_split,
        )
        df_train = df_utils.return_df_in_original_format(df_train, received_ID_col, received_single_time_series)
        df_val = df_utils.return_df_in_original_format(df_val, received_ID_col, received_single_time_series)
        return df_train, df_val

    def crossvalidation_split_df(
        self,
        df: pd.DataFrame,
        freq: str = "auto",
        k: int = 5,
        fold_pct: float = 0.1,
        fold_overlap_pct: float = 0.5,
        global_model_cv_type: str = "global-time",
    ):
        """Splits timeseries data in k folds for crossvalidation.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
            freq : str
                data step sizes. Frequency of data recording,

                Note
                ----
                Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto`` (default) to automatically set frequency.
            k : int
                number of CV folds
            fold_pct : float
                percentage of overall samples to be in each fold
            fold_overlap_pct : float
                percentage of overlap between the validation folds.
            global_model_cv_type : str
                Type of crossvalidation to apply to the dict of time series.

                    options:

                        ``global-time`` (default) crossvalidation is performed according to a timestamp threshold.

                        ``local`` each episode will be crossvalidated locally (may cause time leakage among different episodes)

                        ``intersect`` only the time intersection of all the episodes will be considered. A considerable amount of data may not be used. However, this approach guarantees an equal number of train/test samples for each episode.

        Returns
        -------
            list of k tuples [(df_train, df_val), ...]

                training data

                validation data
        See Also
        --------
            split_df : Splits timeseries df into train and validation sets.
            double_crossvalidation_split_df : Splits timeseries data in two sets of k folds for crossvalidation on training and testing data.

        Examples
        --------
            >>> df1 = pd.DataFrame({'ds': pd.date_range(start = '2022-12-01', periods = 10, freq = 'D'),
            ...                     'y': [9.59, 8.52, 8.18, 8.07, 7.89, 8.09, 7.84, 7.65, 8.71, 8.09]})
            >>> df2 = pd.DataFrame({'ds': pd.date_range(start = '2022-12-02', periods = 10, freq = 'D'),
            ...                     'y': [8.71, 8.09, 7.84, 7.65, 8.02, 8.52, 8.18, 8.07, 8.25, 8.30]})
            >>> df3 = pd.DataFrame({'ds': pd.date_range(start = '2022-12-03', periods = 10, freq = 'D'),
            ...                     'y': [7.67, 7.64, 7.55, 8.25, 8.32, 9.59, 8.52, 7.55, 8.25, 8.09]})
            >>> df3
                ds	        y
            0	2022-12-03	7.67
            1	2022-12-04	7.64
            2	2022-12-05	7.55
            3	2022-12-06	8.25
            4	2022-12-07	8.32
            5	2022-12-08	9.59
            6	2022-12-09	8.52
            7	2022-12-10	7.55
            8	2022-12-11	8.25
            9	2022-12-12	8.09

        You can create folds for a single dataframe.
            >>> folds = m.crossvalidation_split_df(df3, k = 2, fold_pct = 0.2)
            >>> folds
            [(  ds            y
                0 2022-12-03  7.67
                1 2022-12-04  7.64
                2 2022-12-05  7.55
                3 2022-12-06  8.25
                4 2022-12-07  8.32
                5 2022-12-08  9.59
                6 2022-12-09  8.52,
                ds            y
                0 2022-12-10  7.55
                1 2022-12-11  8.25),
            (   ds            y
                0 2022-12-03  7.67
                1 2022-12-04  7.64
                2 2022-12-05  7.55
                3 2022-12-06  8.25
                4 2022-12-07  8.32
                5 2022-12-08  9.59
                6 2022-12-09  8.52
                7 2022-12-10  7.55,
                ds            y
                0 2022-12-11  8.25
                1 2022-12-12  8.09)]

        We can also create a df with many IDs.
            >>> df1['ID'] = 'data1'
            >>> df2['ID'] = 'data2'
            >>> df3['ID'] = 'data3'
            >>> df = pd.concat((df1, df2, df3))

        When using the df with many IDs, there are three types of possible crossvalidation. The default crossvalidation is performed according to a timestamp threshold. In this case, we can have a different number of samples for each time series per fold. This approach prevents time leakage.
            >>> folds = m.crossvalidation_split_df(df, k = 2, fold_pct = 0.2)
        One can notice how each of the folds has a different number of samples for the validation set. Nonetheless, time leakage does not occur.
            >>> folds[0][1]
                ds	y	ID
            0	2022-12-10	8.09	data1
            1	2022-12-10	8.25	data2
            2	2022-12-11	8.30	data2
            3	2022-12-10	7.55	data3
            4	2022-12-11	8.25	data3
            >>> folds[1][1]
                ds	y	ID
            0	2022-12-11	8.30	data2
            1	2022-12-11	8.25	data3
            2	2022-12-12	8.09	data3
        In some applications, crossvalidating each of the time series locally may be more adequate.
            >>> folds = m.crossvalidation_split_df(df, k = 2, fold_pct = 0.2, global_model_cv_type = 'local')
        In this way, we prevent a different number of validation samples in each fold.
            >>> folds[0][1]
                ds	y	ID
            0	2022-12-08	7.65	data1
            1	2022-12-09	8.71	data1
            2	2022-12-09	8.07	data2
            3	2022-12-10	8.25	data2
            4	2022-12-10	7.55	data3
            5	2022-12-11	8.25	data3
            >>> folds[1][1]
                ds	y	ID
            0	2022-12-09	8.71	data1
            1	2022-12-10	8.09	data1
            2	2022-12-10	8.25	data2
            3	2022-12-11	8.30	data2
            4	2022-12-11	8.25	data3
            5	2022-12-12	8.09	data3
        The last type of global model crossvalidation gets the time intersection among all the time series used. There is no time leakage in this case, and we preserve the same number of samples per fold. The only drawback of this approach is that some of the samples may not be used (those not in the time intersection).
            >>> folds = m.crossvalidation_split_df(df, k = 2, fold_pct = 0.2, global_model_cv_type = 'intersect')
            >>> folds[0][1]
                ds	y	ID
            0	2022-12-09	8.71	data1
            1	2022-12-09	8.07	data2
            2	2022-12-09	8.52	data3
            0 2022-12-09  8.52}
            >>> folds[1][1]
                ds	y	ID
            0	2022-12-10	8.09	data1
            1	2022-12-10	8.25	data2
            2	2022-12-10	7.55	data3
        """
        df, received_ID_col, received_single_time_series, _ = df_utils.prep_or_copy_df(df)
        df = _check_dataframe(self, df, check_y=False, exogenous=False)
        freq = df_utils.infer_frequency(df, n_lags=self.max_lags, freq=freq)
        df = _handle_missing_data(self, df, freq=freq, predicting=False)
        folds = df_utils.crossvalidation_split_df(
            df,
            n_lags=self.max_lags,
            n_forecasts=self.n_forecasts,
            k=k,
            fold_pct=fold_pct,
            fold_overlap_pct=fold_overlap_pct,
            global_model_cv_type=global_model_cv_type,
        )
        if not received_ID_col and received_single_time_series:
            # Delete ID column (__df__) of df_train and df_val of all folds in case ID was not previously provided
            for i in range(len(folds)):
                del folds[i][0]["ID"]
                del folds[i][1]["ID"]
        return folds

    def double_crossvalidation_split_df(
        self,
        df: pd.DataFrame,
        freq: str = "auto",
        k: int = 5,
        valid_pct: float = 0.1,
        test_pct: float = 0.1,
    ):
        """Splits timeseries data in two sets of k folds for crossvalidation on training and testing data.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
            freq : str
                data step sizes. Frequency of data recording,

                Note
                ----
                Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto`` (default) to automatically set frequency.
            k : int
                number of CV folds
            valid_pct : float
                percentage of overall samples to be in validation
            test_pct : float
                percentage of overall samples to be in test

        Returns
        -------
            tuple of k tuples [(folds_val, folds_test), …]
                elements same as :meth:`crossvalidation_split_df` returns
        """
        df, _, _, _ = df_utils.prep_or_copy_df(df)
        df = _check_dataframe(self, df, check_y=False, exogenous=False)
        freq = df_utils.infer_frequency(df, n_lags=self.max_lags, freq=freq)
        df = _handle_missing_data(self, df, freq=freq, predicting=False)
        folds_val, folds_test = df_utils.double_crossvalidation_split_df(
            df,
            n_lags=self.max_lags,
            n_forecasts=self.n_forecasts,
            k=k,
            valid_pct=valid_pct,
            test_pct=test_pct,
        )
        return folds_val, folds_test

    def create_df_with_events(self, df: pd.DataFrame, events_df: pd.DataFrame):
        """
        Create a concatenated dataframe with the time series data along with the events data expanded.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
            events_df : dict, pd.DataFrame
                containing column ``ds`` and ``event``

        Returns
        -------
            dict, pd.DataFrame
                columns ``y``, ``ds`` and other user specified events
        """
        if self.config_events is None:
            raise Exception(
                "The events configs should be added to the NeuralProphet object (add_events fn)"
                "before creating the data with events features"
            )
        df, received_ID_col, received_single_time_series, _ = df_utils.prep_or_copy_df(df)
        df = _check_dataframe(self, df, check_y=True, exogenous=False)
        df_dict_events = df_utils.create_dict_for_events_or_regressors(df, events_df, "events")
        df_created = pd.DataFrame()
        for df_name, df_i in df.groupby("ID"):
            for name in df_dict_events[df_name]["event"].unique():
                assert name in self.config_events
            df_aux = df_utils.convert_events_to_features(
                df_i,
                config_events=self.config_events,
                events_df=df_dict_events[df_name],
            )
            df_aux["ID"] = df_name
            df_created = pd.concat((df_created, df_aux), ignore_index=True)
        df = df_utils.return_df_in_original_format(df_created, received_ID_col, received_single_time_series)
        return df

    def make_future_dataframe(
        self,
        df: pd.DataFrame,
        events_df: Optional[pd.DataFrame] = None,
        regressors_df: Optional[pd.DataFrame] = None,
        periods: Optional[int] = None,
        n_historic_predictions: Union[bool, int] = False,
    ):
        """
        Extends dataframe a number of periods (time steps) into the future.

        Only use if you predict into the *unknown* future.
        New timestamps are added to the historic dataframe, with the 'y' column being NaN, as it remains to be predicted.
        Further, the given future events and regressors are added to the periods new timestamps.
        The returned dataframe will include historic data needed to additionally produce `n_historic_predictions`,
        for which there are historic observances of the series 'y'.

        Parameters
        ----------
            df: pd.DataFrame
                History to date. DataFrame containing all columns up to present
            events_df : pd.DataFrame
                Future event occurrences corresponding to `periods` steps into future.
                Contains columns ``ds`` and ``event``. The event column contains the name of the event.
            regressor_df : pd.DataFrame
                Future regressor values corresponding to `periods` steps into future.
                Contains column ``ds`` and one column for each of the external regressors.
            periods : int
                number of steps to extend the DataFrame into the future
            n_historic_predictions : bool, int
                Includes historic data needed to predict `n_historic_predictions` timesteps,
                for which there are historic observances of the series 'y'.
                False: drop historic data except for needed inputs to predict future.
                True: include entire history.

        Returns
        -------
            pd.DataFrame
                input df with ``ds`` extended into future, ``y`` set to None,
                with future events and regressors added.

        Examples
        --------
            >>> from neuralprophet import NeuralProphet
            >>> m = NeuralProphet()
            >>> # set the model to expect these events
            >>> m = m.add_events(["playoff", "superbowl"])
            >>> # create the data df with events
            >>> history_df = m.create_df_with_events(df, events_df)
            >>> metrics = m.fit(history_df, freq="D")
            >>> # forecast with events known ahead
            >>> future = m.make_future_dataframe(
            >>>     history_df, events_df, periods=365, n_historic_predictions=180
            >>> )
            >>> # get 180 past and 365 future predictions.
            >>> forecast = m.predict(df=future)

        """
        df, received_ID_col, received_single_time_series, _ = df_utils.prep_or_copy_df(df)
        events_dict = df_utils.create_dict_for_events_or_regressors(df, events_df, "events")
        regressors_dict = df_utils.create_dict_for_events_or_regressors(df, regressors_df, "regressors")

        df_future_dataframe = pd.DataFrame()
        for df_name, df_i in df.groupby("ID"):
            df_aux = _make_future_dataframe(
                self,
                df=df_i,
                events_df=events_dict[df_name],
                regressors_df=regressors_dict[df_name],
                periods=periods,
                n_historic_predictions=n_historic_predictions,
            )
            df_aux["ID"] = df_name
            df_future_dataframe = pd.concat((df_future_dataframe, df_aux), ignore_index=True)

        df_future = df_utils.return_df_in_original_format(
            df_future_dataframe, received_ID_col, received_single_time_series
        )
        return df_future

    def handle_negative_values(
        self,
        df: pd.DataFrame,
        handle: Union[str, int, float, None] = "remove",
        columns: Optional[List[str]] = None,
    ):
        """
        Handle negative values in the given columns.
        If no column or handling are provided, negative values in all numeric columns are removed.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y`` with all data
            handling : {str, int, float}, optional
                specified handling of negative values in the regressor column. Can be one of the following options:

                Options
                        * (default) ``remove``: Remove all negative values in the specified columns.
                        * ``error``: Raise an error in case of a negative value.
                        * ``float`` or ``int``: Replace negative values with the provided value.
            columns : list of str, optional
                names of the columns to process

        Returns
        -------
            pd.DataFrame
                input df with negative values handled
        """
        # Identify the columns to process
        # Either process the provided columns or default to all columns
        if columns:
            cols = columns
        else:
            cols = list(df.select_dtypes(include=np.number).columns)
        # Handle the negative values
        for col in cols:
            df = df_utils.handle_negative_values(df, col=col, handle_negatives=handle)
        return df

    def predict_trend(self, df: pd.DataFrame, quantile: float = 0.5):
        """Predict only trend component of the model.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
            quantile : float
                the quantile in (0, 1) that needs to be predicted

        Returns
        -------
            pd.DataFrame, dict
                trend on prediction dates.
        """
        if quantile is not None and not (0 < quantile < 1):
            raise ValueError("The quantile specified need to be a float in-between (0,1)")

        df, received_ID_col, received_single_time_series, _ = df_utils.prep_or_copy_df(df)
        df = _check_dataframe(self, df, check_y=False, exogenous=False)
        df = _normalize(self, df)
        df_trend = pd.DataFrame()
        for df_name, df_i in df.groupby("ID"):
            t = torch.from_numpy(np.expand_dims(df_i["t"].values, 1))  # type: ignore

            # Creating and passing meta, in this case the meta['df_name'] is the ID of the dataframe
            # Note: meta is only used on the trend method if trend_global_local is not "global"
            meta = OrderedDict()
            meta["df_name"] = [df_name for _ in range(t.shape[0])]
            if self.meta_used_in_model:
                meta_name_tensor = torch.tensor([self.model.id_dict[i] for i in meta["df_name"]])
            else:
                meta_name_tensor = None

            quantile_index = self.config_train.quantiles.index(quantile)
            trend = self.model.trend(t, meta_name_tensor).detach().numpy()[:, :, quantile_index].squeeze()

            data_params = self.config_normalization.get_data_params(df_name)
            trend = trend * data_params["y"].scale + data_params["y"].shift
            df_aux = pd.DataFrame({"ds": df_i["ds"], "trend": trend, "ID": df_name})
            df_trend = pd.concat((df_trend, df_aux), ignore_index=True)
        df = df_utils.return_df_in_original_format(df_trend, received_ID_col, received_single_time_series)
        return df

    def predict_seasonal_components(self, df: pd.DataFrame, quantile: float = 0.5):
        """Predict seasonality components

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing columns ``ds``, ``y``, and optionally``ID`` with all data
            quantile : float
                the quantile in (0, 1) that needs to be predicted

        Returns
        -------
            pd.DataFrame, dict
                seasonal components with columns of name <seasonality component name>
        """
        if quantile is not None and not (0 < quantile < 1):
            raise ValueError("The quantile specified need to be a float in-between (0,1)")

        df, received_ID_col, received_single_time_series, _ = df_utils.prep_or_copy_df(df)
        df = _check_dataframe(self, df, check_y=False, exogenous=False)
        df = _normalize(self, df)
        df_seasonal = pd.DataFrame()
        for df_name, df_i in df.groupby("ID"):
            dataset = time_dataset.TimeDataset(
                df_i,
                name=df_name,
                config_seasonality=self.config_seasonality,
                # n_lags=0,
                # n_forecasts=1,
                predict_steps=self.predict_steps,
                predict_mode=True,
                config_missing=self.config_missing,
                prediction_frequency=self.prediction_frequency,
            )
            loader = DataLoader(dataset, batch_size=min(4096, len(df)), shuffle=False, drop_last=False)
            predicted = {}
            for name in self.config_seasonality.periods:
                predicted[name] = list()
            for inputs, _, meta in loader:
                # Meta as a tensor for prediction
                if self.model.config_seasonality is None:
                    meta_name_tensor = None
                elif self.model.config_seasonality.global_local == "local":
                    meta = OrderedDict()
                    meta["df_name"] = [df_name for _ in range(inputs["time"].shape[0])]
                    meta_name_tensor = torch.tensor([self.model.id_dict[i] for i in meta["df_name"]])
                else:
                    meta_name_tensor = None

                for name in self.config_seasonality.periods:
                    features = inputs["seasonalities"][name]
                    quantile_index = self.config_train.quantiles.index(quantile)
                    y_season = torch.squeeze(
                        self.model.seasonality.compute_fourier(features=features, name=name, meta=meta_name_tensor)[
                            :, :, quantile_index
                        ]
                    )
                    predicted[name].append(y_season.data.numpy())

            for name in self.config_seasonality.periods:
                predicted[name] = np.concatenate(predicted[name])
                if self.config_seasonality.mode == "additive":
                    data_params = self.config_normalization.get_data_params(df_name)
                    predicted[name] = predicted[name] * data_params["y"].scale
            df_i = df_i[:: self.prediction_frequency].reset_index(drop=True)
            df_aux = pd.DataFrame({"ds": df_i["ds"], "ID": df_i["ID"], **predicted})
            df_seasonal = pd.concat((df_seasonal, df_aux), ignore_index=True)
        df = df_utils.return_df_in_original_format(df_seasonal, received_ID_col, received_single_time_series)
        return df

    def set_true_ar_for_eval(self, true_ar_weights: np.ndarray):
        """Configures model to evaluate closeness of AR weights to true weights.

        Parameters
        ----------
            true_ar_weights : np.array
                true AR-parameters, if known.
        """
        self.true_ar_weights = true_ar_weights

    def set_plotting_backend(self, plotting_backend: str):
        """Set plotting backend.

        Parameters
        ----------
            plotting_backend : str
            Specifies plotting backend to use for all plots. Can be configured individually for each plot.

            Options
                * ``plotly-resampler``: Use the plotly backend for plotting in resample mode. This mode uses the
                    plotly-resampler package to accelerate visualizing large data by resampling it. Only supported for
                    jupyterlab notebooks and vscode notebooks.
                * ``plotly``: Use the plotly backend for plotting
                * ``matplotlib``: use matplotlib for plotting
        """
        if plotting_backend in ["plotly", "matplotlib", "plotly-resampler"]:
            self.plotting_backend = plotting_backend
            log_warning_deprecation_plotly(self.plotting_backend)
        else:
            raise ValueError(
                "The parameter `plotting_backend` must be either 'plotly', 'plotly-resampler' or 'matplotlib'."
            )

    def highlight_nth_step_ahead_of_each_forecast(self, step_number: Optional[int] = None):
        """Set which forecast step to focus on for metrics evaluation and plotting.

        Parameters
        ----------
            step_number : int
                i-th step ahead forecast to use for statistics and plotting.

                Note
                ----
                Set to None to reset.
        """
        if step_number is not None:
            assert step_number <= self.n_forecasts
        self.highlight_forecast_step_n = step_number
        return self

    def plot(
        self,
        fcst: pd.DataFrame,
        df_name: Optional[str] = None,
        ax: Optional[Axes] = None,
        xlabel: str = "ds",
        ylabel: str = "y",
        figsize: Tuple[int, int] = (10, 6),
        forecast_in_focus: Optional[int] = None,
        plotting_backend: Optional[str] = None,
    ):
        """Plot the NeuralProphet forecast, including history.

        Parameters
        ----------
            fcst : pd.DataFrame
                output of self.predict.
            df_name : str
                ID from time series that should be plotted
            ax : matplotlib axes
                optional, matplotlib axes on which to plot.
            xlabel : string
                label name on X-axis
            ylabel : string
                label name on Y-axis
            figsize : tuple
                width, height in inches. default: (10, 6)
            plotting_backend : str
                optional, overwrites the default plotting backend.

                Options
                * ``plotly-resampler``: Use the plotly backend for plotting in resample mode. This mode uses the
                    plotly-resampler package to accelerate visualizing large data by resampling it. For some
                    environments (colab, pycharm interpreter) plotly-resampler might not properly vizualise the figures.
                    In this case, consider switching to 'plotly-auto'.
                * ``plotly``: Use the plotly backend for plotting
                * ``matplotlib``: use matplotlib for plotting
                * (default) None: Plotting backend ist set automatically. Use plotly with resampling for jupyterlab
                    notebooks and vscode notebooks. Automatically switch to plotly without resampling for all other
                    environments.
            forecast_in_focus: int
                optinal, i-th step ahead forecast to plot

                Note
                ----
                None (default): plot self.highlight_forecast_step_n by default
        """
        fcst, received_ID_col, received_single_time_series, _ = df_utils.prep_or_copy_df(fcst)
        if not received_single_time_series:
            if df_name not in fcst["ID"].unique():
                assert len(fcst["ID"].unique()) > 1
                raise Exception(
                    "Many time series are present in the pd.DataFrame (more than one ID). Please, especify ID to be plotted."
                )
            else:
                fcst = fcst[fcst["ID"] == df_name].copy(deep=True)
                log.info(f"Plotting data from ID {df_name}")
        if forecast_in_focus is None:
            forecast_in_focus = self.highlight_forecast_step_n
        if len(self.config_train.quantiles) > 1:
            if (self.highlight_forecast_step_n) is None and (
                self.n_forecasts > 1 or self.n_lags > 0
            ):  # rather query if n_forecasts >1 than n_lags>1
                raise ValueError(
                    "Please specify step_number using the highlight_nth_step_ahead_of_each_forecast function"
                    " for quantiles plotting when auto-regression enabled."
                )
            if (self.highlight_forecast_step_n or forecast_in_focus) is not None and self.n_lags == 0:
                log.warning("highlight_forecast_step_n is ignored since auto-regression not enabled.")
                self.highlight_forecast_step_n = None
        if forecast_in_focus is not None and forecast_in_focus > self.n_forecasts:
            raise ValueError(
                "Forecast_in_focus is out of range. Specify a number smaller or equal to the steps ahead of "
                "prediction time step to forecast "
            )

        if self.max_lags > 0:
            num_forecasts = sum(fcst["yhat1"].notna())
            if num_forecasts < self.n_forecasts:
                log.warning(
                    "Too few forecasts to plot a line per forecast step." "Plotting a line per forecast origin instead."
                )
                return self.plot_latest_forecast(
                    fcst,
                    ax=ax,
                    df_name=df_name,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    figsize=figsize,
                    include_previous_forecasts=num_forecasts - 1,
                    plot_history_data=True,
                )

        plotting_backend = select_plotting_backend(model=self, plotting_backend=plotting_backend)

        log_warning_deprecation_plotly(plotting_backend)
        if plotting_backend.startswith("plotly"):
            return plot_plotly(
                fcst=fcst,
                quantiles=self.config_train.quantiles,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=tuple(x * 70 for x in figsize),
                highlight_forecast=forecast_in_focus,
                resampler_active=plotting_backend == "plotly-resampler",
            )
        else:
            return plot(
                fcst=fcst,
                quantiles=self.config_train.quantiles,
                ax=ax,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=figsize,
                highlight_forecast=forecast_in_focus,
            )

    def get_latest_forecast(
        self,
        fcst: pd.DataFrame,
        df_name: Optional[str] = None,
        include_history_data: bool = False,
        include_previous_forecasts: int = 0,
    ):
        """Get the latest NeuralProphet forecast, optional including historical data.

        Parameters
        ----------
            fcst : pd.DataFrame, dict
                output of self.predict.
            df_name : str
                ID from time series that should forecast
            include_history_data : bool
                specifies whether to include historical data
            include_previous_forecasts : int
                specifies how many forecasts before latest forecast to include
        Returns
        -------
            pd.DataFrame
                columns ``ds``, ``y``, and [``origin-<i>``]

                Note
                ----
                where origin-<i> refers to the (i+1)-th latest prediction for this row's datetime.
                e.g. origin-3 is the prediction for this datetime, predicted 4 steps before the last step.
                The very latest predcition is origin-0.
        Examples
        --------
        We may get the df of the latest forecast:
            >>> forecast = m.predict(df)
            >>> df_forecast = m.get_latest_forecast(forecast)

        Number of steps before latest forecast could be included:
            >>> df_forecast = m.get_latest_forecast(forecast, include_previous_forecast=3)

        Historical data could be included, however be aware that the df could be large:
            >>> df_forecast = m.get_latest_forecast(forecast, include_history_data=True)
        """
        if self.max_lags == 0:
            raise ValueError("Use the standard plot function for models without lags.")
        fcst, received_ID_col, received_single_time_series, _ = df_utils.prep_or_copy_df(fcst)
        if not received_single_time_series:
            if df_name not in fcst["ID"].unique():
                assert len(fcst["ID"].unique()) > 1
                raise Exception(
                    "Many time series are present in the pd.DataFrame (more than one ID). Please specify ID to be "
                    "forecasted. "
                )
            else:
                fcst = fcst[fcst["ID"] == df_name].copy(deep=True)
                log.info(f"Getting data from ID {df_name}")
        if include_history_data is None:
            fcst = fcst[-(include_previous_forecasts + self.n_forecasts + self.max_lags) :]
        elif include_history_data is False:
            fcst = fcst[-(include_previous_forecasts + self.n_forecasts) :]
        elif include_history_data is True:
            fcst = fcst
        fcst = utils.fcst_df_to_latest_forecast(
            fcst, self.config_train.quantiles, n_last=1 + include_previous_forecasts
        )
        return fcst

    def plot_latest_forecast(
        self,
        fcst: pd.DataFrame,
        df_name: Optional[str] = None,
        ax: Optional[Axes] = None,
        xlabel: str = "ds",
        ylabel: str = "y",
        figsize: Tuple[int, int] = (10, 6),
        include_previous_forecasts: int = 0,
        plot_history_data: Optional[bool] = None,
        plotting_backend: Optional[str] = None,
    ):
        """Plot the latest NeuralProphet forecast(s), including history.

        Parameters
        ----------
            fcst : pd.DataFrame
                output of self.predict.
            df_name : str
                ID from time series that should be plotted
            ax : matplotlib axes
                Optional, matplotlib axes on which to plot.
            xlabel : str
                label name on X-axis
            ylabel : str
                abel name on Y-axis
            figsize : tuple
                 width, height in inches. default: (10, 6)
            include_previous_forecasts : int
                number of previous forecasts to include in plot
            plot_history_data : bool
                specifies plot of historical data
            plotting_backend : str
                optional, overwrites the default plotting backend.

                Options
                * ``plotly-resampler``: Use the plotly backend for plotting in resample mode. This mode uses the
                    plotly-resampler package to accelerate visualizing large data by resampling it. For some
                    environments (colab, pycharm interpreter) plotly-resampler might not properly vizualise the figures.
                    In this case, consider switching to 'plotly-auto'.
                * ``plotly``: Use the plotly backend for plotting
                * ``matplotlib``: use matplotlib for plotting
                ** (default) None: Plotting backend ist set automatically. Use plotly with resampling for jupyterlab
                    notebooks and vscode notebooks. Automatically switch to plotly without resampling for all other
                    environments.
        Returns
        -------
            matplotlib.axes.Axes
                plot of NeuralProphet forecasting
        """
        if self.max_lags == 0:
            raise ValueError("Use the standard plot function for models without lags.")
        fcst, received_ID_col, received_single_time_series, _ = df_utils.prep_or_copy_df(fcst)
        if not received_single_time_series:
            if df_name not in fcst["ID"].unique():
                assert len(fcst["ID"].unique()) > 1
                raise Exception(
                    "Many time series are present in the pd.DataFrame (more than one ID). Please, especify ID to be plotted."
                )
            else:
                fcst = fcst[fcst["ID"] == df_name].copy(deep=True)
                log.info(f"Plotting data from ID {df_name}")
        if len(self.config_train.quantiles) > 1:
            log.warning(
                "Plotting latest forecasts when uncertainty estimation enabled"
                " plots the forecasts only for the median quantile."
            )
        if plot_history_data is None:
            fcst = fcst[-(include_previous_forecasts + self.n_forecasts + self.max_lags) :]
        elif plot_history_data is False:
            fcst = fcst[-(include_previous_forecasts + self.n_forecasts) :]
        elif plot_history_data is True:
            fcst = fcst
        fcst = utils.fcst_df_to_latest_forecast(
            fcst, self.config_train.quantiles, n_last=1 + include_previous_forecasts
        )

        # Check whether a local or global plotting backend is set.
        plotting_backend = select_plotting_backend(model=self, plotting_backend=plotting_backend)

        log_warning_deprecation_plotly(plotting_backend)
        if plotting_backend.startswith("plotly"):
            return plot_plotly(
                fcst=fcst,
                quantiles=self.config_train.quantiles,
                ylabel=ylabel,
                xlabel=xlabel,
                figsize=tuple(x * 70 for x in figsize),
                highlight_forecast=self.highlight_forecast_step_n,
                line_per_origin=True,
                resampler_active=plotting_backend == "plotly-resampler",
            )
        else:
            return plot(
                fcst=fcst,
                quantiles=self.config_train.quantiles,
                ax=ax,
                ylabel=ylabel,
                xlabel=xlabel,
                figsize=figsize,
                highlight_forecast=self.highlight_forecast_step_n,
                line_per_origin=True,
            )

    def plot_last_forecast(
        self,
        fcst: pd.DataFrame,
        df_name: Optional[str] = None,
        ax: Optional[Axes] = None,
        xlabel: str = "ds",
        ylabel: str = "y",
        figsize: Tuple[int, int] = (10, 6),
        include_previous_forecasts: int = 0,
        plot_history_data: Optional[bool] = None,
        plotting_backend: Optional[str] = None,
    ):
        args = locals()
        log.warning(
            "plot_last_forecast() has been renamed to plot_latest_forecast() and is therefore deprecated. "
            "Please use plot_latst_forecast() in the future"
        )

        return NeuralProphet.plot_latest_forecast(**args)

    def plot_components(
        self,
        fcst: pd.DataFrame,
        df_name: str = "__df__",
        figsize: Optional[Tuple[int, int]] = None,
        forecast_in_focus: Optional[int] = None,
        plotting_backend: Optional[str] = None,
        components: Union[None, str, List[str]] = None,
        one_period_per_season: bool = False,
    ):
        """Plot the NeuralProphet forecast components.

        Parameters
        ----------
            fcst : pd.DataFrame
                output of self.predict
            df_name : str
                ID from time series that should be plotted
            figsize : tuple
                width, height in inches.

                Note
                ----
                None (default):  automatic (10, 3 * npanel)
            forecast_in_focus: int
                optinal, i-th step ahead forecast to plot

                Note
                ----
                None (default): plot self.highlight_forecast_step_n by default
            plotting_backend : str
                optional, overwrites the default plotting backend.

                Options
                * ``plotly-resampler``: Use the plotly backend for plotting in resample mode. This mode uses the
                    plotly-resampler package to accelerate visualizing large data by resampling it. For some
                    environments (colab, pycharm interpreter) plotly-resampler might not properly vizualise the figures.
                    In this case, consider switching to 'plotly-auto'.
                * ``plotly``: Use the plotly backend for plotting
                * ``matplotlib``: use matplotlib for plotting
                * (default) None: Plotting backend ist set automatically. Use plotly with resampling for jupyterlab
                    notebooks and vscode notebooks. Automatically switch to plotly without resampling for all other
                    environments.
            components: str or list, optional
                name or list of names of components to plot

                Options
                ----
                * (default)``None``:  All components the user set in the model configuration are plotted.
                * ``trend``
                * ``seasonality``: select all seasonalities
                * ``autoregression``
                * ``lagged_regressors``: select all lagged regressors
                * ``future_regressors``: select all future regressors
                * ``events``: select all events and country holidays
                * ``uncertainty``
            one_period_per_season : bool
                Plot one period per season, instead of the true seasonal components of the forecast.

        Returns
        -------
            matplotlib.axes.Axes
                plot of NeuralProphet components
        """
        fcst, received_ID_col, received_single_time_series, _ = df_utils.prep_or_copy_df(fcst)
        if not received_single_time_series:
            if df_name not in fcst["ID"].unique():
                assert len(fcst["ID"].unique()) > 1
                raise Exception(
                    "Many time series are present in the pd.DataFrame (more than one ID). Please, especify ID to be plotted."
                )
            else:
                fcst = fcst[fcst["ID"] == df_name].copy(deep=True)
                log.info(f"Plotting data from ID {df_name}")
        else:
            if df_name is None:
                df_name = "__df__"

        # Check if highlighted forecast step is overwritten
        if forecast_in_focus is None:
            forecast_in_focus = self.highlight_forecast_step_n
        if (self.highlight_forecast_step_n or forecast_in_focus) is not None and self.config_ar.n_lags == 0:
            log.warning("highlight_forecast_step_n is ignored since autoregression not enabled.")
            # self.highlight_forecast_step_n = None
            forecast_in_focus = None
        if forecast_in_focus is not None and forecast_in_focus > self.n_forecasts:
            raise ValueError(
                "Forecast_in_focus is out of range. Specify a number smaller or equal to the steps ahead of "
                "prediction time step to forecast "
            )

        # Error if local modelling of season and df_name not provided
        if self.model.config_seasonality is not None:
            if self.model.config_seasonality.global_local == "local" and df_name is None:
                raise Exception(
                    "df_name parameter is required for multiple time series and local modeling of at least one component."
                )

        # Validate components to be plotted
        valid_components_set = [
            "trend",
            "seasonality",
            "autoregression",
            "lagged_regressors",
            "events",
            "future_regressors",
            "uncertainty",
        ]
        valid_plot_configuration = get_valid_configuration(
            m=self,
            components=components,
            df_name=df_name,
            valid_set=valid_components_set,
            validator="plot_components",
            forecast_in_focus=forecast_in_focus,
        )

        # Check whether a local or global plotting backend is set.
        plotting_backend = select_plotting_backend(model=self, plotting_backend=plotting_backend)

        log_warning_deprecation_plotly(plotting_backend)
        if plotting_backend.startswith("plotly"):
            return plot_components_plotly(
                m=self,
                fcst=fcst,
                plot_configuration=valid_plot_configuration,
                figsize=tuple(x * 70 for x in figsize) if figsize else (700, 210),
                df_name=df_name,
                one_period_per_season=one_period_per_season,
                resampler_active=plotting_backend == "plotly-resampler",
            )
        else:
            return plot_components(
                m=self,
                fcst=fcst,
                plot_configuration=valid_plot_configuration,
                quantile=self.config_train.quantiles[0],  # plot components only for median quantile
                figsize=figsize,
                df_name=df_name,
                one_period_per_season=one_period_per_season,
            )

    def plot_parameters(
        self,
        weekly_start: int = 0,
        yearly_start: int = 0,
        figsize: Optional[Tuple[int, int]] = None,
        forecast_in_focus: Optional[int] = None,
        df_name: Optional[str] = None,
        plotting_backend: Optional[str] = None,
        quantile: Optional[float] = None,
        components: Union[None, str, List[str]] = None,
    ):
        """Plot the NeuralProphet forecast components.

        Parameters
        ----------
            weekly_start : int
                specifying the start day of the weekly seasonality plot.

                Note
                ----
                0 (default) starts the week on Sunday. 1 shifts by 1 day to Monday, and so on.
            yearly_start : int
                specifying the start day of the yearly seasonality plot.

                Note
                ----
                0 (default) starts the year on Jan 1. 1 shifts by 1 day to Jan 2, and so on.
            df_name : str
                name of dataframe to refer to data params from original keys of train dataframes (used for local normalization in global modeling)
            figsize : tuple
                width, height in inches.

                Note
                ----
                None (default):  automatic (10, 3 * npanel)
            forecast_in_focus: int
                optinal, i-th step ahead forecast to plot

                Note
                ----
                None (default): plot self.highlight_forecast_step_n by default

            plotting_backend : str
                optional, overwrites the default plotting backend.

                Options
                * ``plotly-resampler``: Use the plotly backend for plotting in resample mode. This mode uses the
                    plotly-resampler package to accelerate visualizing large data by resampling it. For some
                    environments (colab, pycharm interpreter) plotly-resampler might not properly vizualise the figures.
                    In this case, consider switching to 'plotly-auto'.
                * ``plotly``: Use the plotly backend for plotting
                * ``matplotlib``: use matplotlib for plotting
                * (default) None: Plotting backend ist set automatically. Use plotly with resampling for jupyterlab
                    notebooks and vscode notebooks. Automatically switch to plotly without resampling for all other
                    environments.

                Note
                ----
                For multiple time series and local modeling of at least one component, the df_name parameter is required.

            quantile : float
                The quantile for which the model parameters are to be plotted

                Note
                ----
                None (default):  Parameters will be plotted for the median quantile.

            components: str or list, optional
                name or list of names of parameters to plot

               Options
                ----
                * (default) ``None``:  All parameter the user set in the model configuration are plotted.
                * ``trend``
                * ``trend_rate_change``
                * ``seasonality``: : select all seasonalities
                * ``autoregression``
                * ``lagged_regressors``: select all lagged regressors
                * ``events``: select all events and country holidays
                * ``future_regressors``: select all future regressors

        Returns
        -------
            matplotlib.axes.Axes
                plot of NeuralProphet forecasting
        """
        # Check if highlighted forecast step is overwritten
        if forecast_in_focus is None:
            forecast_in_focus = self.highlight_forecast_step_n
        if (self.highlight_forecast_step_n or forecast_in_focus) is not None and self.config_ar.n_lags == 0:
            log.warning("highlight_forecast_step_n is ignored since autoregression not enabled.")
            forecast_in_focus = None
        if forecast_in_focus is not None and forecast_in_focus > self.n_forecasts:
            raise ValueError(
                "Forecast_in_focus is out of range. Specify a number smaller or equal to the steps ahead of "
                "prediction time step to forecast "
            )
        if quantile is not None:
            # ValueError if model was not trained or predicted with selected quantile for plotting
            if not (0 < quantile < 1):
                raise ValueError("The quantile selected needs to be a float in-between (0,1)")
            # ValueError if selected quantile is out of range
            if quantile not in self.config_train.quantiles:
                raise ValueError("Selected quantile is not specified in the model configuration.")
        else:
            # plot parameters for median quantile if not specified
            quantile = self.config_train.quantiles[0]

        # Validate components to be plotted
        valid_parameters_set = [
            "trend",
            "trend_rate_change",
            "seasonality",
            "autoregression",
            "lagged_regressors",
            "events",
            "future_regressors",
        ]
        valid_plot_configuration = get_valid_configuration(
            m=self,
            components=components,
            df_name=df_name,
            forecast_in_focus=forecast_in_focus,
            valid_set=valid_parameters_set,
            validator="plot_parameters",
            quantile=quantile,
        )

        # Check whether a local or global plotting backend is set.
        plotting_backend = select_plotting_backend(model=self, plotting_backend=plotting_backend)

        log_warning_deprecation_plotly(plotting_backend)
        if plotting_backend.startswith("plotly"):
            return plot_parameters_plotly(
                m=self,
                quantile=quantile,
                weekly_start=weekly_start,
                yearly_start=yearly_start,
                figsize=tuple(x * 70 for x in figsize) if figsize else (700, 210),
                df_name=valid_plot_configuration["df_name"],
                plot_configuration=valid_plot_configuration,
                forecast_in_focus=forecast_in_focus,
                resampler_active=plotting_backend == "plotly-resampler",
            )
        else:
            return plot_parameters(
                m=self,
                quantile=quantile,
                weekly_start=weekly_start,
                yearly_start=yearly_start,
                figsize=figsize,
                df_name=valid_plot_configuration["df_name"],
                plot_configuration=valid_plot_configuration,
                forecast_in_focus=forecast_in_focus,
            )

    def _init_model(self):
        """Build Pytorch model with configured hyperparamters.

        Returns
        -------
            TimeNet model
        """
        self.model = time_net.TimeNet(
            config_train=self.config_train,
            config_trend=self.config_trend,
            config_ar=self.config_ar,
            config_seasonality=self.config_seasonality,
            config_lagged_regressors=self.config_lagged_regressors,
            config_regressors=self.config_regressors,
            config_events=self.config_events,
            config_holidays=self.config_country_holidays,
            config_normalization=self.config_normalization,
            n_forecasts=self.n_forecasts,
            n_lags=self.n_lags,
            max_lags=self.max_lags,
            num_hidden_layers=self.config_model.num_hidden_layers,
            d_hidden=self.config_model.d_hidden,
            metrics=self.metrics,
            id_list=self.id_list,
            num_trends_modelled=self.num_trends_modelled,
            num_seasonalities_modelled=self.num_seasonalities_modelled,
            meta_used_in_model=self.meta_used_in_model,
        )
        log.debug(self.model)
        return self.model

    def _init_train_loader(self, df, num_workers=0):
        """Executes data preparation steps and initiates training procedure.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
            num_workers : int
                number of workers for data loading

        Returns
        -------
            torch DataLoader
        """
        df, _, _, _ = df_utils.prep_or_copy_df(df)
        # if not self.fitted:
        self.config_normalization.init_data_params(
            df=df,
            config_lagged_regressors=self.config_lagged_regressors,
            config_regressors=self.config_regressors,
            config_events=self.config_events,
            config_seasonality=self.config_seasonality,
        )

        df = _normalize(self, df)
        # if not self.fitted:
        if self.config_trend.changepoints is not None:
            # scale user-specified changepoint times
            df_aux = pd.DataFrame({"ds": pd.Series(self.config_trend.changepoints)})
            self.config_trend.changepoints = _normalize(self, df_aux)["t"].values  # type: ignore # types are numpy.ArrayLike and list

        # df_merged, _ = df_utils.join_dataframes(df)
        # df_merged = df_merged.sort_values("ds")
        # df_merged.drop_duplicates(inplace=True, keep="first", subset=["ds"])
        df_merged = df_utils.merge_dataframes(df)
        self.config_seasonality = utils.set_auto_seasonalities(df_merged, config_seasonality=self.config_seasonality)
        if self.config_country_holidays is not None:
            self.config_country_holidays.init_holidays(df_merged)

        dataset = _create_dataset(
            self, df, predict_mode=False, prediction_frequency=self.prediction_frequency
        )  # needs to be called after set_auto_seasonalities

        # Determine the max_number of epochs
        self.config_train.set_auto_batch_epoch(n_data=len(dataset))

        loader = DataLoader(dataset, batch_size=self.config_train.batch_size, shuffle=True, num_workers=num_workers)

        return loader

    def _init_val_loader(self, df):
        """Executes data preparation steps and initiates evaluation procedure.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data

        Returns
        -------
            torch DataLoader
        """
        df, _, _, _ = df_utils.prep_or_copy_df(df)
        df = _normalize(self, df)
        dataset = _create_dataset(self, df, predict_mode=False)
        loader = DataLoader(dataset, batch_size=min(1024, len(dataset)), shuffle=False, drop_last=False)
        return loader

    def _train(
        self,
        df: pd.DataFrame,
        df_val: Optional[pd.DataFrame] = None,
        progress_bar_enabled: bool = True,
        metrics_enabled: bool = False,
        checkpointing_enabled: bool = False,
        continue_training=False,
        num_workers=0,
    ):
        """
        Execute model training procedure for a configured number of epochs.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
            df_val : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with validation data
            progress_bar_enabled : bool
                whether to show a progress bar during training
            metrics_enabled : bool
                whether to collect metrics during training
            checkpointing_enabled : bool
                whether to save checkpoints during training
            continue_training : bool
                whether to continue training from the last checkpoint
            num_workers : int
                number of workers for data loading

        Returns
        -------
            pd.DataFrame
                metrics
        """
        # Set up data the training dataloader
        df, _, _, _ = df_utils.prep_or_copy_df(df)
        train_loader = self._init_train_loader(df, num_workers)
        dataset_size = len(df)  # train_loader.dataset

        # Internal flag to check if validation is enabled
        validation_enabled = df_val is not None

        # Init the model, if not continue from checkpoint
        if continue_training:
            raise NotImplementedError(
                "Continuing training from checkpoint is not implemented yet. This feature is planned for one of the upcoming releases."
            )
        else:
            self.model = self._init_model()

        # Init the Trainer
        self.trainer, checkpoint_callback = utils.configure_trainer(
            config_train=self.config_train,
            config=self.trainer_config,
            metrics_logger=self.metrics_logger,
            early_stopping=self.early_stopping,
            early_stopping_target="Loss_val" if validation_enabled else "Loss",
            accelerator=self.accelerator,
            progress_bar_enabled=progress_bar_enabled,
            metrics_enabled=metrics_enabled,
            checkpointing_enabled=checkpointing_enabled,
            num_batches_per_epoch=len(train_loader),
        )

        # Tune hyperparams and train
        if validation_enabled:
            # Set up data the validation dataloader
            df_val, _, _, _ = df_utils.prep_or_copy_df(df_val)
            val_loader = self._init_val_loader(df_val)

            if not continue_training and not self.config_train.learning_rate:
                # Set parameters for the learning rate finder
                self.config_train.set_lr_finder_args(dataset_size=dataset_size, num_batches=len(train_loader))
                # Find suitable learning rate
                lr_finder = self.trainer.tuner.lr_find(
                    self.model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    **self.config_train.lr_finder_args,
                )
                # Estimate the optimat learning rate from the loss curve
                assert lr_finder is not None
                _, _, lr_suggestion = utils.smooth_loss_and_suggest(lr_finder.results)
                self.model.learning_rate = lr_suggestion
            start = time.time()
            self.trainer.fit(
                self.model,
                train_loader,
                val_loader,
                ckpt_path=self.metrics_logger.checkpoint_path if continue_training else None,
            )
        else:
            if not continue_training and not self.config_train.learning_rate:
                # Set parameters for the learning rate finder
                self.config_train.set_lr_finder_args(dataset_size=dataset_size, num_batches=len(train_loader))
                # Find suitable learning rate
                lr_finder = self.trainer.tuner.lr_find(
                    self.model,
                    train_dataloaders=train_loader,
                    **self.config_train.lr_finder_args,
                )
                assert lr_finder is not None
                # Estimate the optimat learning rate from the loss curve
                _, _, lr_suggestion = utils.smooth_loss_and_suggest(lr_finder.results)
                self.model.learning_rate = lr_suggestion
            start = time.time()
            self.trainer.fit(
                self.model,
                train_loader,
                ckpt_path=self.metrics_logger.checkpoint_path if continue_training else None,
            )

        log.debug("Train Time: {:8.3f}".format(time.time() - start))

        # Load best model from training
        if checkpoint_callback is not None:
            if checkpoint_callback.best_model_score < checkpoint_callback.current_score:
                log.info(
                    f"Loading best model with score {checkpoint_callback.best_model_score} from checkpoint (latest score is {checkpoint_callback.current_score})"
                )
                self.model = time_net.TimeNet.load_from_checkpoint(checkpoint_callback.best_model_path)

        if not metrics_enabled:
            return None

        # Return metrics collected in logger as dataframe
        metrics_df = pd.DataFrame(self.metrics_logger.history)
        return metrics_df

    def restore_trainer(self):
        """
        Restore the trainer based on the forecaster configuration.
        """
        self.trainer, _ = utils.configure_trainer(
            config_train=self.config_train,
            config=self.trainer_config,
            metrics_logger=self.metrics_logger,
            early_stopping=self.early_stopping,
            accelerator=self.accelerator,
            metrics_enabled=bool(self.metrics),
        )

    def _eval_true_ar(self):
        assert self.max_lags > 0
        if self.highlight_forecast_step_n is None:
            if self.max_lags > 1:
                raise ValueError("Please define forecast_lag for sTPE computation")
            forecast_pos = 1
        else:
            forecast_pos = self.highlight_forecast_step_n
        weights = self.model.ar_weights.detach().numpy()  # type: ignore
        weights = weights[forecast_pos - 1, :][::-1]
        sTPE = utils.symmetric_total_percentage_error(self.true_ar_weights, weights)
        log.info("AR parameters: ", self.true_ar_weights, "\n", "Model weights: ", weights)
        return sTPE

    def _predict_raw(self, df, df_name, include_components=False, prediction_frequency=None):
        """Runs the model to make predictions.

        Predictions are returned in raw vector format without decomposition.
        Predictions are given on a forecast origin basis, not on a target basis.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
            df_name : str
                name of the data params from which the current dataframe refers to (only in case of local_normalization)
            include_components : bool
                whether to return individual components of forecast
        prediction_frequency: dict
            periodic interval in which forecasts should be made.
            Key: str
                periodicity of the predictions to be made, e.g. 'daily-hour'.

            Options
                * ``'hourly-minute'``: forecast once per hour at a specified minute
                * ``'daily-hour'``: forecast once per day at a specified hour
                * ``'weekly-day'``: forecast once per week at a specified day
                * ``'monthly-day'``: forecast once per month at a specified day
                * ``'yearly-month'``: forecast once per year at a specified month

            value: int
                forecast origin of the predictions to be made, e.g. 7 for 7am in case of 'daily-hour'.

        Returns
        -------
            pd.Series
                timestamps referring to the start of the predictions.
            np.array
                array containing the forecasts
            dict[np.array]
                Dictionary of components containing an array of each components contribution to the forecast
        """
        # Receives df with single ID column
        assert len(df["ID"].unique()) == 1
        if "y_scaled" not in df.columns or "t" not in df.columns:
            raise ValueError("Received unprepared dataframe to predict. " "Please call predict_dataframe_to_predict.")
        dataset = _create_dataset(self, df, predict_mode=True, prediction_frequency=prediction_frequency)
        loader = DataLoader(dataset, batch_size=min(1024, len(df)), shuffle=False, drop_last=False)
        if self.n_forecasts > 1:
            dates = df["ds"].iloc[self.max_lags : -self.n_forecasts + 1]
        else:
            dates = df["ds"].iloc[self.max_lags :]

        # Pass the include_components flag to the model
        self.model.set_compute_components(include_components)
        # Compute the predictions and components (if requested)
        result = self.trainer.predict(self.model, loader)
        # Extract the prediction and components
        predicted, component_vectors = zip(*result)
        predicted = np.concatenate(predicted)

        # Post-process and normalize the predictions
        data_params = self.config_normalization.get_data_params(df_name)
        scale_y, shift_y = data_params["y"].scale, data_params["y"].shift
        predicted = predicted * scale_y + shift_y

        if include_components:
            component_keys = component_vectors[0].keys()
            components = {key: None for key in component_keys}
            # Transform the components list into a dictionary
            for batch in component_vectors:
                for key in component_keys:
                    components[key] = (
                        np.concatenate([components[key], batch[key]]) if (components[key] is not None) else batch[key]  # type: ignore
                    )
            for name, value in components.items():
                multiplicative = False  # Flag for multiplicative components
                if "trend" in name:
                    trend = value
                elif "event_" in name or "events_" in name:  # accounts for events and holidays
                    event_name = name.split("_")[1]
                    if self.config_events is not None and event_name in self.config_events:
                        if self.config_events[event_name].mode == "multiplicative":
                            multiplicative = True
                    elif (
                        self.config_country_holidays is not None
                        and event_name in self.config_country_holidays.holiday_names
                    ):
                        if self.config_country_holidays.mode == "multiplicative":
                            multiplicative = True
                    elif "multiplicative" in name:
                        multiplicative = True
                elif "season" in name and self.config_seasonality.mode == "multiplicative":
                    multiplicative = True
                elif (
                    "future_regressor_" in name or "future_regressors_" in name
                ) and self.config_regressors is not None:
                    regressor_name = name.split("_")[2]
                    if self.config_regressors is not None and regressor_name in self.config_regressors:
                        if self.config_regressors[regressor_name].mode == "multiplicative":
                            multiplicative = True
                    elif "multiplicative" in regressor_name:
                        multiplicative = True

                # scale additive components
                if not multiplicative:
                    components[name] = value * scale_y
                    if "trend" in name:
                        components[name] += shift_y
                # scale multiplicative components
                elif multiplicative:
                    components[name] = value * trend * scale_y  # type: ignore # output absolute value of respective additive component

        else:
            components = None

        return dates, predicted, components

    def conformal_predict(
        self,
        df: pd.DataFrame,
        calibration_df: pd.DataFrame,
        alpha: Union[float, Tuple[float, float]],
        method: str = "naive",
        plotting_backend: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Apply a given conformal prediction technique to get the uncertainty prediction intervals (or q-hats). Then predict.

        Parameters
        ----------
            df : pd.DataFrame
                test dataframe containing column ``ds``, ``y``, and optionally ``ID`` with data
            calibration_df : pd.DataFrame
                holdout calibration dataframe for split conformal prediction
            alpha : float or tuple
                user-specified significance level of the prediction interval, float if coverage error spread arbitrarily over
                left and right tails, tuple of two floats for different coverage error over left and right tails respectively
            method : str
                name of conformal prediction technique used

                Options
                    * (default) ``naive``: Naive or Absolute Residual
                    * ``cqr``: Conformalized Quantile Regression
            plotting_backend : str
                specifies the plotting backend for the nonconformity scores plot, if any

                Options
                    * ``plotly-resampler``: Use the plotly backend for plotting in resample mode. This mode uses the
                    plotly-resampler package to accelerate visualizing large data by resampling it. For some
                    environments (colab, pycharm interpreter) plotly-resampler might not properly vizualise the figures.
                    In this case, consider switching to 'plotly-auto'.
                    * ``plotly``: Use the plotly backend for plotting
                    * ``matplotlib``: Use matplotlib backend for plotting
                    * (default) None: Plotting backend ist set automatically. Use plotly with resampling for jupyterlab
                    notebooks and vscode notebooks. Automatically switch to plotly without resampling for all other
                    environments.
            kwargs : dict
                additional predict parameters for test df

        Returns
        -------
            pd.DataFrame, Optional[pd.DataFrame]
                test dataframe with the conformal prediction intervals and evaluation dataframe if evaluate set to True
        """
        # get predictions for calibration dataframe
        df_cal = self.predict(calibration_df)
        # get predictions for test dataframe
        df_test = self.predict(df, **kwargs)
        # initiate Conformal instance
        c = Conformal(
            alpha=alpha,
            method=method,
            n_forecasts=self.n_forecasts,
            quantiles=self.config_train.quantiles,
        )
        # call Conformal's predict to output test df with conformal prediction intervals
        df_forecast = c.predict(df=df_test, df_cal=df_cal)
        # plot one-sided prediction interval width with q
        if plotting_backend:
            c.plot(plotting_backend)

        return df_forecast

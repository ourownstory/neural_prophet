import time
from collections import OrderedDict
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

from neuralprophet import configure
from neuralprophet import time_net
from neuralprophet import time_dataset
from neuralprophet import df_utils
from neuralprophet import utils
from neuralprophet.plot_forecast import plot, plot_components
from neuralprophet.plot_forecast_plotly import plot as plot_plotly, plot_components as plot_components_plotly
from neuralprophet.plot_model_parameters_plotly import plot_parameters as plot_parameters_plotly
from neuralprophet.plot_model_parameters import plot_parameters
from neuralprophet import metrics

log = logging.getLogger("NP.forecaster")


METRICS = {
    "mae": metrics.MAE,
    "mse": metrics.MSE,
    "rmse": metrics.RMSE,
}


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

        collect_metrics : list of str, bool
            Set metrics to compute.

            Options
                * (default) ``True``: [``mae``, ``rmse``]
                * ``False``: No metrics
                * ``list``:  Valid options: [``mae``, ``rmse``, ``mse``]

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
    """

    def __init__(
        self,
        growth="linear",
        changepoints=None,
        n_changepoints=10,
        changepoints_range=0.8,
        trend_reg=0,
        trend_reg_threshold=False,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        seasonality_mode="additive",
        seasonality_reg=0,
        n_forecasts=1,
        n_lags=0,
        num_hidden_layers=0,
        d_hidden=None,
        ar_reg=None,
        learning_rate=None,
        epochs=None,
        batch_size=None,
        loss_func="Huber",
        optimizer="AdamW",
        newer_samples_weight=2,
        newer_samples_start=0.0,
        quantiles=None,
        impute_missing=True,
        impute_linear=10,
        impute_rolling=10,
        drop_missing=False,
        collect_metrics=True,
        normalize="auto",
        global_normalization=False,
        global_time_normalization=True,
        unknown_data_normalization=False,
    ):
        kwargs = locals()

        # General
        self.name = "NeuralProphet"
        self.n_forecasts = n_forecasts

        # Data Normalization settings
        self.config_normalization = configure.Normalization(
            normalize=normalize,
            global_normalization=global_normalization,
            global_time_normalization=global_time_normalization,
            unknown_data_normalization=unknown_data_normalization,
        )

        # Missing Data Preprocessing
        self.config_missing = configure.from_kwargs(configure.MissingDataHandling, kwargs)

        # Training
        self.config_train = configure.from_kwargs(configure.Train, kwargs)

        if len(self.config_train.quantiles) > 1:
            loss = metrics.LossMetric(self.config_train.loss_func.loss_func)
        else:
            loss = metrics.LossMetric(self.config_train.loss_func)

        if collect_metrics is None:
            collect_metrics = []
        elif collect_metrics is True:
            collect_metrics = ["mae", "rmse"]
        elif isinstance(collect_metrics, str):
            if not collect_metrics.lower() in METRICS.keys():
                raise ValueError("Received unsupported argument for collect_metrics.")
            collect_metrics = [collect_metrics]
        elif isinstance(collect_metrics, list):
            if not all([m.lower() in METRICS.keys() for m in collect_metrics]):
                raise ValueError("Received unsupported argument for collect_metrics.")
        elif collect_metrics is not False:
            raise ValueError("Received unsupported argument for collect_metrics.")

        self.metrics = None
        if isinstance(collect_metrics, list):
            self.metrics = metrics.MetricsCollection(
                metrics=[loss] + [METRICS[m.lower()]() for m in collect_metrics],
                value_metrics=[metrics.ValueMetric("Loss"), metrics.ValueMetric("RegLoss")],
            )

        # AR
        self.config_ar = configure.from_kwargs(configure.AR, kwargs)
        self.n_lags = self.config_ar.n_lags
        self.max_lags = self.n_lags

        # Model
        self.config_model = configure.from_kwargs(configure.Model, kwargs)

        # Trend
        self.config_trend = configure.from_kwargs(configure.Trend, kwargs)

        # Seasonality
        self.config_season = configure.AllSeason(
            mode=seasonality_mode,
            reg_lambda=seasonality_reg,
            yearly_arg=yearly_seasonality,
            weekly_arg=weekly_seasonality,
            daily_arg=daily_seasonality,
        )
        self.config_train.reg_lambda_season = self.config_season.reg_lambda

        # Events
        self.config_events = None
        self.config_country_holidays = None

        # Extra Regressors
        self.config_covar = None
        self.config_regressors = None

        # set during fit()
        self.data_freq = None

        # Set during _train()
        self.fitted = False
        self.data_params = None
        self.optimizer = None
        self.scheduler = None
        self.model = None

        # set during prediction
        self.future_periods = None
        self.predict_steps = self.n_forecasts
        # later set by user (optional)
        self.highlight_forecast_step_n = None
        self.true_ar_weights = None

    def add_lagged_regressor(self, names, n_lags="auto", regularization=None, normalize="auto"):
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
            regularization : float
                optional  scale for regularization strength
            normalize : bool
                optional, specify whether this regressor will benormalized prior to fitting.
                if ``auto``, binary regressors will not be normalized.
        """
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
            self._validate_column_name(name)
            if self.config_covar is None:
                self.config_covar = OrderedDict({})
            self.config_covar[name] = configure.Covar(
                reg_lambda=regularization,
                normalize=normalize,
                as_scalar=only_last_value,
                n_lags=n_lags,
            )
        return self

    def add_future_regressor(self, name, regularization=None, normalize="auto", mode="additive"):
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
        self._validate_column_name(name)

        if self.config_regressors is None:
            self.config_regressors = {}
        self.config_regressors[name] = configure.Regressor(reg_lambda=regularization, normalize=normalize, mode=mode)
        return self

    def add_events(self, events, lower_window=0, upper_window=0, regularization=None, mode="additive"):
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
            self._validate_column_name(event_name)
            self.config_events[event_name] = configure.Event(
                lower_window=lower_window, upper_window=upper_window, reg_lambda=regularization, mode=mode
            )
        return self

    def add_country_holidays(self, country_name, lower_window=0, upper_window=0, regularization=None, mode="additive"):
        """
        Add a country into the NeuralProphet object to include country specific holidays
        and create the corresponding configs such as lower, upper windows and the regularization
        parameters

        Holidays can only be added for a single country. Calling the function
        multiple times will override already added country holidays.

        Parameters
        ----------
            country_name : string
                name of the country
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

    def add_seasonality(self, name, period, fourier_order):
        """Add a seasonal component with specified period, number of Fourier components, and regularization.

        Increasing the number of Fourier components allows the seasonality to change more quickly
        (at risk of overfitting).
        Note: regularization and mode (additive/multiplicative) are set in the main init.

        Parameters
        ----------
            name : string
                name of the seasonality component.
            period : float
                number of days in one period.
            fourier_order : int
                number of Fourier components to use.

        """
        if self.fitted:
            raise Exception("Seasonality must be added prior to model fitting.")
        if name in ["daily", "weekly", "yearly"]:
            log.error("Please use inbuilt daily, weekly, or yearly seasonality or set another name.")
        # Do not Allow overwriting built-in seasonalities
        self._validate_column_name(name, seasons=True)
        if fourier_order <= 0:
            raise ValueError("Fourier Order must be > 0")
        self.config_season.append(name=name, period=period, resolution=fourier_order, arg="custom")
        return self

    def fit(self, df, freq="auto", validation_df=None, progress="bar", minimal=False):
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
                if provided, model with performance  will be evaluated after each training epoch over this data.
            epochs : int
                number of epochs to train (overrides default setting).
                default: if not specified, uses self.epochs
            progress : str
                Method of progress display

                Options
                    * (default) ``bar`` display updating progress bar (tqdm)
                    * ``print`` print out progress (fallback option)
                    * ``plot`` plot a live updating graph of the training loss, requires [live] install or livelossplot package installed.
                    * ``plot-all`` extended to all recorded metrics.
            minimal : bool
                whether to train without any printouts or metrics collection

        Returns
        -------
            pd.DataFrame
                metrics with training and potentially evaluation metrics
        """
        df, _, _, _ = df_utils.prep_or_copy_df(df)
        if self.fitted is True:
            log.error("Model has already been fitted. Re-fitting may break or produce different results.")
        self.max_lags = df_utils.get_max_num_lags(self.config_covar, self.n_lags)
        if self.max_lags == 0 and self.n_forecasts > 1:
            self.n_forecasts = 1
            self.predict_steps = 1
            log.warning(
                "Changing n_forecasts to 1. Without lags, the forecast can be "
                "computed for any future time, independent of lagged values"
            )
        df = self._check_dataframe(df, check_y=True, exogenous=True)
        self.data_freq = df_utils.infer_frequency(df, n_lags=self.max_lags, freq=freq)
        df = self._handle_missing_data(df, freq=self.data_freq)
        if validation_df is not None and (self.metrics is None or minimal):
            log.warning("Ignoring validation_df because no metrics set or minimal training set.")
            validation_df = None
        if validation_df is None:
            if minimal:
                self._train_minimal(df, progress_bar=progress == "bar")
                metrics_df = None
            else:
                metrics_df = self._train(df, progress=progress)
        else:
            df_val, _, _, _ = df_utils.prep_or_copy_df(validation_df)
            df_val = self._check_dataframe(df_val, check_y=False, exogenous=False)
            df_val = self._handle_missing_data(df_val, freq=self.data_freq)
            metrics_df = self._train(df, df_val=df_val, progress=progress)

        self.fitted = True
        return metrics_df

    def predict(self, df, decompose=True, raw=False):
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
        df, received_ID_col, received_single_time_series, received_dict = df_utils.prep_or_copy_df(df)
        # to get all forecasteable values with df given, maybe extend into future:
        df, periods_added = self._maybe_extend_df(df)
        df = self._prepare_dataframe_to_predict(df)
        # normalize
        df = self._normalize(df)
        forecast = pd.DataFrame()
        for df_name, df_i in df.groupby("ID"):
            dates, predicted, components = self._predict_raw(df_i, df_name, include_components=decompose)
            df_i = df_utils.drop_missing_from_df(
                df_i, self.config_missing.drop_missing, self.predict_steps, self.n_lags
            )
            if raw:
                fcst = self._convert_raw_predictions_to_raw_df(dates, predicted, components)
                if periods_added[df_name] > 0:
                    fcst = fcst[:-1]
            else:
                fcst = self._reshape_raw_predictions_to_forecst_df(df_i, predicted, components)
                if periods_added[df_name] > 0:
                    fcst = fcst[: -periods_added[df_name]]
            forecast = pd.concat((forecast, fcst), ignore_index=True)
        df = df_utils.return_df_in_original_format(
            forecast, received_ID_col, received_single_time_series, received_dict
        )
        self.predict_steps = self.n_forecasts
        return df

    def test(self, df):
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
        df = self._check_dataframe(df, check_y=True, exogenous=True)
        _ = df_utils.infer_frequency(df, n_lags=self.max_lags, freq=self.data_freq)
        df = self._handle_missing_data(df, freq=self.data_freq)
        loader = self._init_val_loader(df)
        val_metrics_df = self._evaluate(loader)
        if not self.config_normalization.global_normalization:
            log.warning("Note that the metrics are displayed in normalized scale because of local normalization.")
        return val_metrics_df

    def split_df(self, df, freq="auto", valid_p=0.2, local_split=False):
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
        df, received_ID_col, received_single_time_series, received_dict = df_utils.prep_or_copy_df(df)
        df = self._check_dataframe(df, check_y=False, exogenous=False)
        freq = df_utils.infer_frequency(df, n_lags=self.max_lags, freq=freq)
        df = self._handle_missing_data(df, freq=freq, predicting=False)
        df_train, df_val = df_utils.split_df(
            df,
            n_lags=self.max_lags,
            n_forecasts=self.n_forecasts,
            valid_p=valid_p,
            inputs_overbleed=True,
            local_split=local_split,
        )
        df_train = df_utils.return_df_in_original_format(
            df_train, received_ID_col, received_single_time_series, received_dict
        )
        df_val = df_utils.return_df_in_original_format(
            df_val, received_ID_col, received_single_time_series, received_dict
        )
        return df_train, df_val

    def crossvalidation_split_df(
        self, df, freq="auto", k=5, fold_pct=0.1, fold_overlap_pct=0.5, global_model_cv_type="global-time"
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
        df = self._check_dataframe(df, check_y=False, exogenous=False)
        freq = df_utils.infer_frequency(df, n_lags=self.max_lags, freq=freq)
        df = self._handle_missing_data(df, freq=freq, predicting=False)
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

    def double_crossvalidation_split_df(self, df, freq="auto", k=5, valid_pct=0.10, test_pct=0.10):
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
        df = self._check_dataframe(df, check_y=False, exogenous=False)
        freq = df_utils.infer_frequency(df, n_lags=self.max_lags, freq=freq)
        df = self._handle_missing_data(df, freq=freq, predicting=False)
        folds_val, folds_test = df_utils.double_crossvalidation_split_df(
            df,
            n_lags=self.max_lags,
            n_forecasts=self.n_forecasts,
            k=k,
            valid_pct=valid_pct,
            test_pct=test_pct,
        )
        return folds_val, folds_test

    def create_df_with_events(self, df, events_df):
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
        df, received_ID_col, received_single_time_series, received_dict = df_utils.prep_or_copy_df(df)
        df = self._check_dataframe(df, check_y=True, exogenous=False)
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
        df = df_utils.return_df_in_original_format(
            df_created, received_ID_col, received_single_time_series, received_dict
        )
        return df

    def make_future_dataframe(self, df, events_df=None, regressors_df=None, periods=None, n_historic_predictions=False):
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
        df, received_ID_col, received_single_time_series, received_dict = df_utils.prep_or_copy_df(df)
        events_dict = df_utils.create_dict_for_events_or_regressors(df, events_df, "events")
        regressors_dict = df_utils.create_dict_for_events_or_regressors(df, regressors_df, "regressors")

        df_future_dataframe = pd.DataFrame()
        for df_name, df_i in df.groupby("ID"):
            df_aux = self._make_future_dataframe(
                df=df_i,
                events_df=events_dict[df_name],
                regressors_df=regressors_dict[df_name],
                periods=periods,
                n_historic_predictions=n_historic_predictions,
            )
            df_aux["ID"] = df_name
            df_future_dataframe = pd.concat((df_future_dataframe, df_aux), ignore_index=True)

        df_future = df_utils.return_df_in_original_format(
            df_future_dataframe, received_ID_col, received_single_time_series, received_dict
        )
        return df_future

    def handle_negative_values(self, df, handle="remove", columns=None):
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

    def predict_trend(self, df, quantile=0.5):
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

        df, received_ID_col, received_single_time_series, received_dict = df_utils.prep_or_copy_df(df)
        df = self._check_dataframe(df, check_y=False, exogenous=False)
        df = self._normalize(df)
        df_trend = pd.DataFrame()
        for df_name, df_i in df.groupby("ID"):
            t = torch.from_numpy(np.expand_dims(df_i["t"].values, 1))
            quantile_index = self.config_train.quantiles.index(quantile)
            trend = self.model.trend(t).detach().numpy()[:, :, quantile_index].squeeze()
            data_params = self.config_normalization.get_data_params(df_name)
            trend = trend * data_params["y"].scale + data_params["y"].shift
            df_aux = pd.DataFrame({"ds": df_i["ds"], "trend": trend, "ID": df_name})
            df_trend = pd.concat((df_trend, df_aux), ignore_index=True)
        df = df_utils.return_df_in_original_format(
            df_trend, received_ID_col, received_single_time_series, received_dict
        )
        return df

    def predict_seasonal_components(self, df, quantile=0.5):
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

        df, received_ID_col, received_single_time_series, received_dict = df_utils.prep_or_copy_df(df)
        df = self._check_dataframe(df, check_y=False, exogenous=False)
        df = self._normalize(df)
        df_seasonal = pd.DataFrame()
        for df_name, df_i in df.groupby("ID"):
            dataset = time_dataset.TimeDataset(
                df_i,
                name=df_name,
                config_season=self.config_season,
                # n_lags=0,
                # n_forecasts=1,
                predict_steps=self.predict_steps,
                predict_mode=True,
                config_missing=self.config_missing,
            )
            loader = DataLoader(dataset, batch_size=min(4096, len(df)), shuffle=False, drop_last=False)
            predicted = {}
            for name in self.config_season.periods:
                predicted[name] = list()
            for inputs, _, _ in loader:
                for name in self.config_season.periods:
                    features = inputs["seasonalities"][name]
                    quantile_index = self.config_train.quantiles.index(quantile)
                    y_season = torch.squeeze(self.model.seasonality(features=features, name=name)[:, :, quantile_index])
                    predicted[name].append(y_season.data.numpy())

            for name in self.config_season.periods:
                predicted[name] = np.concatenate(predicted[name])
                if self.config_season.mode == "additive":
                    data_params = self.config_normalization.get_data_params(df_name)
                    predicted[name] = predicted[name] * data_params["y"].scale
            df_aux = pd.DataFrame({"ds": df_i["ds"], "ID": df_i["ID"], **predicted})
            df_seasonal = pd.concat((df_seasonal, df_aux), ignore_index=True)
        df = df_utils.return_df_in_original_format(
            df_seasonal, received_ID_col, received_single_time_series, received_dict
        )
        return df

    def set_true_ar_for_eval(self, true_ar_weights):
        """Configures model to evaluate closeness of AR weights to true weights.

        Parameters
        ----------
            true_ar_weights : np.array
                true AR-parameters, if known.
        """
        self.true_ar_weights = true_ar_weights

    def set_plotting_backend(self, plotting_backend):
        """Set plotting backend.

        Parameters
        ----------
            plotting_backend : str
            Specifies plotting backend to use for all plots. Can be configured individually for each plot.

            Options
                * ``plotly``: Use the plotly backend for plotting
                * (default) ``matplotlib``: use matplotlib for plotting
        """
        if plotting_backend in ["plotly", "matplotlib"]:
            self.plotting_backend = plotting_backend
            if self.plotting_backend == "matplotlib":
                log.warning(
                    "DeprecationWarning: matplotlib as plotting backend will be deprecated in a future version. Switch to plotly by calling `m.set_plotting_backend('plotly')`."
                )
        else:
            raise ValueError("The parameter `plotting_backend` must be either 'plotly' or 'matplotlib'.")

    def highlight_nth_step_ahead_of_each_forecast(self, step_number=None):
        """Set which forecast step to focus on for metrics evaluation and plotting.

        Parameters
        ----------
            step_number : int
                i-th step ahead forecast to use for statistics and plotting.
        """
        if step_number is not None:
            assert step_number <= self.n_forecasts
        self.highlight_forecast_step_n = step_number
        return self

    def plot(self, fcst, df_name=None, ax=None, xlabel="ds", ylabel="y", figsize=(10, 6), plotting_backend="default"):
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
                * ``plotly``: Use plotly for plotting
                * ``matplotlib``: use matplotlib for plotting
                * (default) ``default``: use the global default for plotting
        """
        fcst, received_ID_col, received_single_time_series, received_dict = df_utils.prep_or_copy_df(fcst)
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
            if self.highlight_forecast_step_n is None and self.n_lags != 0:
                raise ValueError(
                    "Please specify step_number using the highlight_nth_step_ahead_of_each_forecast function"
                    " for quantiles plotting when auto-regression enabled."
                )
            if self.highlight_forecast_step_n is not None and self.n_lags == 0:
                log.warning("highlight_forecast_step_n is ignored since auto-regression not enabled.")
                self.highlight_forecast_step_n = None
        if self.max_lags > 0:
            num_forecasts = sum(fcst["yhat1"].notna())
            if num_forecasts < self.n_forecasts:
                log.warning(
                    "Too few forecasts to plot a line per forecast step." "Plotting a line per forecast origin instead."
                )
                return self.plot_last_forecast(
                    fcst,
                    ax=ax,
                    df_name=df_name,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    figsize=figsize,
                    include_previous_forecasts=num_forecasts - 1,
                    plot_history_data=True,
                )

        # Check whether the default plotting backend is overwritten
        plotting_backend = (
            plotting_backend
            if plotting_backend != "default"
            else (self.plotting_backend if hasattr(self, "plotting_backend") else "matplotlib")
        )
        if plotting_backend == "plotly":
            return plot_plotly(
                fcst=fcst,
                quantiles=self.config_train.quantiles,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=tuple(x * 70 for x in figsize),
                highlight_forecast=self.highlight_forecast_step_n,
            )
        else:
            return plot(
                fcst=fcst,
                quantiles=self.config_train.quantiles,
                ax=ax,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=figsize,
                highlight_forecast=self.highlight_forecast_step_n,
            )

    def get_latest_forecast(
        self,
        fcst,
        df_name=None,
        include_history_data=False,
        include_previous_forecasts=0,
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
                columns ``ds``, ``y``, and [``yhat<i>``]

                Note
                ----
                where yhat<i> refers to the i-step-ahead prediction for this row's datetime.
                e.g. yhat3 is the prediction for this datetime, predicted 3 steps ago, "3 steps old".
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
        fcst, received_ID_col, received_single_time_series, received_dict = df_utils.prep_or_copy_df(fcst)
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
        fcst = utils.fcst_df_to_last_forecast(fcst, self.config_train.quantiles, n_last=1 + include_previous_forecasts)
        return fcst

    def plot_last_forecast(
        self,
        fcst,
        df_name=None,
        ax=None,
        xlabel="ds",
        ylabel="y",
        figsize=(10, 6),
        include_previous_forecasts=0,
        plot_history_data=None,
        plotting_backend="default",
    ):
        """Plot the NeuralProphet forecast, including history.

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
                * ``plotly``: Use plotly for plotting
                * ``matplotlib``: use matplotlib for plotting
                * (default) ``default``: use the global default for plotting
        Returns
        -------
            matplotlib.axes.Axes
                plot of NeuralProphet forecasting
        """
        if self.max_lags == 0:
            raise ValueError("Use the standard plot function for models without lags.")
        fcst, received_ID_col, received_single_time_series, received_dict = df_utils.prep_or_copy_df(fcst)
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
                "Plotting last forecasts when uncertainty estimation enabled"
                " plots the forecasts only for the median quantile."
            )
        if plot_history_data is None:
            fcst = fcst[-(include_previous_forecasts + self.n_forecasts + self.max_lags) :]
        elif plot_history_data is False:
            fcst = fcst[-(include_previous_forecasts + self.n_forecasts) :]
        elif plot_history_data is True:
            fcst = fcst
        fcst = utils.fcst_df_to_last_forecast(fcst, self.config_train.quantiles, n_last=1 + include_previous_forecasts)

        # Check whether the default plotting backend is overwritten
        plotting_backend = (
            plotting_backend
            if plotting_backend != "default"
            else (self.plotting_backend if hasattr(self, "plotting_backend") else "matplotlib")
        )
        if plotting_backend == "plotly":
            return plot_plotly(
                fcst=fcst,
                quantiles=self.config_train.quantiles,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=tuple(x * 70 for x in figsize),
                highlight_forecast=self.highlight_forecast_step_n,
                line_per_origin=True,
            )
        else:
            return plot(
                fcst=fcst,
                quantiles=self.config_train.quantiles,
                ax=ax,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=figsize,
                highlight_forecast=self.highlight_forecast_step_n,
                line_per_origin=True,
            )

    def plot_components(
        self, fcst, df_name=None, figsize=None, forecast_in_focus=None, residuals=False, plotting_backend="default"
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
            plotting_backend : str
                optional, overwrites the default plotting backend.

                Options
                * ``plotly``: Use plotly for plotting
                * ``matplotlib``: use matplotlib for plotting
                * (default) ``default``: use the global default for plotting

        Returns
        -------
            matplotlib.axes.Axes
                plot of NeuralProphet components
        """
        fcst, received_ID_col, received_single_time_series, received_dict = df_utils.prep_or_copy_df(fcst)
        if not received_single_time_series:
            if df_name not in fcst["ID"].unique():
                assert len(fcst["ID"].unique()) > 1
                raise Exception(
                    "Many time series are present in the pd.DataFrame (more than one ID). Please, especify ID to be plotted."
                )
            else:
                fcst = fcst[fcst["ID"] == df_name].copy(deep=True)
                log.info(f"Plotting data from ID {df_name}")

        # Check whether the default plotting backend is overwritten
        plotting_backend = (
            plotting_backend
            if plotting_backend != "default"
            else (self.plotting_backend if hasattr(self, "plotting_backend") else "matplotlib")
        )
        if plotting_backend == "plotly":
            return plot_components_plotly(
                m=self,
                fcst=fcst,
                figsize=tuple(x * 70 for x in figsize) if figsize else (700, 210),
                forecast_in_focus=forecast_in_focus if forecast_in_focus else self.highlight_forecast_step_n,
                residuals=residuals,
            )
        else:
            return plot_components(
                m=self,
                fcst=fcst,
                quantile=self.config_train.quantiles[0],  # plot components only for median quantile
                figsize=figsize,
                forecast_in_focus=forecast_in_focus if forecast_in_focus else self.highlight_forecast_step_n,
                residuals=residuals,
            )

    def plot_parameters(
        self,
        weekly_start=0,
        yearly_start=0,
        figsize=None,
        forecast_in_focus=None,
        df_name=None,
        plotting_backend="default",
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
            plotting_backend : str
                optional, overwrites the default plotting backend.

                Options
                * ``plotly``: Use plotly for plotting
                * ``matplotlib``: use matplotlib for plotting
                * (default) ``default``: use the global default for plotting

        Returns
        -------
            matplotlib.axes.Axes
                plot of NeuralProphet forecasting
        """

        # Check whether the default plotting backend is overwritten
        plotting_backend = (
            plotting_backend
            if plotting_backend != "default"
            else (self.plotting_backend if hasattr(self, "plotting_backend") else "matplotlib")
        )
        if plotting_backend == "plotly":
            return plot_parameters_plotly(
                m=self,
                forecast_in_focus=forecast_in_focus if forecast_in_focus else self.highlight_forecast_step_n,
                weekly_start=weekly_start,
                yearly_start=yearly_start,
                figsize=tuple(x * 70 for x in figsize) if figsize else (700, 210),
                df_name=df_name,
            )
        else:
            return plot_parameters(
                m=self,
                quantile=self.config_train.quantiles[0],  # plot components only for median quantile
                forecast_in_focus=forecast_in_focus if forecast_in_focus else self.highlight_forecast_step_n,
                weekly_start=weekly_start,
                yearly_start=yearly_start,
                figsize=figsize,
                df_name=df_name,
            )

    def _init_model(self):
        """Build Pytorch model with configured hyperparamters.

        Returns
        -------
            TimeNet model
        """
        self.model = time_net.TimeNet(
            config_trend=self.config_trend,
            config_season=self.config_season,
            config_covar=self.config_covar,
            config_regressors=self.config_regressors,
            config_events=self.config_events,
            config_holidays=self.config_country_holidays,
            n_forecasts=self.n_forecasts,
            n_lags=self.n_lags,
            num_hidden_layers=self.config_model.num_hidden_layers,
            d_hidden=self.config_model.d_hidden,
            quantiles=self.config_train.quantiles,
        )
        log.debug(self.model)
        return self.model

    def _create_dataset(self, df, predict_mode):
        """Construct dataset from dataframe.

        (Configured Hyperparameters can be overridden by explicitly supplying them.
        Useful to predict a single model component.)

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` and
                normalized columns normalized columns ``ds``, ``y``, ``t``, ``y_scaled``
            predict_mode : bool
                specifies predict mode

                Options
                    * ``False``: includes target values.
                    * ``True``: does not include targets but includes entire dataset as input

        Returns
        -------
            TimeDataset
        """
        df, _, _, _ = df_utils.prep_or_copy_df(df)
        return time_dataset.GlobalTimeDataset(
            df,
            predict_mode=predict_mode,
            n_lags=self.n_lags,
            n_forecasts=self.n_forecasts,
            predict_steps=self.predict_steps,
            config_season=self.config_season,
            config_events=self.config_events,
            config_country_holidays=self.config_country_holidays,
            config_covar=self.config_covar,
            config_regressors=self.config_regressors,
            config_missing=self.config_missing,
        )

    def __handle_missing_data(self, df, freq, predicting):
        """Checks and normalizes new data

        Data is also auto-imputed, unless impute_missing is set to ``False``.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y`` with all data
            freq : str
                data step sizes. Frequency of data recording,

                Note
                ----
                Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto`` (default) to automatically set frequency.
            predicting : bool
                when no lags, allow NA values in ``y`` of forecast series or ``y`` to miss completely

        Returns
        -------
            pd.DataFrame
                preprocessed dataframe
        """
        # Receives df with single ID column
        assert len(df["ID"].unique()) == 1
        if self.n_lags == 0 and not predicting:
            # we can drop rows with NA in y
            sum_na = sum(df["y"].isna())
            if sum_na > 0:
                df = df[df["y"].notna()]
                log.info(f"dropped {sum_na} NAN row in 'y'")

        # add missing dates for autoregression modelling
        if self.n_lags > 0:
            df, missing_dates = df_utils.add_missing_dates_nan(df, freq=freq)
            if missing_dates > 0:
                if self.config_missing.impute_missing:
                    log.info(f"{missing_dates} missing dates added.")
                # FIX Issue#52
                # Comment error raising to allow missing data for autoregression flow.
                # else:
                #     raise ValueError(f"{missing_dates} missing dates found. Please preprocess data manually or set impute_missing to True.")
                # END FIX

        if self.config_regressors is not None:
            # if future regressors, check that they are not nan at end, else drop
            # we ignore missing events, as those will be filled in with zeros.
            reg_nan_at_end = 0
            for col, regressor in self.config_regressors.items():
                # check for completeness of the regressor values
                col_nan_at_end = 0
                while len(df) > col_nan_at_end and df[col].isnull().iloc[-(1 + col_nan_at_end)]:
                    col_nan_at_end += 1
                reg_nan_at_end = max(reg_nan_at_end, col_nan_at_end)
            if reg_nan_at_end > 0:
                # drop rows at end due to missing future regressors
                df = df[:-reg_nan_at_end]
                log.info("Dropped {reg_nan_at_end} rows at end due to missing future regressor values.")

        df_end_to_append = None
        nan_at_end = 0
        while len(df) > nan_at_end and df["y"].isnull().iloc[-(1 + nan_at_end)]:
            nan_at_end += 1
        if nan_at_end > 0:
            if predicting:
                # allow nans at end - will re-add at end
                if self.n_forecasts > 1 and self.n_forecasts < nan_at_end:
                    # check that not more than n_forecasts nans, else drop surplus
                    df = df[: -(nan_at_end - self.n_forecasts)]
                    # correct new length:
                    nan_at_end = self.n_forecasts
                    log.info(
                        "Detected y to have more NaN values than n_forecast can predict. "
                        f"Dropped {nan_at_end - self.n_forecasts} rows at end."
                    )
                df_end_to_append = df[-nan_at_end:]
                df = df[:-nan_at_end]
            else:
                # training - drop nans at end
                df = df[:-nan_at_end]
                log.info(
                    f"Dropped {nan_at_end} consecutive nans at end. "
                    "Training data can only be imputed up to last observation."
                )

        # impute missing values
        data_columns = []
        if self.n_lags > 0:
            data_columns.append("y")
        if self.config_covar is not None:
            data_columns.extend(self.config_covar.keys())
        if self.config_regressors is not None:
            data_columns.extend(self.config_regressors.keys())
        if self.config_events is not None:
            data_columns.extend(self.config_events.keys())
        for column in data_columns:
            sum_na = sum(df[column].isnull())
            if sum_na > 0:
                log.warning(f"{sum_na} missing values in column {column} were detected in total. ")
                if self.config_missing.impute_missing:
                    # use 0 substitution for holidays and events missing values
                    if self.config_events is not None and column in self.config_events.keys():
                        df[column].fillna(0, inplace=True)
                        remaining_na = 0
                    else:
                        df.loc[:, column], remaining_na = df_utils.fill_linear_then_rolling_avg(
                            df[column],
                            limit_linear=self.config_missing.impute_linear,
                            rolling=self.config_missing.impute_rolling,
                        )
                    log.info(f"{sum_na - remaining_na} NaN values in column {column} were auto-imputed.")
                    if remaining_na > 0:
                        log.warning(
                            f"More than {2 * self.config_missing.impute_linear + self.config_missing.impute_rolling} consecutive missing values encountered in column {column}. "
                            f"{remaining_na} NA remain after auto-imputation. "
                        )
                # FIX Issue#52
                # Comment error raising to allow missing data for autoregression flow.
                # else:  # fail because set to not impute missing
                #    raise ValueError(
                #        "Missing values found. " "Please preprocess data manually or set impute_missing to True."
                #    )
                # END FIX
        if df_end_to_append is not None:
            df = pd.concat([df, df_end_to_append])
        return df

    def _handle_missing_data(self, df, freq, predicting=False):
        """Checks and normalizes new data

        Data is also auto-imputed, unless impute_missing is set to ``False``.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
            freq : str
                data step sizes. Frequency of data recording,

                Note
                ----
                Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto`` (default) to automatically set frequency.
            predicting (bool): when no lags, allow NA values in ``y`` of forecast series or ``y`` to miss completely

        Returns
        -------
            pre-processed df
        """
        df, _, _, _ = df_utils.prep_or_copy_df(df)
        df_handled_missing = pd.DataFrame()
        for df_name, df_i in df.groupby("ID"):
            df_handled_missing_aux = self.__handle_missing_data(df_i, freq, predicting).copy(deep=True)
            df_handled_missing_aux["ID"] = df_name
            df_handled_missing = pd.concat((df_handled_missing, df_handled_missing_aux), ignore_index=True)
        return df_handled_missing

    def _check_dataframe(self, df, check_y=True, exogenous=True):
        """Performs basic data sanity checks and ordering

        Prepare dataframe for fitting or predicting.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
            check_y : bool
                if df must have series values

                Note
                ----
                set to True if training or predicting with autoregression
            exogenous : bool
                whether to check covariates, regressors and events column names

        Returns
        -------
            pd.DataFrame
                checked dataframe
        """
        df, _, _, _ = df_utils.prep_or_copy_df(df)
        return df_utils.check_dataframe(
            df=df,
            check_y=check_y,
            covariates=self.config_covar if exogenous else None,
            regressors=self.config_regressors if exogenous else None,
            events=self.config_events if exogenous else None,
        )

    def _validate_column_name(self, name, events=True, seasons=True, regressors=True, covariates=True):
        """Validates the name of a seasonality, event, or regressor.

        Parameters
        ----------
            name : str
                name of seasonality, event or regressor
            events : bool
                check if name already used for event
            seasons : bool
                check if name already used for seasonality
            regressors : bool
                check if name already used for regressor
        """
        reserved_names = [
            "trend",
            "additive_terms",
            "daily",
            "weekly",
            "yearly",
            "events",
            "holidays",
            "zeros",
            "extra_regressors_additive",
            "yhat",
            "extra_regressors_multiplicative",
            "multiplicative_terms",
            "ID",
        ]
        rn_l = [n + "_lower" for n in reserved_names]
        rn_u = [n + "_upper" for n in reserved_names]
        reserved_names.extend(rn_l)
        reserved_names.extend(rn_u)
        reserved_names.extend(["ds", "y", "cap", "floor", "y_scaled", "cap_scaled"])
        if name in reserved_names:
            raise ValueError(f"Name {name!r} is reserved.")
        if events and self.config_events is not None:
            if name in self.config_events.keys():
                raise ValueError(f"Name {name!r} already used for an event.")
        if events and self.config_country_holidays is not None:
            if name in self.config_country_holidays.holiday_names:
                raise ValueError(f"Name {name!r} is a holiday name in {self.config_country_holidays.country}.")
        if seasons and self.config_season is not None:
            if name in self.config_season.periods:
                raise ValueError(f"Name {name!r} already used for a seasonality.")
        if covariates and self.config_covar is not None:
            if name in self.config_covar:
                raise ValueError(f"Name {name!r} already used for an added covariate.")
        if regressors and self.config_regressors is not None:
            if name in self.config_regressors.keys():
                raise ValueError(f"Name {name!r} already used for an added regressor.")

    def _normalize(self, df):
        """Apply data scales.

        Applies data scaling factors to df using data_params.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data

        Returns
        -------
            df: pd.DataFrame, normalized
        """
        df, _, _, _ = df_utils.prep_or_copy_df(df)
        df_norm = pd.DataFrame()
        for df_name, df_i in df.groupby("ID"):
            data_params = self.config_normalization.get_data_params(df_name)
            df_i.drop("ID", axis=1, inplace=True)
            df_aux = df_utils.normalize(df_i, data_params).copy(deep=True)
            df_aux["ID"] = df_name
            df_norm = pd.concat((df_norm, df_aux), ignore_index=True)
        return df_norm

    def _init_train_loader(self, df):
        """Executes data preparation steps and initiates training procedure.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data

        Returns
        -------
            torch DataLoader
        """
        df, _, _, _ = df_utils.prep_or_copy_df(df)
        # if not self.fitted:
        self.config_normalization.init_data_params(
            df=df,
            config_covariates=self.config_covar,
            config_regressor=self.config_regressors,
            config_events=self.config_events,
        )

        df = self._normalize(df)
        # if not self.fitted:
        if self.config_trend.changepoints is not None:
            # scale user-specified changepoint times
            df_aux = pd.DataFrame({"ds": pd.Series(self.config_trend.changepoints)})
            self.config_trend.changepoints = self._normalize(df_aux)["t"].values

        # df_merged, _ = df_utils.join_dataframes(df)
        # df_merged = df_merged.sort_values("ds")
        # df_merged.drop_duplicates(inplace=True, keep="first", subset=["ds"])
        df_merged = df_utils.merge_dataframes(df)
        self.config_season = utils.set_auto_seasonalities(df_merged, config_season=self.config_season)
        if self.config_country_holidays is not None:
            self.config_country_holidays.init_holidays(df_merged)

        dataset = self._create_dataset(df, predict_mode=False)  # needs to be called after set_auto_seasonalities
        self.config_train.set_auto_batch_epoch(n_data=len(dataset))

        loader = DataLoader(dataset, batch_size=self.config_train.batch_size, shuffle=True)

        # if not self.fitted:
        self.model = self._init_model()  # needs to be called after set_auto_seasonalities

        if self.config_train.learning_rate is None:
            self.config_train.learning_rate = self.config_train.find_learning_rate(self.model, dataset)
            log.info(f"lr-range-test selected learning rate: {self.config_train.learning_rate:.2E}")
        self.optimizer = self.config_train.get_optimizer(self.model.parameters())
        self.scheduler = self.config_train.get_scheduler(self.optimizer, steps_per_epoch=len(loader))
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
        df = self._normalize(df)
        dataset = self._create_dataset(df, predict_mode=False)
        loader = DataLoader(dataset, batch_size=min(1024, len(dataset)), shuffle=False, drop_last=False)
        return loader

    def _get_time_based_sample_weight(self, t):
        weight = torch.ones_like(t)
        if self.config_train.newer_samples_weight > 1.0:
            end_w = self.config_train.newer_samples_weight
            start_t = self.config_train.newer_samples_start
            time = (t.detach() - start_t) / (1.0 - start_t)
            time = torch.maximum(torch.zeros_like(time), time)
            time = torch.minimum(torch.ones_like(time), time)  # time = 0 to 1
            time = np.pi * (time - 1.0)  # time =  -pi to 0
            time = 0.5 * torch.cos(time) + 0.5  # time =  0 to 1
            # scales end to be end weight times bigger than start weight
            # with end weight being 1.0
            weight = (1.0 + time * (end_w - 1.0)) / end_w
        return weight.unsqueeze(dim=2)  # add an extra dimension for the quantiles

    def _train_epoch(self, e, loader):
        """Make one complete iteration over all samples in dataloader and update model after each batch.

        Parameters
        ----------
            e : int
                current epoch number
            loader : torch DataLoader
                Training Dataloader
        """
        self.model.train()
        for i, (inputs, targets, meta) in enumerate(loader):
            # Run forward calculation
            predicted = self.model.forward(inputs)
            # store predictions in self for later network visualization
            self.train_epoch_prediction = predicted
            # Compute loss. no reduction.
            loss = self.config_train.loss_func(predicted, targets)
            # Weigh newer samples more.
            loss = loss * self._get_time_based_sample_weight(t=inputs["time"])
            loss = loss.sum(dim=2).mean()
            # Regularize.
            loss, reg_loss = self._add_batch_regularizations(loss, e, i / float(len(loader)))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if self.metrics is not None:
                self.metrics.update(
                    predicted=predicted.detach()[:, :, 0],
                    target=targets.detach().squeeze(dim=2),
                    values={"Loss": loss, "RegLoss": reg_loss},
                )  # compute metrics only for the median quantile (index 0)
        if self.metrics is not None:
            return self.metrics.compute(save=True)
        else:
            return None

    def _add_batch_regularizations(self, loss, e, iter_progress):
        """Add regularization terms to loss, if applicable

        Parameters
        ----------
            loss : torch.Tensor, scalar
                current batch loss
            e : int
                current epoch number
            iter_progress : float
                this epoch's progress of iterating over dataset [0, 1]

        Returns
        -------
            loss, reg_loss
        """
        delay_weight = self.config_train.get_reg_delay_weight(e, iter_progress)

        reg_loss = torch.zeros(1, dtype=torch.float, requires_grad=False)
        if delay_weight > 0:
            # Add regularization of AR weights - sparsify
            if self.max_lags > 0 and self.config_ar.reg_lambda is not None:
                reg_ar = self.config_ar.regularize(self.model.ar_weights)
                reg_ar = torch.sum(reg_ar).squeeze() / self.n_forecasts
                reg_loss += self.config_ar.reg_lambda * reg_ar

            # Regularize trend to be smoother/sparse
            l_trend = self.config_trend.trend_reg
            if self.config_trend.n_changepoints > 0 and l_trend is not None and l_trend > 0:
                reg_trend = utils.reg_func_trend(
                    weights=self.model.get_trend_deltas,
                    threshold=self.config_train.trend_reg_threshold,
                )
                reg_loss += l_trend * reg_trend

            # Regularize seasonality: sparsify fourier term coefficients
            l_season = self.config_train.reg_lambda_season
            if self.model.season_dims is not None and l_season is not None and l_season > 0:
                for name in self.model.season_params.keys():
                    reg_season = utils.reg_func_season(self.model.season_params[name])
                    reg_loss += l_season * reg_season

            # Regularize events: sparsify events features coefficients
            if self.config_events is not None or self.config_country_holidays is not None:
                reg_events_loss = utils.reg_func_events(self.config_events, self.config_country_holidays, self.model)
                reg_loss += reg_events_loss

            # Regularize lagged regressors: sparsify covariate features coefficients
            if self.config_covar is not None:
                reg_covariate_loss = utils.reg_func_covariates(self.config_covar, self.model)
                reg_loss += reg_covariate_loss

            # Regularize future regressors: sparsify regressor features coefficients
            if self.config_regressors is not None:
                reg_regressor_loss = utils.reg_func_regressors(self.config_regressors, self.model)
                reg_loss += reg_regressor_loss

        reg_loss = delay_weight * reg_loss
        loss = loss + reg_loss
        return loss, reg_loss

    def _evaluate_epoch(self, loader, val_metrics):
        """Evaluates model performance.

        Parameters
        ----------
            loader : torch DataLoader
                instantiated Validation Dataloader (with TimeDataset)
            val_metrics : MetricsCollection
                alidation metrics to be computed.

        Returns
        -------
            dict with evaluation metrics
        """
        with torch.no_grad():
            self.model.eval()
            for inputs, targets, meta in loader:
                predicted = self.model.forward(inputs)
                val_metrics.update(
                    predicted=predicted.detach()[:, :, 0], target=targets.detach().squeeze()
                )  # compute metrics only for the median quantile
            val_metrics = val_metrics.compute(save=True)
        return val_metrics

    def _train(self, df, df_val=None, progress="bar"):
        """Execute model training procedure for a configured number of epochs.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
            df_val : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with validation data
            progress : str
                Method of progress display.

                Options
                    * (default) ``bar`` display updating progress bar (tqdm)
                    * ``print`` print out progress (fallback option)
                    * ``plot`` plot a live updating graph of the training loss, requires [live] install or livelossplot package installed.
                    * ``plot-all`` "plot" extended to all recorded metrics.

        Returns
        -------
            pd.DataFrame
                metrics
        """
        df, _, _, _ = df_utils.prep_or_copy_df(df)
        if df_val is not None:
            df_val, _, _, _ = df_utils.prep_or_copy_df(df_val)
        # parse progress arg
        progress_bar = False
        progress_print = False
        plot_live_loss = False
        plot_live_all_metrics = False
        if progress.lower() == "bar":
            progress_bar = True
        elif progress.lower() == "print":
            progress_print = True
        elif progress.lower() == "plot":
            plot_live_loss = True
        elif progress.lower() in ["plot-all", "plotall", "plot all"]:
            plot_live_loss = True
            plot_live_all_metrics = True
        elif not progress.lower() == "none":
            raise ValueError(f"received unexpected value for progress {progress}")

        if self.metrics is None:
            log.info("No progress prints or plots possible because metrics are deactivated.")
            if df_val is not None:
                log.warning("Ignoring supplied df_val as no metrics are specified.")
            if plot_live_loss or plot_live_all_metrics:
                log.warning("Can not plot live loss as no metrics are specified.")
                progress_bar = True
            if progress_print:
                log.warning("Can not print progress as no metrics are specified.")
            return self._train_minimal(df, progress_bar=progress_bar)

        # set up data loader
        loader = self._init_train_loader(df)
        # set up Metrics
        if self.highlight_forecast_step_n is not None:
            self.metrics.add_specific_target(target_pos=self.highlight_forecast_step_n - 1)
        if not self.config_normalization.global_normalization:
            log.warning("When Global modeling with local normalization, metrics are displayed in normalized scale.")
        else:
            if not self.config_normalization.normalize == "off":
                self.metrics.set_shift_scale(
                    (
                        self.config_normalization.global_data_params["y"].shift,
                        self.config_normalization.global_data_params["y"].scale,
                    )
                )

        validate = df_val is not None
        if validate:
            val_loader = self._init_val_loader(df_val)
            val_metrics = metrics.MetricsCollection([m.new() for m in self.metrics.batch_metrics])

        # set up printing and plotting
        if plot_live_loss:
            try:
                from livelossplot import PlotLosses

                live_out = ["MatplotlibPlot"]
                if not progress_bar:
                    live_out.append("ExtremaPrinter")
                live_loss = PlotLosses(outputs=live_out)
                plot_live_loss = True
            except:
                log.warning(
                    "To plot live loss, please install neuralprophet[live]."
                    "Using pip: 'pip install neuralprophet[live]'"
                    "Or install the missing package manually: 'pip install livelossplot'",
                    exc_info=True,
                )
                plot_live_loss = False
                progress_bar = True
        if progress_bar:
            training_loop = tqdm(
                range(self.config_train.epochs),
                total=self.config_train.epochs,
                leave=log.getEffectiveLevel() <= 20,
            )
        else:
            training_loop = range(self.config_train.epochs)

        start = time.time()
        # run training loop
        for e in training_loop:
            metrics_live = OrderedDict({})
            self.metrics.reset()
            if validate:
                val_metrics.reset()
            # run epoch
            epoch_metrics = self._train_epoch(e, loader)
            # collect metrics
            if validate:
                val_epoch_metrics = self._evaluate_epoch(val_loader, val_metrics)
                print_val_epoch_metrics = {k + "_val": v for k, v in val_epoch_metrics.items()}
            else:
                val_epoch_metrics = None
                print_val_epoch_metrics = OrderedDict({})
            # print metrics
            if progress_bar:
                training_loop.set_description(f"Epoch[{(e+1)}/{self.config_train.epochs}]")
                training_loop.set_postfix(ordered_dict=epoch_metrics, **print_val_epoch_metrics)
            elif progress_print:
                metrics_string = utils.print_epoch_metrics(epoch_metrics, e=e, val_metrics=val_epoch_metrics)
                if e == 0:
                    log.info(metrics_string.splitlines()[0])
                    log.info(metrics_string.splitlines()[1])
                else:
                    log.info(metrics_string.splitlines()[1])
            # plot metrics
            if plot_live_loss:
                metrics_train = list(epoch_metrics)
                metrics_live[f"log-{metrics_train[0]}"] = np.log(epoch_metrics[metrics_train[0]])
                if plot_live_all_metrics and len(metrics_train) > 1:
                    for i in range(1, len(metrics_train)):
                        metrics_live[f"{metrics_train[i]}"] = epoch_metrics[metrics_train[i]]
                if validate:
                    metrics_val = list(val_epoch_metrics)
                    metrics_live[f"val_log-{metrics_val[0]}"] = np.log(val_epoch_metrics[metrics_val[0]])
                    if plot_live_all_metrics and len(metrics_val) > 1:
                        for i in range(1, len(metrics_val)):
                            metrics_live[f"val_{metrics_val[i]}"] = val_epoch_metrics[metrics_val[i]]
                live_loss.update(metrics_live)
                if e % (1 + self.config_train.epochs // 20) == 0 or e + 1 == self.config_train.epochs:
                    live_loss.send()

        # return metrics as df
        log.debug(f"Train Time: {(time.time() - start):8.3f}")
        log.debug(f"Total Batches: {self.metrics.total_updates}")
        metrics_df = self.metrics.get_stored_as_df()
        if validate:
            metrics_df_val = val_metrics.get_stored_as_df()
            for col in metrics_df_val.columns:
                metrics_df[f"{col}_val"] = metrics_df_val[col]
        return metrics_df

    def _train_minimal(self, df, progress_bar=False):
        """Execute minimal model training procedure for a configured number of epochs.

        Parameters
        ----------
            df: pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data

        Returns
        -------
            None
        """
        df, _, _, _ = df_utils.prep_or_copy_df(df)
        loader = self._init_train_loader(df)
        if progress_bar:
            training_loop = tqdm(
                range(self.config_train.epochs),
                total=self.config_train.epochs,
                leave=log.getEffectiveLevel() <= 20,
            )
        else:
            training_loop = range(self.config_train.epochs)
        for e in training_loop:
            if progress_bar:
                training_loop.set_description(f"Epoch[{(e+1)}/{self.config_train.epochs}]")
            _ = self._train_epoch(e, loader)

    def _eval_true_ar(self):
        assert self.max_lags > 0
        if self.highlight_forecast_step_n is None:
            if self.max_lags > 1:
                raise ValueError("Please define forecast_lag for sTPE computation")
            forecast_pos = 1
        else:
            forecast_pos = self.highlight_forecast_step_n
        weights = self.model.ar_weights.detach().numpy()
        weights = weights[forecast_pos - 1, :][::-1]
        sTPE = utils.symmetric_total_percentage_error(self.true_ar_weights, weights)
        log.info("AR parameters: ", self.true_ar_weights, "\n", "Model weights: ", weights)
        return sTPE

    def _evaluate(self, loader):
        """Evaluates model performance.

        Parameters
        ----------
            loader : torch DataLoader
                instantiated Validation Dataloader (with TimeDataset)

        Returns
        -------
            pd.DataFrame
                evaluation metrics
        """
        val_metrics = metrics.MetricsCollection([m.new() for m in self.metrics.batch_metrics])
        if self.highlight_forecast_step_n is not None:
            val_metrics.add_specific_target(target_pos=self.highlight_forecast_step_n - 1)
        ## Run
        val_metrics_dict = self._evaluate_epoch(loader, val_metrics)

        if self.true_ar_weights is not None:
            val_metrics_dict["sTPE"] = self._eval_true_ar()
        log.info(f"Validation metrics: {utils.print_epoch_metrics(val_metrics_dict)}")
        val_metrics_df = val_metrics.get_stored_as_df()
        return val_metrics_df

    def _make_future_dataframe(self, df, events_df, regressors_df, periods, n_historic_predictions):
        # Receives df with single ID column
        assert len(df["ID"].unique()) == 1
        if periods == 0 and n_historic_predictions is True:
            log.warning(
                "Not extending df into future as no periods specified." "You can call predict directly instead."
            )
        df = df.copy(deep=True)
        _ = df_utils.infer_frequency(df, n_lags=self.max_lags, freq=self.data_freq)
        last_date = pd.to_datetime(df["ds"].copy(deep=True).dropna()).sort_values().max()
        if events_df is not None:
            events_df = events_df.copy(deep=True).reset_index(drop=True)
        if regressors_df is not None:
            regressors_df = regressors_df.copy(deep=True).reset_index(drop=True)
        if periods is None:
            periods = 1 if self.max_lags == 0 else self.n_forecasts
        else:
            assert periods >= 0

        if isinstance(n_historic_predictions, bool):
            if n_historic_predictions:
                n_historic_predictions = len(df) - self.max_lags
            else:
                n_historic_predictions = 0
        elif not isinstance(n_historic_predictions, int):
            log.error("non-integer value for n_historic_predictions set to zero.")
            n_historic_predictions = 0

        if periods == 0 and n_historic_predictions == 0:
            raise ValueError("Set either history or future to contain more than zero values.")

        # check for external regressors known in future
        if self.config_regressors is not None and periods > 0:
            if regressors_df is None:
                raise ValueError("Future values of all user specified regressors not provided")
            else:
                for regressor in self.config_regressors.keys():
                    if regressor not in regressors_df.columns:
                        raise ValueError(f"Future values of user specified regressor {regressor} not provided")

        if len(df) < self.max_lags:
            raise ValueError(
                    "Insufficient input data for a prediction." 
                    "Please supply historic observations (number of rows) of at least max_lags (max of number of n_lags)."
                )
        elif len(df) < self.max_lags + n_historic_predictions:
            log.warning(
                f"Insufficient data for {n_historic_predictions} historic forecasts, reduced to {len(df) - self.max_lags}."
            )
            n_historic_predictions = len(df) - self.max_lags
        if (n_historic_predictions + self.max_lags) == 0:
            df = pd.DataFrame(columns=df.columns)
        else:
            df = df[-(self.max_lags + n_historic_predictions) :]
            nan_at_end = 0
            while len(df) > nan_at_end and df["y"].isnull().iloc[-(1 + nan_at_end)]:
                nan_at_end += 1
            if nan_at_end > 0:
                if self.max_lags > 0 and (nan_at_end + 1) >= self.max_lags:
                    raise ValueError(
                        f"{nan_at_end + 1} missing values were detected at the end of df before df was extended into the future. "
                        "Please make sure there are no NaN values at the end of df."
                    )
                df["y"].iloc[-(nan_at_end + 1) :].ffill(inplace=True)
                log.warning(
                    f"{nan_at_end + 1} missing values were forward-filled at the end of df before df was extended into the future. "
                    "Please make sure there are no NaN values at the end of df."
                )

        if len(df) > 0:
            if len(df.columns) == 1 and "ds" in df:
                assert self.max_lags == 0
                df = self._check_dataframe(df, check_y=False, exogenous=False)
            else:
                df = self._check_dataframe(df, check_y=self.max_lags > 0, exogenous=True)
        # future data
        # check for external events known in future
        if self.config_events is not None and periods > 0 and events_df is None:
            log.warning(
                "Future values not supplied for user specified events. "
                "All events being treated as not occurring in future"
            )

        if self.max_lags > 0:
            if periods > 0 and periods != self.n_forecasts:
                periods = self.n_forecasts
                log.warning(f"Number of forecast steps is defined by n_forecasts. " "Adjusted to {self.n_forecasts}.")

        if periods > 0:
            future_df = df_utils.make_future_df(
                df_columns=df.columns,
                last_date=last_date,
                periods=periods,
                freq=self.data_freq,
                config_events=self.config_events,
                events_df=events_df,
                config_regressor=self.config_regressors,
                regressors_df=regressors_df,
            )
            if len(df) > 0:
                df = pd.concat([df, future_df])
            else:
                df = future_df
        df = df.reset_index(drop=True)
        self.predict_steps = periods
        return df

    def _get_maybe_extend_periods(self, df):
        # Receives df with single ID column
        assert len(df["ID"].unique()) == 1
        periods_add = 0
        nan_at_end = 0
        while len(df) > nan_at_end and df["y"].isnull().iloc[-(1 + nan_at_end)]:
            nan_at_end += 1
        if self.max_lags > 0:
            if self.config_regressors is None:
                # if dataframe has already been extended into future,
                # don't extend beyond n_forecasts.
                periods_add = max(0, self.n_forecasts - nan_at_end)
            else:
                # can not extend as we lack future regressor values.
                periods_add = 0
        return periods_add

    def _maybe_extend_df(self, df):
        # Receives df with ID column
        periods_add = {}
        extended_df = pd.DataFrame()
        for df_name, df_i in df.groupby("ID"):
            _ = df_utils.infer_frequency(df_i, n_lags=self.max_lags, freq=self.data_freq)
            # to get all forecasteable values with df given, maybe extend into future:
            periods_add[df_name] = self._get_maybe_extend_periods(df_i)
            if periods_add[df_name] > 0:
                # This does not include future regressors or events.
                # periods should be 0 if those are configured.
                last_date = pd.to_datetime(df_i["ds"].copy(deep=True)).sort_values().max()
                future_df = df_utils.make_future_df(
                    df_columns=df_i.columns,
                    last_date=last_date,
                    periods=periods_add[df_name],
                    freq=self.data_freq,
                )
                future_df["ID"] = df_name
                df_i = pd.concat([df_i, future_df])
                df_i.reset_index(drop=True, inplace=True)
            extended_df = pd.concat((extended_df, df_i.copy(deep=True)), ignore_index=True)
        return extended_df, periods_add

    def _prepare_dataframe_to_predict(self, df):
        # Receives df with ID column
        df_prepared = pd.DataFrame()
        for df_name, df_i in df.groupby("ID"):
            df_i = df_i.copy(deep=True)
            _ = df_utils.infer_frequency(df_i, n_lags=self.max_lags, freq=self.data_freq)
            # check if received pre-processed df
            if "y_scaled" in df_i.columns or "t" in df_i.columns:
                raise ValueError(
                    "DataFrame has already been normalized. " "Please provide raw dataframe or future dataframe."
                )
            # Checks
            if len(df_i) == 0 or len(df_i) < self.max_lags:
                raise ValueError(
                    "Insufficient input data for a prediction." 
                    "Please supply historic observations (number of rows) of at least max_lags (max of number of n_lags)."
                )
            if len(df_i.columns) == 1 and "ds" in df_i:
                if self.max_lags != 0:
                    raise ValueError("only datestamps provided but y values needed for auto-regression.")
                df_i = self._check_dataframe(df_i, check_y=False, exogenous=False)
            else:
                df_i = self._check_dataframe(df_i, check_y=self.max_lags > 0, exogenous=False)
                # fill in missing nans except for nans at end
                df_i = self._handle_missing_data(df_i, freq=self.data_freq, predicting=True)
            df_prepared = pd.concat((df_prepared, df_i.copy(deep=True).reset_index(drop=True)), ignore_index=True)
        return df_prepared

    def _predict_raw(self, df, df_name, include_components=False):
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
        dataset = self._create_dataset(df, predict_mode=True)
        loader = DataLoader(dataset, batch_size=min(1024, len(df)), shuffle=False, drop_last=False)
        if self.n_forecasts > 1:
            dates = df["ds"].iloc[self.max_lags : -self.n_forecasts + 1]
        else:
            dates = df["ds"].iloc[self.max_lags :]
        predicted_vectors = list()
        component_vectors = None

        with torch.no_grad():
            self.model.eval()
            for inputs, _, _ in loader:
                inputs["predict_mode"] = True
                predicted = self.model.forward(inputs)
                predicted_vectors.append(predicted.detach().numpy())

                if include_components:
                    components = self.model.compute_components(inputs)
                    if component_vectors is None:
                        component_vectors = {name: [value.detach().numpy()] for name, value in components.items()}
                    else:
                        for name, value in components.items():
                            component_vectors[name].append(value.detach().numpy())

        predicted = np.concatenate(predicted_vectors)
        data_params = self.config_normalization.get_data_params(df_name)
        scale_y, shift_y = data_params["y"].scale, data_params["y"].shift
        predicted = predicted * scale_y + shift_y

        if include_components:
            components = {name: np.concatenate(value) for name, value in component_vectors.items()}
            for name, value in components.items():
                if "multiplicative" in name:
                    continue
                elif "event_" in name:
                    event_name = name.split("_")[1]
                    if self.config_events is not None and event_name in self.config_events:
                        if self.config_events[event_name].mode == "multiplicative":
                            continue
                    elif (
                        self.config_country_holidays is not None
                        and event_name in self.config_country_holidays.holiday_names
                    ):
                        if self.config_country_holidays.mode == "multiplicative":
                            continue
                elif "season" in name and self.config_season.mode == "multiplicative":
                    continue

                # scale additive components
                components[name] = value * scale_y
                if "trend" in name:
                    components[name] += shift_y
        else:
            components = None
        return dates, predicted, components

    def _convert_raw_predictions_to_raw_df(self, dates, predicted, components=None):
        """Turns forecast-origin-wise predictions into forecast-target-wise predictions.

        Parameters
        ----------
            dates : pd.Series
                timestamps referring to the start of the predictions.
            predicted : np.array
                Array containing the forecasts
            components : dict[np.array]
                Dictionary of components containing an array of each components' contribution to the forecast

        Returns
        -------
            pd. DataFrame
                columns ``ds``, ``y``, and [``step<i>``]

                Note
                ----
                where step<i> refers to the i-step-ahead prediction *made at* this row's datetime.
                e.g. the first forecast step0 is the prediction for this timestamp,
                the step1 is for the timestamp after, ...
                ... step3 is the prediction for 3 steps into the future,
                predicted using information up to (excluding) this datetime.
        """
        all_data = predicted
        df_raw = pd.DataFrame()
        df_raw.insert(0, "ds", dates.values)
        df_raw.insert(1, "ID", "__df__")
        for forecast_lag in range(self.n_forecasts):
            for quantile_idx in range(len(self.config_train.quantiles)):
                # 0 is the median quantile index
                if quantile_idx == 0:
                    step_name = f"step{forecast_lag}"
                else:
                    step_name = f"step{forecast_lag} {self.config_train.quantiles[quantile_idx] * 100}%"
                data = all_data[:, forecast_lag, quantile_idx]
                ser = pd.Series(data=data, name=step_name)
                df_raw = df_raw.merge(ser, left_index=True, right_index=True)
            if components is not None:
                for comp_name, comp_data in components.items():
                    comp_name_ = f"{comp_name}{forecast_lag}"
                    data = comp_data[:, forecast_lag, 0]  # for components the quantiles are ignored for now
                    ser = pd.Series(data=data, name=comp_name_)
                    df_raw = df_raw.merge(ser, left_index=True, right_index=True)
        return df_raw

    def _reshape_raw_predictions_to_forecst_df(self, df, predicted, components):
        """Turns forecast-origin-wise predictions into forecast-target-wise predictions.

        Parameters
        ----------
            df : pd.DataFrame
                input dataframe
            predicted : np.array
                Array containing the forecasts
            components : dict[np.array]
                Dictionary of components containing an array of each components' contribution to the forecast

        Returns
        -------
            pd.DataFrame
                columns ``ds``, ``y``, ``trend`` and [``yhat<i>``]

                Note
                ----
                where yhat<i> refers to the i-step-ahead prediction for this row's datetime.
                e.g. yhat3 is the prediction for this datetime, predicted 3 steps ago, "3 steps old".
        """
        # Receives df with single ID column
        assert len(df["ID"].unique()) == 1
        cols = ["ds", "y", "ID"]  # cols to keep from df
        df_forecast = pd.concat((df[cols],), axis=1)
        # create a line for each forecast_lag
        # 'yhat<i>' is the forecast for 'y' at 'ds' from i steps ago.
        for j in range(len(self.config_train.quantiles)):
            for forecast_lag in range(1, self.n_forecasts + 1):
                forecast = predicted[:, forecast_lag - 1, j]
                pad_before = self.max_lags + forecast_lag - 1
                pad_after = self.n_forecasts - forecast_lag
                yhat = np.concatenate(([None] * pad_before, forecast, [None] * pad_after))
                # 0 is the median quantile index
                if j == 0:
                    name = f"yhat{forecast_lag}"
                    df_forecast[f"residual{forecast_lag}"] = yhat - df_forecast["y"]
                else:
                    name = f"yhat{forecast_lag} {self.config_train.quantiles[j] * 100}%"
                df_forecast[name] = yhat

        if components is None:
            return df_forecast

        # else add components
        lagged_components = [
            "ar",
        ]
        if self.config_covar is not None:
            for name in self.config_covar.keys():
                lagged_components.append(f"lagged_regressor_{name}")
        for comp in lagged_components:
            if comp in components:
                for j in range(len(self.config_train.quantiles)):
                    for forecast_lag in range(1, self.n_forecasts + 1):
                        forecast = components[comp][:, forecast_lag - 1, j]  # 0 is the median quantile
                        pad_before = self.max_lags + forecast_lag - 1
                        pad_after = self.n_forecasts - forecast_lag
                        yhat = np.concatenate(([None] * pad_before, forecast, [None] * pad_after))
                        if j == 0:  # temporary condition to add only the median component
                            name = f"{comp}{forecast_lag}"
                            df_forecast[name] = yhat

        # only for non-lagged components
        for comp in components:
            if comp not in lagged_components:
                for j in range(len(self.config_train.quantiles)):
                    forecast_0 = components[comp][0, :, j]
                    forecast_rest = components[comp][1:, self.n_forecasts - 1, j]
                    yhat = np.concatenate(([None] * self.max_lags, forecast_0, forecast_rest))
                    if j == 0:  # temporary condition to add only the median component
                        # add yhat into dataframe, using df_forecast indexing
                        yhat_df = pd.Series(yhat, name=comp).set_axis(df_forecast.index)
                        df_forecast = pd.concat([df_forecast, yhat_df], axis=1, ignore_index=False)
        return df_forecast

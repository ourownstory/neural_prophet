import time
from collections import OrderedDict
from attrdict import AttrDict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import optim
import logging
from tqdm import tqdm
from torch_lr_finder import LRFinder

from neuralprophet import configure
from neuralprophet import time_net
from neuralprophet import time_dataset
from neuralprophet import df_utils
from neuralprophet import utils
from neuralprophet.plot_forecast import plot, plot_components
from neuralprophet.plot_model_parameters import plot_parameters
from neuralprophet import metrics
from neuralprophet.utils import set_logger_level

log = logging.getLogger("nprophet")


class NeuralProphet:
    """NeuralProphet forecaster.

    A simple yet powerful forecaster that models:
    Trend, seasonality, events, holidays, auto-regression, lagged covariates, and future-known regressors.
    Can be regualrized and configured to model nonlinear relationships.
    """

    def __init__(
        self,
        growth="linear",
        changepoints=None,
        n_changepoints=5,
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
        ar_sparsity=None,
        learning_rate=None,
        epochs=None,
        batch_size=None,
        loss_func="Huber",
        train_speed=None,
        normalize="auto",
        impute_missing=True,
        log_level=None,
    ):
        """
        Args:
            ## Trend Config
            growth (str): 'off', 'discontinuous', 'linear' to specify
                no trend, a discontinuous linear or a linear trend.
            changepoints (np.array): List of dates at which to include potential changepoints. If
                not specified, potential changepoints are selected automatically.
            n_changepoints (int): Number of potential changepoints to include.
                Changepoints are selected uniformly from the first `changepoint_range` proportion of the history.
                Not used if input `changepoints` is supplied. If `changepoints` is not supplied.
            changepoints_range (float): Proportion of history in which trend changepoints will
                be estimated. Defaults to 0.8 for the first 80%. Not used if `changepoints` is specified.
            trend_reg (float): Parameter modulating the flexibility of the automatic changepoint selection.
                Large values (~1-100) will limit the variability of changepoints.
                Small values (~0.001-1.0) will allow changepoints to change faster.
                default: 0 will fully fit a trend to each segment.
            trend_reg_threshold (bool, float): Allowance for trend to change without regularization.
                True: Automatically set to a value that leads to a smooth trend.
                False: All changes in changepoints are regularized

            ## Seasonality Config
            yearly_seasonality (bool, int): Fit yearly seasonality.
                Can be 'auto', True, False, or a number of Fourier/linear terms to generate.
            weekly_seasonality (bool, int): Fit monthly seasonality.
                Can be 'auto', True, False, or a number of Fourier/linear terms to generate.
            daily_seasonality (bool, int): Fit daily seasonality.
                Can be 'auto', True, False, or a number of Fourier/linear terms to generate.
            seasonality_mode (str): 'additive' (default) or 'multiplicative'.
            seasonality_reg (float): Parameter modulating the strength of the seasonality model.
                Smaller values (~0.1-1) allow the model to fit larger seasonal fluctuations,
                larger values (~1-100) dampen the seasonality.
                default: None, no regularization

            ## AR Config
            n_lags (int): Previous time series steps to include in auto-regression. Aka AR-order
            ar_sparsity (float): [0-1], how much sparsity to enduce in the AR-coefficients.
                Should be around (# nonzero components) / (AR order), eg. 3/100 = 0.03
                -1 will allow discontinuous trend (overfitting danger)

            ## Model Config
            n_forecasts (int): Number of steps ahead of prediction time step to forecast.
            num_hidden_layers (int): number of hidden layer to include in AR-Net. defaults to 0.
            d_hidden (int): dimension of hidden layers of the AR-Net. Ignored if num_hidden_layers == 0.

            ## Train Config
            learning_rate (float): Maximum learning rate setting for 1cycle policy scheduler.
                default: None: Automatically sets the learning_rate based on a learning rate range test.
                For manual values, try values ~0.001-10.
            epochs (int): Number of epochs (complete iterations over dataset) to train model.
                default: None: Automatically sets the number of epochs based on dataset size.
                    For best results also leave batch_size to None.
                For manual values, try ~5-500.
            batch_size (int): Number of samples per mini-batch.
                default: None: Automatically sets the batch_size based on dataset size.
                    For best results also leave epochs to None.
                For manual values, try ~1-512.
            loss_func (str, torch.nn.modules.loss._Loss): Type of loss to use ['Huber', 'MAE', 'MSE']
            train_speed (int, float) a quick setting to speed up or slow down model fitting [-3, -2, -1, 0, 1, 2, 3]
                potentially useful when under-, over-fitting, or simply in a hurry.
                applies epochs *= 2**-train_speed, batch_size *= 2**train_speed, learning_rate *= 2**train_speed,
                default None: equivalent to 0.

            ## Data config
            normalize (str): Type of normalization to apply to the time series.
                options: ['auto', 'soft', 'off', 'minmax, 'standardize']
                default: 'auto' uses 'minmax' if variable is binary, else 'soft'
                'soft' scales minimum to 0.1 and the 90th quantile to 0.9
            impute_missing (bool): whether to automatically impute missing dates/values
                imputation follows a linear method up to 10 missing values, more are filled with trend.

            ## General Config
            log_level (str): The log level of the logger objects used for printing procedure status
                updates for debugging/monitoring. Should be one of 'NOTSET', 'DEBUG', 'INFO', 'WARNING',
                'ERROR' or 'CRITICAL'
        """
        kwargs = locals()
        # Logging
        if log_level is not None:
            set_logger_level(log, log_level)

        # General
        self.name = "NeuralProphet"
        self.n_forecasts = n_forecasts

        # Data Preprocessing
        self.normalize = normalize
        self.impute_missing = impute_missing
        self.impute_limit_linear = 5
        self.impute_rolling = 20

        # Training
        self.config_train = configure.from_kwargs(configure.Train, kwargs)

        self.metrics = metrics.MetricsCollection(
            metrics=[
                metrics.LossMetric(self.config_train.loss_func),
                metrics.MAE(),
                # metrics.MSE(),
            ],
            value_metrics=[
                # metrics.ValueMetric("Loss"),
                metrics.ValueMetric("RegLoss"),
            ],
        )

        # AR
        self.n_lags = n_lags
        if n_lags == 0 and n_forecasts > 1:
            self.n_forecasts = 1
            log.warning(
                "Changing n_forecasts to 1. Without lags, the forecast can be "
                "computed for any future time, independent of lagged values"
            )

        # Model
        self.config_model = configure.from_kwargs(configure.Model, kwargs)

        # Trend
        self.config_trend = configure.from_kwargs(configure.Trend, kwargs)

        # Seasonality
        self.season_config = configure.AllSeason(
            mode=seasonality_mode,
            reg_lambda=seasonality_reg,
            yearly_arg=yearly_seasonality,
            weekly_arg=weekly_seasonality,
            daily_arg=daily_seasonality,
        )
        self.config_train.reg_lambda_season = self.season_config.reg_lambda

        # Events
        self.events_config = None
        self.country_holidays_config = None

        # Extra Regressors
        self.config_covar = None
        self.regressors_config = None

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
        # later set by user (optional)
        self.highlight_forecast_step_n = None
        self.true_ar_weights = None

    def _init_model(self):
        """Build Pytorch model with configured hyperparamters.

        Returns:
            TimeNet model
        """
        self.model = time_net.TimeNet(
            config_trend=self.config_trend,
            config_season=self.season_config,
            config_covar=self.config_covar,
            config_regressors=self.regressors_config,
            config_events=self.events_config,
            config_holidays=self.country_holidays_config,
            n_forecasts=self.n_forecasts,
            n_lags=self.n_lags,
            num_hidden_layers=self.config_model.num_hidden_layers,
            d_hidden=self.config_model.d_hidden,
        )
        log.debug(self.model)
        return self.model

    def _create_dataset(self, df, predict_mode):
        """Construct dataset from dataframe.

        (Configured Hyperparameters can be overridden by explicitly supplying them.
        Useful to predict a single model component.)

        Args:
            df (pd.DataFrame): containing original and normalized columns 'ds', 'y', 't', 'y_scaled'
            predict_mode (bool): False includes target values.
                True does not include targets but includes entire dataset as input
        Returns:
            TimeDataset
        """
        return time_dataset.TimeDataset(
            df,
            season_config=self.season_config,
            events_config=self.events_config,
            country_holidays_config=self.country_holidays_config,
            n_lags=self.n_lags,
            n_forecasts=self.n_forecasts,
            predict_mode=predict_mode,
            covar_config=self.config_covar,
            regressors_config=self.regressors_config,
        )

    def _handle_missing_data(self, df, predicting=False, allow_missing_dates="auto"):
        """Checks, auto-imputes and normalizes new data

        Args:
            df (pd.DataFrame): raw data with columns 'ds' and 'y'
            predicting (bool): allow NA values in 'y' of forecast series or 'y' to miss completely
            allow_missing_dates (bool): do not fill missing dates
                (only possible if no lags defined.)

        Returns:
            pre-processed df
        """
        if allow_missing_dates == "auto":
            allow_missing_dates = self.n_lags == 0
        elif allow_missing_dates:
            assert self.n_lags == 0
        if not allow_missing_dates:
            df, missing_dates = df_utils.add_missing_dates_nan(df, freq=self.data_freq)
            if missing_dates > 0:
                if self.impute_missing:
                    log.info("{} missing dates were added.".format(missing_dates))
                else:
                    raise ValueError(
                        "Missing dates found. " "Please preprocess data manually or set impute_missing to True."
                    )
        ## impute missing values
        data_columns = []
        if not (predicting and self.n_lags == 0):
            data_columns.append("y")
        if self.config_covar is not None:
            data_columns.extend(self.config_covar.keys())
        if self.regressors_config is not None:
            data_columns.extend(self.regressors_config.keys())
        if self.events_config is not None:
            data_columns.extend(self.events_config.keys())
        for column in data_columns:
            sum_na = sum(df[column].isnull())
            if sum_na > 0:
                if self.impute_missing is True:
                    # use 0 substitution for holidays and events missing values
                    if self.events_config is not None and column in self.events_config.keys():
                        df[column].fillna(0, inplace=True)
                        remaining_na = 0
                    else:
                        df, remaining_na = df_utils.fill_linear_then_rolling_avg(
                            df,
                            column=column,
                            allow_missing_dates=allow_missing_dates,
                            limit_linear=self.impute_limit_linear,
                            rolling=self.impute_rolling,
                            freq=self.data_freq,
                        )
                    log.info("{} NaN values in column {} were auto-imputed.".format(sum_na - remaining_na, column))
                    if remaining_na > 0:
                        raise ValueError(
                            "More than {} consecutive missing values encountered in column {}. "
                            "Please preprocess data manually.".format(
                                2 * self.impute_limit_linear + self.impute_rolling, column
                            )
                        )
                else:
                    raise ValueError(
                        "Missing values found. " "Please preprocess data manually or set impute_missing to True."
                    )
        return df

    def _validate_column_name(self, name, events=True, seasons=True, regressors=True, covariates=True):
        """Validates the name of a seasonality, event, or regressor.

        Args:
            name (str):
            events (bool):  check if name already used for event
            seasons (bool):  check if name already used for seasonality
            regressors (bool): check if name already used for regressor
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
        ]
        rn_l = [n + "_lower" for n in reserved_names]
        rn_u = [n + "_upper" for n in reserved_names]
        reserved_names.extend(rn_l)
        reserved_names.extend(rn_u)
        reserved_names.extend(["ds", "y", "cap", "floor", "y_scaled", "cap_scaled"])
        if name in reserved_names:
            raise ValueError("Name {name!r} is reserved.".format(name=name))
        if events and self.events_config is not None:
            if name in self.events_config.keys():
                raise ValueError("Name {name!r} already used for an event.".format(name=name))
        if events and self.country_holidays_config is not None:
            if name in self.country_holidays_config["holiday_names"]:
                raise ValueError(
                    "Name {name!r} is a holiday name in {country_holidays}.".format(
                        name=name, country_holidays=self.country_holidays_config["country"]
                    )
                )
        if seasons and self.season_config is not None:
            if name in self.season_config.periods:
                raise ValueError("Name {name!r} already used for a seasonality.".format(name=name))
        if covariates and self.config_covar is not None:
            if name in self.config_covar:
                raise ValueError("Name {name!r} already used for an added covariate.".format(name=name))
        if regressors and self.regressors_config is not None:
            if name in self.regressors_config.keys():
                raise ValueError("Name {name!r} already used for an added regressor.".format(name=name))

    def _lr_range_test(self, dataset, skip_start=10, skip_end=10, plot=False):
        lrtest_loader = DataLoader(dataset, batch_size=self.config_train.batch_size, shuffle=True)
        lrtest_optimizer = optim.Adam(self.model.parameters(), lr=1e-7, weight_decay=1e-2)
        with utils.HiddenPrints():
            lr_finder = LRFinder(self.model, lrtest_optimizer, self.config_train.loss_func)
            lr_finder.range_test(lrtest_loader, end_lr=100, num_iter=100)
            lrs = lr_finder.history["lr"]
            losses = lr_finder.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]
        if plot:
            with utils.HiddenPrints():
                ax, steepest_lr = lr_finder.plot()  # to inspect the loss-learning rate graph
        avg_idx = None
        try:
            steep_idx = (np.gradient(np.array(losses))).argmin()
            min_idx = (np.array(losses)).argmin()
            avg_idx = int((steep_idx + 2 * min_idx) / 3.0)
        except ValueError:
            log.error("Failed to compute the gradients, there might not be enough points.")
        if avg_idx is not None:
            max_lr = lrs[avg_idx]
            log.info("learning rate range test found optimal lr: {:.2E}".format(max_lr))
        else:
            max_lr = 0.1
            log.error("lr range test failed. defaulting to lr: {}".format(max_lr))
        with utils.HiddenPrints():
            lr_finder.reset()  # to reset the model and optimizer to their initial state
        return max_lr

    def _init_train_loader(self, df):
        """Executes data preparation steps and initiates training procedure.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with training data

        Returns:
            torch DataLoader
        """
        if not self.fitted:
            self.data_params = df_utils.init_data_params(
                df,
                normalize=self.normalize,
                covariates_config=self.config_covar,
                regressor_config=self.regressors_config,
                events_config=self.events_config,
            )
        df = df_utils.normalize(df, self.data_params)
        if not self.fitted:
            if self.config_trend.changepoints is not None:
                self.config_trend.changepoints = df_utils.normalize(
                    pd.DataFrame({"ds": pd.Series(self.config_trend.changepoints)}), self.data_params
                )["t"].values
            self.season_config = utils.set_auto_seasonalities(
                dates=df["ds"].copy(deep=True), season_config=self.season_config
            )
            if self.country_holidays_config is not None:
                self.country_holidays_config["holiday_names"] = utils.get_holidays_from_country(
                    self.country_holidays_config["country"], df["ds"]
                )
        self.config_train.set_auto_batch_epoch(n_data=len(df))
        self.config_train.apply_train_speed(batch=True, epoch=True)
        dataset = self._create_dataset(df, predict_mode=False)  # needs to be called after set_auto_seasonalities
        loader = DataLoader(dataset, batch_size=self.config_train.batch_size, shuffle=True)
        if not self.fitted:
            self.model = self._init_model()  # needs to be called after set_auto_seasonalities
        if self.config_train.learning_rate is None:
            self.config_train.learning_rate = self._lr_range_test(dataset)
        self.config_train.apply_train_speed(lr=True)
        self.optimizer = optim.AdamW(self.model.parameters())
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config_train.learning_rate,
            epochs=self.config_train.epochs,
            steps_per_epoch=len(loader),
            final_div_factor=1000,
        )
        return loader

    def _init_val_loader(self, df):
        """Executes data preparation steps and initiates evaluation procedure.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with validation data

        Returns:
            torch DataLoader
        """
        df = df_utils.normalize(df, self.data_params)
        dataset = self._create_dataset(df, predict_mode=False)
        loader = DataLoader(dataset, batch_size=min(1024, len(dataset)), shuffle=False, drop_last=False)
        return loader

    def _train_epoch(self, e, loader):
        """Make one complete iteration over all samples in dataloader and update model after each batch.

        Args:
            e (int): current epoch number
            loader (torch DataLoader): Training Dataloader
        """
        self.model.train()
        reg_lambda_ar = None
        if self.n_lags > 0:  # slowly increase regularization until lambda_delay epoch
            reg_lambda_ar = utils.get_regularization_lambda(
                self.config_train.ar_sparsity, self.config_train.lambda_delay, e
            )
        for inputs, targets in loader:
            # Run forward calculation
            predicted = self.model.forward(inputs)
            # Compute loss.
            loss = self.config_train.loss_func(predicted, targets)
            # Regularize.
            loss, reg_loss = self._add_batch_regualarizations(loss, reg_lambda_ar)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.metrics.update(
                predicted=predicted.detach(), target=targets.detach(), values={"Loss": loss, "RegLoss": reg_loss}
            )
        epoch_metrics = self.metrics.compute(save=True)
        return epoch_metrics

    def _add_batch_regualarizations(self, loss, reg_lambda_ar):
        """Add regulatization terms to loss, if applicable

        Args:
            loss (torch Tensor, scalar): current batch loss
            reg_lambda_ar (float): current AR regularization lambda

        Returns:
            loss, reg_loss
        """
        reg_loss = torch.zeros(1, dtype=torch.float, requires_grad=False)

        # Add regularization of AR weights - sparsify
        if self.model.n_lags > 0 and reg_lambda_ar is not None and reg_lambda_ar > 0:
            reg_ar = utils.reg_func_ar(self.model.ar_weights)
            reg_loss += reg_lambda_ar * reg_ar
            loss += reg_lambda_ar * reg_ar

        # Regularize trend to be smoother/sparse
        l_trend = self.config_trend.trend_reg
        if self.config_trend.n_changepoints > 0 and l_trend is not None and l_trend > 0:
            reg_trend = utils.reg_func_trend(
                weights=self.model.get_trend_deltas,
                threshold=self.config_train.trend_reg_threshold,
            )
            reg_loss += l_trend * reg_trend
            loss += l_trend * reg_trend

        # Regularize seasonality: sparsify fourier term coefficients
        l_season = self.config_train.reg_lambda_season
        if self.model.season_dims is not None and l_season is not None and l_season > 0:
            for name in self.model.season_params.keys():
                reg_season = utils.reg_func_season(self.model.season_params[name])
                reg_loss += l_season * reg_season
                loss += l_season * reg_season

        # Regularize events: sparsify events features coefficients
        if self.events_config is not None or self.country_holidays_config is not None:
            reg_events_loss = utils.reg_func_events(self.events_config, self.country_holidays_config, self.model)
            reg_loss += reg_events_loss
            loss += reg_events_loss

        # Regularize regressors: sparsify regressor features coefficients
        if self.regressors_config is not None:
            reg_regressor_loss = utils.reg_func_regressors(self.regressors_config, self.model)
            reg_loss += reg_regressor_loss
            loss += reg_regressor_loss

        return loss, reg_loss

    def _evaluate_epoch(self, loader, val_metrics):
        """Evaluates model performance.

        Args:
            loader (torch DataLoader):  instantiated Validation Dataloader (with TimeDataset)
            val_metrics (MetricsCollection): validation metrics to be computed.
        Returns:
            dict with evaluation metrics
        """
        with torch.no_grad():
            self.model.eval()
            for inputs, targets in loader:
                predicted = self.model.forward(inputs)
                val_metrics.update(predicted=predicted.detach(), target=targets.detach())
            val_metrics = val_metrics.compute(save=True)
        return val_metrics

    def _train(self, df, df_val=None, use_tqdm=True, plot_live_loss=False):
        """Execute model training procedure for a configured number of epochs.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with training data
            df_val (pd.DataFrame): containing column 'ds', 'y' with validation data
            use_tqdm (bool): display updating progress bar
            plot_live_loss (bool): plot live training loss,
                requires [live] install or livelossplot package installed.
        Returns:
            df with metrics
        """
        if plot_live_loss:
            try:
                from livelossplot import PlotLosses
            except:
                plot_live_loss = False
                log.warning(
                    "To plot live loss, please install neuralprophet[live]."
                    "Using pip: 'pip install neuralprophet[live]'"
                    "Or install the missing package manually: 'pip install livelossplot'",
                    exc_info=True,
                )

        loader = self._init_train_loader(df)
        val = df_val is not None
        ## Metrics
        if self.highlight_forecast_step_n is not None:
            self.metrics.add_specific_target(target_pos=self.highlight_forecast_step_n - 1)
        if not self.normalize == "off":
            self.metrics.set_shift_scale((self.data_params["y"].shift, self.data_params["y"].scale))
        if val:
            val_loader = self._init_val_loader(df_val)
            val_metrics = metrics.MetricsCollection([m.new() for m in self.metrics.batch_metrics])

        ## Run
        start = time.time()
        if use_tqdm:
            training_loop = tqdm(
                range(self.config_train.epochs), total=self.config_train.epochs, leave=log.getEffectiveLevel() <= 20
            )
        else:
            training_loop = range(self.config_train.epochs)
        if plot_live_loss:
            live_out = ["MatplotlibPlot"]
            if not use_tqdm:
                live_out.append("ExtremaPrinter")
            live_loss = PlotLosses(outputs=live_out)
        for e in training_loop:
            metrics_live = {}
            self.metrics.reset()
            if val:
                val_metrics.reset()
            epoch_metrics = self._train_epoch(e, loader)
            metrics_live["{}".format(list(epoch_metrics)[0])] = epoch_metrics[list(epoch_metrics)[0]]
            if val:
                val_epoch_metrics = self._evaluate_epoch(val_loader, val_metrics)
                metrics_live["val_{}".format(list(val_epoch_metrics)[0])] = val_epoch_metrics[
                    list(val_epoch_metrics)[0]
                ]
                print_val_epoch_metrics = {k + "_val": v for k, v in val_epoch_metrics.items()}
            else:
                val_epoch_metrics = None
                print_val_epoch_metrics = OrderedDict()
            if use_tqdm:
                training_loop.set_description(f"Epoch[{(e+1)}/{self.config_train.epochs}]")
                training_loop.set_postfix(ordered_dict=epoch_metrics, **print_val_epoch_metrics)
            else:
                metrics_string = utils.print_epoch_metrics(epoch_metrics, e=e, val_metrics=val_epoch_metrics)
                if e == 0:
                    log.info(metrics_string.splitlines()[0])
                    log.info(metrics_string.splitlines()[1])
                else:
                    log.info(metrics_string.splitlines()[1])
            if plot_live_loss:
                live_loss.update(metrics_live)
            if plot_live_loss and (e % (1 + self.config_train.epochs // 10) == 0 or e + 1 == self.config_train.epochs):
                live_loss.send()

        ## Metrics
        log.debug("Train Time: {:8.3f}".format(time.time() - start))
        log.debug("Total Batches: {}".format(self.metrics.total_updates))
        metrics_df = self.metrics.get_stored_as_df()
        if val:
            metrics_df_val = val_metrics.get_stored_as_df()
            for col in metrics_df_val.columns:
                metrics_df["{}_val".format(col)] = metrics_df_val[col]
        return metrics_df

    def _eval_true_ar(self):
        assert self.n_lags > 0
        if self.highlight_forecast_step_n is None:
            if self.n_lags > 1:
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

        Args:
            loader (torch DataLoader):  instantiated Validation Dataloader (with TimeDataset)
        Returns:
            df with evaluation metrics
        """
        val_metrics = metrics.MetricsCollection([m.new() for m in self.metrics.batch_metrics])
        if self.highlight_forecast_step_n is not None:
            val_metrics.add_specific_target(target_pos=self.highlight_forecast_step_n - 1)
        ## Run
        val_metrics_dict = self._evaluate_epoch(loader, val_metrics)

        if self.true_ar_weights is not None:
            val_metrics_dict["sTPE"] = self._eval_true_ar()
        log.info("Validation metrics: {}".format(utils.print_epoch_metrics(val_metrics_dict)))
        val_metrics_df = val_metrics.get_stored_as_df()
        return val_metrics_df

    @staticmethod
    def set_log_level(log_level, include_handlers=False):
        """
        Set the log level of all underlying logger objects

        Args:
            log_level (str): The log level of the logger objects used for printing procedure status
                updates for debugging/monitoring. Should be one of 'NOTSET', 'DEBUG', 'INFO', 'WARNING',
                'ERROR' or 'CRITICAL'
        """
        set_logger_level(log, log_level, include_handlers)

    def split_df(self, df, valid_p=0.2, inputs_overbleed=True):
        """Splits timeseries df into train and validation sets.

        Convenience function. See documentation on df_utils.split_df."""
        df = df_utils.check_dataframe(df, check_y=False)
        df = self._handle_missing_data(df, predicting=False)
        df_train, df_val = df_utils.split_df(
            df,
            n_lags=self.n_lags,
            n_forecasts=self.n_forecasts,
            valid_p=valid_p,
            inputs_overbleed=inputs_overbleed,
        )
        return df_train, df_val

    def fit(self, df, freq, epochs=None, validate_each_epoch=False, valid_p=0.2, use_tqdm=True, plot_live_loss=False):
        """Train, and potentially evaluate model.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with all data
            freq (str):Data step sizes. Frequency of data recording,
                Any valid frequency for pd.date_range, such as 'D' or 'M'
            epochs (int): number of epochs to train.
                default: if not specified, uses self.epochs
            validate_each_epoch (bool): whether to evaluate performance after each training epoch
            valid_p (float): fraction of data to hold out from training for model evaluation
            use_tqdm (bool): display updating progress bar
            plot_live_loss (bool): plot live training loss,
                requires [live] install or livelossplot package installed.
        Returns:
            metrics with training and potentially evaluation metrics
        """
        if freq != "D":
            # TODO: implement other frequency handling than daily.
            log.warning("Parts of code may break if using other than daily data.")
        self.data_freq = freq
        if epochs is not None:
            default_epochs = self.config_train.epochs
            self.config_train.epochs = epochs
        if self.fitted is True:
            log.warning("Model has already been fitted. Re-fitting will produce different results.")
        df = df_utils.check_dataframe(
            df, check_y=True, covariates=self.config_covar, regressors=self.regressors_config, events=self.events_config
        )
        df = self._handle_missing_data(df)
        if validate_each_epoch:
            df_train, df_val = df_utils.split_df(df, n_lags=self.n_lags, n_forecasts=self.n_forecasts, valid_p=valid_p)
            metrics_df = self._train(df_train, df_val, use_tqdm=use_tqdm, plot_live_loss=plot_live_loss)
        else:
            metrics_df = self._train(df, use_tqdm=use_tqdm, plot_live_loss=plot_live_loss)
        if epochs is not None:
            self.config_train.epochs = default_epochs
        self.fitted = True
        return metrics_df

    def test(self, df):
        """Evaluate model on holdout data.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with holdout data
        Returns:
            df with evaluation metrics
        """
        if self.fitted is False:
            log.warning("Model has not been fitted. Test results will be random.")
        df = df_utils.check_dataframe(df, check_y=True, covariates=self.config_covar, events=self.events_config)
        df = self._handle_missing_data(df)
        loader = self._init_val_loader(df)
        val_metrics_df = self._evaluate(loader)
        return val_metrics_df

    def make_future_dataframe(self, df, events_df=None, regressors_df=None, periods=None, n_historic_predictions=0):
        df = df.copy(deep=True)
        if events_df is not None:
            events_df = events_df.copy(deep=True).reset_index(drop=True)
        if regressors_df is not None:
            regressors_df = regressors_df.copy(deep=True).reset_index(drop=True)
        n_lags = 0 if self.n_lags is None else self.n_lags
        if periods is None:
            periods = 1 if n_lags == 0 else self.n_forecasts
        else:
            assert periods >= 0

        if isinstance(n_historic_predictions, bool):
            if n_historic_predictions:
                n_historic_predictions = len(df) - n_lags
            else:
                n_historic_predictions = 0
        elif not isinstance(n_historic_predictions, int):
            log.error("non-integer value for n_historic_predictions set to zero.")
            n_historic_predictions = 0

        if periods == 0 and n_historic_predictions == 0:
            raise ValueError("Set either history or future to contain more than zero values.")

        # check for external regressors known in future
        if self.regressors_config is not None and periods > 0:
            if regressors_df is None:
                raise ValueError("Future values of all user specified regressors not provided")
            else:
                for regressor in self.regressors_config.keys():
                    if regressor not in regressors_df.columns:
                        raise ValueError("Future values of user specified regressor {} not provided".format(regressor))

        last_date = pd.to_datetime(df["ds"].copy(deep=True)).sort_values().max()

        if len(df) < n_lags:
            raise ValueError("Insufficient data for a prediction")
        elif len(df) < n_lags + n_historic_predictions:
            log.warning(
                "Insufficient data for {} historic forecasts, reduced to {}.".format(
                    n_historic_predictions, len(df) - n_lags
                )
            )
            n_historic_predictions = len(df) - n_lags
        if (n_historic_predictions + n_lags) == 0:
            df = pd.DataFrame(columns=df.columns)
        else:
            df = df[-(n_lags + n_historic_predictions) :]

        if len(df) > 0:
            if len(df.columns) == 1 and "ds" in df:
                assert n_lags == 0
                df = df_utils.check_dataframe(df, check_y=False)
            else:
                df = df_utils.check_dataframe(
                    df, check_y=n_lags > 0, covariates=self.config_covar, events=self.events_config
                )
                df = self._handle_missing_data(df, predicting=True)
            df = df_utils.normalize(df, self.data_params)

        # future data
        # check for external events known in future
        if self.events_config is not None and periods > 0 and events_df is None:
            log.warning(
                "Future values not supplied for user specified events. "
                "All events being treated as not occurring in future"
            )

        if n_lags > 0:
            if periods > 0 and periods != self.n_forecasts:
                periods = self.n_forecasts
                log.warning(
                    "Number of forecast steps is defined by n_forecasts. " "Adjusted to {}.".format(self.n_forecasts)
                )

        if periods > 0:
            future_df = df_utils.make_future_df(
                df_columns=df.columns,
                last_date=last_date,
                periods=periods,
                freq=self.data_freq,
                events_config=self.events_config,
                events_df=events_df,
                regressor_config=self.regressors_config,
                regressors_df=regressors_df,
            )
            future_df = df_utils.normalize(future_df, self.data_params)
            if len(df) > 0:
                df = df.append(future_df)
            else:
                df = future_df
        df.reset_index(drop=True, inplace=True)
        return df

    def create_df_with_events(self, df, events_df):
        """
        Create a concatenated dataframe with the time series data along with the events data expanded.

        Args:
            df (pd.DataFrame): containing column 'ds' and 'y'
            events_df (pd.DataFrame): containing column 'ds' and 'event'
        Returns:
            pd.DataFrame with columns 'y', 'ds' and other user specified events

        """
        if self.events_config is None:
            raise Exception(
                "The events configs should be added to the NeuralProphet object (add_events fn)"
                "before creating the data with events features"
            )
        else:
            for name in events_df["event"].unique():
                assert name in self.events_config
            df = df_utils.check_dataframe(df)
            df_out = df_utils.convert_events_to_features(
                df.copy(deep=True),
                events_config=self.events_config,
                events_df=events_df.copy(deep=True),
            )

        return df_out.reset_index(drop=True)

    def predict(self, df):
        """Runs the model to make predictions.

        and compute stats (MSE, MAE)
        Args:
            df (pandas DataFrame): Dataframe with columns 'ds' datestamps, 'y' time series values and
                other external variables

        Returns:
            df_forecast (pandas DataFrame): columns 'ds', 'y', 'trend' and ['yhat<i>']
        """
        # TODO: Implement data sanity checks?
        if self.fitted is False:
            log.warning("Model has not been fitted. Predictions will be random.")
        dataset = self._create_dataset(df, predict_mode=True)
        loader = DataLoader(dataset, batch_size=min(1024, len(df)), shuffle=False, drop_last=False)

        predicted_vectors = list()
        component_vectors = None
        with torch.no_grad():
            self.model.eval()
            for inputs, _ in loader:
                predicted = self.model.forward(inputs)
                predicted_vectors.append(predicted.detach().numpy())
                components = self.model.compute_components(inputs)
                if component_vectors is None:
                    component_vectors = {name: [value.detach().numpy()] for name, value in components.items()}
                else:
                    for name, value in components.items():
                        component_vectors[name].append(value.detach().numpy())
        components = {name: np.concatenate(value) for name, value in component_vectors.items()}
        predicted = np.concatenate(predicted_vectors)

        scale_y, shift_y = self.data_params["y"].scale, self.data_params["y"].shift
        predicted = predicted * scale_y + shift_y
        for name, value in components.items():
            if "multiplicative" in name:
                continue
            elif "event_" in name:
                event_name = name.split("_")[1]
                if self.events_config is not None and event_name in self.events_config:
                    if self.events_config[event_name].mode == "multiplicative":
                        continue
                elif self.country_holidays_config is not None and event_name in self.country_holidays_config:
                    if self.country_holidays_config[event_name].mode == "multiplicative":
                        continue
            elif "season" in name and self.season_config.mode == "multiplicative":
                continue
            # scale additive components
            components[name] = value * scale_y
            if "trend" in name:
                components[name] += shift_y

        cols = ["ds", "y"]  # cols to keep from df
        df_forecast = pd.concat((df[cols],), axis=1)

        # create a line for each forecast_lag
        # 'yhat<i>' is the forecast for 'y' at 'ds' from i steps ago.
        for i in range(self.n_forecasts):
            forecast_lag = i + 1
            forecast = predicted[:, forecast_lag - 1]
            pad_before = self.n_lags + forecast_lag - 1
            pad_after = self.n_forecasts - forecast_lag
            yhat = np.concatenate(([None] * pad_before, forecast, [None] * pad_after))
            df_forecast["yhat{}".format(i + 1)] = yhat
            df_forecast["residual{}".format(i + 1)] = yhat - df_forecast["y"]

        lagged_components = [
            "ar",
        ]
        if self.config_covar is not None:
            for name in self.config_covar.keys():
                lagged_components.append("lagged_regressor_{}".format(name))
        for comp in lagged_components:
            if comp in components:
                for i in range(self.n_forecasts):
                    forecast_lag = i + 1
                    forecast = components[comp][:, forecast_lag - 1]
                    pad_before = self.n_lags + forecast_lag - 1
                    pad_after = self.n_forecasts - forecast_lag
                    yhat = np.concatenate(([None] * pad_before, forecast, [None] * pad_after))
                    df_forecast["{}{}".format(comp, i + 1)] = yhat

        # only for non-lagged components
        for comp in components:
            if comp not in lagged_components:
                forecast_0 = components[comp][0, :]
                forecast_rest = components[comp][1:, self.n_forecasts - 1]
                yhat = np.concatenate(([None] * self.n_lags, forecast_0, forecast_rest))
                df_forecast[comp] = yhat
        return df_forecast

    def predict_trend(self, df):
        """Predict only trend component of the model.

        Args:
            df (pd.DataFrame): containing column 'ds', prediction dates

        Returns:
            pd.Dataframe with trend on prediction dates.

        """
        df = df_utils.check_dataframe(df, check_y=False)
        df = df_utils.normalize(df, self.data_params)
        t = torch.from_numpy(np.expand_dims(df["t"].values, 1))
        trend = self.model.trend(t).squeeze().detach().numpy()
        trend = trend * self.data_params["y"].scale
        return pd.DataFrame({"ds": df["ds"], "trend": trend})

    def predict_seasonal_components(self, df):
        """Predict seasonality components

        Args:
            df (pd.DataFrame): containing column 'ds', prediction dates

        Returns:
            pd.Dataframe with seasonal components. with columns of name <seasonality component name>

        """
        df = df_utils.check_dataframe(df, check_y=False)
        df = df_utils.normalize(df, self.data_params)
        dataset = time_dataset.TimeDataset(
            df,
            season_config=self.season_config,
            # n_lags=0,
            # n_forecasts=1,
            predict_mode=True,
        )
        loader = DataLoader(dataset, batch_size=min(4096, len(df)), shuffle=False, drop_last=False)
        predicted = OrderedDict()
        for name in self.season_config.periods:
            predicted[name] = list()
        for inputs, _ in loader:
            for name in self.season_config.periods:
                features = inputs["seasonalities"][name]
                y_season = torch.squeeze(self.model.seasonality(features=features, name=name))
                predicted[name].append(y_season.data.numpy())

        for name in self.season_config.periods:
            predicted[name] = np.concatenate(predicted[name])
            if self.season_config.mode == "additive":
                predicted[name] = predicted[name] * self.data_params["y"].scale
        return pd.DataFrame({"ds": df["ds"], **predicted})

    def set_true_ar_for_eval(self, true_ar_weights):
        """configures model to evaluate closeness of AR weights to true weights.

        Args:
            true_ar_weights (np.array): True AR-parameters, if known.
        """
        self.true_ar_weights = true_ar_weights

    def highlight_nth_step_ahead_of_each_forecast(self, step_number=None):
        """Set which forecast step to focus on for metrics evaluation and plotting.

        Args:
            step_number (int): i-th step ahead forecast to use for statistics and plotting.
                default: None.
        """
        if step_number is not None:
            assert step_number <= self.n_forecasts
        self.highlight_forecast_step_n = step_number
        return self

    def add_lagged_regressor(self, name, regularization=None, normalize="auto", only_last_value=False):
        """Add a covariate time series as an additional lagged regressor to be used for fitting and predicting.

        The dataframe passed to `fit` and `predict` will have a column with the specified name to be used as
        a lagged regressor. When normalize=True, the covariate will be normalized unless it is binary.

        Args:
            name (string):  name of the regressor.
            regularization (float): optional  scale for regularization strength
            normalize (bool): optional, specify whether this regressor will be
                normalized prior to fitting.
                if 'auto', binary regressors will not be normalized.
            only_last_value (bool):
                False (default) use same number of lags as auto-regression
                True: only use last known value as input
        Returns:
            NeuralProphet object
        """
        if self.fitted:
            raise Exception("Covariates must be added prior to model fitting.")
        if self.n_lags == 0:
            raise Exception("Covariates must be set jointly with Auto-Regression.")
        self._validate_column_name(name)
        if self.config_covar is None:
            self.config_covar = OrderedDict({})
        self.config_covar[name] = configure.Covar(
            reg_lambda=regularization,
            normalize=normalize,
            as_scalar=only_last_value,
        )
        return self

    def add_future_regressor(self, name, regularization=None, normalize="auto", mode="additive"):
        """Add a regressor as lagged covariate with order 1 (scalar) or as known in advance (also scalar).

        The dataframe passed to `fit` and `predict` will have a column with the specified name to be used as
        a regressor. When normalize=True, the regressor will be normalized unless it is binary.

        Args:
            name (string):  name of the regressor.
            regularization (float): optional  scale for regularization strength
            normalize (bool): optional, specify whether this regressor will be
                normalized prior to fitting.
                if 'auto', binary regressors will not be normalized.
            mode (str): 'additive' (default) or 'multiplicative'.

        Returns:
            NeuralProphet object
        """
        if self.fitted:
            raise Exception("Regressors must be added prior to model fitting.")
        if regularization is not None:
            if regularization < 0:
                raise ValueError("regularization must be >= 0")
            if regularization == 0:
                regularization = None
        self._validate_column_name(name)

        if self.regressors_config is None:
            self.regressors_config = OrderedDict({})
        self.regressors_config[name] = AttrDict({"trend_reg": regularization, "normalize": normalize, "mode": mode})
        return self

    def add_events(self, events, lower_window=0, upper_window=0, regularization=None, mode="additive"):
        """
        Add user specified events and their corresponding lower, upper windows and the
        regularization parameters into the NeuralProphet object

        Args:
            events (str, list): name or list of names of user specified events
            lower_window (int): the lower window for the events in the list of events
            upper_window (int): the upper window for the events in the list of events
            regularization (float): optional  scale for regularization strength
            mode (str): 'additive' (default) or 'multiplicative'.
        Returns:
            NeuralProphet object
        """
        if self.fitted:
            raise Exception("Events must be added prior to model fitting.")

        if self.events_config is None:
            self.events_config = OrderedDict({})

        if regularization is not None:
            if regularization < 0:
                raise ValueError("regularization must be >= 0")
            if regularization == 0:
                regularization = None

        if not isinstance(events, list):
            events = [events]

        for event_name in events:
            self._validate_column_name(event_name)
            self.events_config[event_name] = AttrDict(
                {"lower_window": lower_window, "upper_window": upper_window, "trend_reg": regularization, "mode": mode}
            )
        return self

    def add_country_holidays(self, country_name, lower_window=0, upper_window=0, regularization=None, mode="additive"):
        """
        Add a country into the NeuralProphet object to include country specific holidays
        and create the corresponding configs such as lower, upper windows and the regularization
        parameters
        Args:
            country_name (string): name of the country
            lower_window (int): the lower window for all the country holidays
            upper_window (int): the upper window for all the country holidays
            regularization (float): optional  scale for regularization strength
            mode (str): 'additive' (default) or 'multiplicative'.
        Returns:
            NeuralProphet object
        """
        if self.fitted:
            raise Exception("Country must be specified prior to model fitting.")

        if regularization is not None:
            if regularization < 0:
                raise ValueError("regularization must be >= 0")
            if regularization == 0:
                regularization = None

        if self.country_holidays_config is None:
            self.country_holidays_config = OrderedDict({})

        self.country_holidays_config["country"] = country_name
        self.country_holidays_config["lower_window"] = lower_window
        self.country_holidays_config["upper_window"] = upper_window
        self.country_holidays_config["trend_reg"] = regularization
        self.country_holidays_config["holiday_names"] = utils.get_holidays_from_country(country_name)
        self.country_holidays_config["mode"] = mode
        return self

    def add_seasonality(self, name, period, fourier_order):
        """Add a seasonal component with specified period, number of Fourier components, and regularization.

        Increasing the number of Fourier components allows the seasonality to change more quickly
        (at risk of overfitting).
        Note: regularization and mode (additive/multiplicative) are set in the main init.

        Args:
            name: string name of the seasonality component.
            period: float number of days in one period.
            fourier_order: int number of Fourier components to use.
        Returns:
            The NeuralProphet object.
        """
        if self.fitted:
            raise Exception("Seasonality must be added prior to model fitting.")
        if name in ["daily", "weekly", "yearly"]:
            log.error("Please use inbuilt daily, weekly, or yearly seasonality or set another name.")
        # Do not Allow overwriting built-in seasonalities
        self._validate_column_name(name, seasons=True)
        if fourier_order <= 0:
            raise ValueError("Fourier Order must be > 0")
        self.season_config.append(name=name, period=period, resolution=fourier_order, arg="custom")
        return self

    def plot(self, fcst, ax=None, xlabel="ds", ylabel="y", figsize=(10, 6)):
        """Plot the NeuralProphet forecast, including history.

        Args:
            fcst (pd.DataFrame): output of self.predict.
            ax (matplotlib axes): Optional, matplotlib axes on which to plot.
            xlabel (string): label name on X-axis
            ylabel (string): label name on Y-axis
            figsize (tuple):   width, height in inches. default: (10, 6)

        Returns:
            A matplotlib figure.
        """
        if self.n_lags > 0:
            num_forecasts = sum(fcst["yhat1"].notna())
            if num_forecasts < self.n_forecasts:
                log.warning(
                    "Too few forecasts to plot a line per forecast step." "Plotting a line per forecast origin instead."
                )
                return self.plot_last_forecast(
                    fcst,
                    ax=ax,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    figsize=figsize,
                    include_previous_forecasts=num_forecasts - 1,
                    plot_history_data=True,
                )
        return plot(
            fcst=fcst,
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            highlight_forecast=self.highlight_forecast_step_n,
        )

    def plot_last_forecast(
        self,
        fcst,
        ax=None,
        xlabel="ds",
        ylabel="y",
        figsize=(10, 6),
        include_previous_forecasts=0,
        plot_history_data=None,
    ):
        """Plot the NeuralProphet forecast, including history.

        Args:
            fcst (pd.DataFrame): output of self.predict.
            ax (matplotlib axes): Optional, matplotlib axes on which to plot.
            xlabel (string): label name on X-axis
            ylabel (string): label name on Y-axis
            figsize (tuple):   width, height in inches. default: (10, 6)
            include_previous_forecasts (int): number of previous forecasts to include in plot
            plot_history_data
        Returns:
            A matplotlib figure.
        """
        if self.n_lags == 0:
            raise ValueError("Use the standard plot function for models without lags.")
        if plot_history_data is None:
            fcst = fcst[-(include_previous_forecasts + self.n_forecasts + self.n_lags) :]
        elif plot_history_data is False:
            fcst = fcst[-(include_previous_forecasts + self.n_forecasts) :]
        elif plot_history_data is True:
            fcst = fcst
        fcst = utils.fcst_df_to_last_forecast(fcst, n_last=1 + include_previous_forecasts)
        return plot(
            fcst=fcst,
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            highlight_forecast=self.highlight_forecast_step_n,
            line_per_origin=True,
        )

    def plot_components(self, fcst, figsize=None, residuals=False):
        """Plot the NeuralProphet forecast components.

        Args:
            fcst (pd.DataFrame): output of self.predict
            figsize (tuple):   width, height in inches.
                None (default):  automatic (10, 3 * npanel)
        Returns:
            A matplotlib figure.
        """
        return plot_components(
            m=self,
            fcst=fcst,
            figsize=figsize,
            forecast_in_focus=self.highlight_forecast_step_n,
            residuals=residuals,
        )

    def plot_parameters(self, weekly_start=0, yearly_start=0, figsize=None):
        """Plot the NeuralProphet forecast components.

        Args:
            weekly_start (int): specifying the start day of the weekly seasonality plot.
                0 (default) starts the week on Sunday. 1 shifts by 1 day to Monday, and so on.
            yearly_start (int): specifying the start day of the yearly seasonality plot.
                0 (default) starts the year on Jan 1. 1 shifts by 1 day to Jan 2, and so on.
            figsize (tuple):   width, height in inches.
                None (default):  automatic (10, 3 * npanel)
        Returns:
            A matplotlib figure.
        """
        return plot_parameters(
            m=self,
            forecast_in_focus=self.highlight_forecast_step_n,
            weekly_start=weekly_start,
            yearly_start=yearly_start,
            figsize=figsize,
        )

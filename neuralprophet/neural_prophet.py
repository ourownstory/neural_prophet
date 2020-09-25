import time
from collections import OrderedDict
from attrdict import AttrDict
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import optim

from neuralprophet import time_net
from neuralprophet import time_dataset
from neuralprophet import df_utils
from neuralprophet import utils
from neuralprophet import plotting_utils as plotting
from neuralprophet import metrics


class NeuralProphet:
    """NeuralProphet forecaster.

    Models Trend, Auto-Regression, Seasonality and Events.
    Can be configured to model nonlinear relationships.
    """
    def __init__(
            self,
            n_forecasts=1,
            n_lags=0,
            n_changepoints=5,
            learning_rate=1.0,
            loss_func='Huber',
            normalize_y=True,
            num_hidden_layers=0,
            d_hidden=None,
            ar_sparsity=None,
            trend_smoothness=0,
            trend_threshold=False,
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality='auto',
            seasonality_mode='additive',
            seasonality_reg=None,
            data_freq='D',
            impute_missing=True,
            verbose=False,
    ):
        """
        Args:
            n_forecasts (int): Number of steps ahead of prediction time step to forecast.
            n_lags (int): Previous time series steps to include in auto-regression. Aka AR-order
            n_changepoints (int): Number of potential changepoints to include.
                TODO: Not used if input `changepoints` is supplied. If `changepoints` is not supplied,
                then n_changepoints potential changepoints are selected uniformly from
                the first `changepoint_range` proportion of the history.
            learning_rate (float): Multiplier for learning rate. Try values ~0.001-10.
            loss_func (str): Type of loss to use ['Huber', 'MAE', 'MSE']
            normalize_y (bool): Whether to normalize the time series before modelling it.
            num_hidden_layers (int): number of hidden layer to include in AR-Net. defaults to 0.
            d_hidden (int): dimension of hidden layers of the AR-Net. Ignored if num_hidden_layers == 0.
            ar_sparsity (float): [0-1], how much sparsity to enduce in the AR-coefficients.
                Should be around (# nonzero components) / (AR order), eg. 3/100 = 0.03
            trend_smoothness (float): Parameter modulating the flexibility of the automatic changepoint selection.
                Large values (~1-100) will limit the variability of changepoints.
                Small values (~0.001-1.0) will allow changepoints to change faster.
                default: 0 will fully fit a trend to each segment.
                -1 will allow discontinuous trend (overfitting danger)
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
            data_freq (str):Data step sizes. Frequency of data recording,
                Any valid frequency for pd.date_range, such as 'D' or 'M'
            impute_missing (bool): whether to automatically impute missing dates/values
                imputation follows a linear method up to 10 missing values, more are filled with trend.
            verbose (bool): Whether to print procedure status updates for debugging/monitoring

        TODO:
            changepoints (np.array): List of dates at which to include potential changepoints. If
                not specified, potential changepoints are selected automatically.
            changepoint_range (float): Proportion of history in which trend changepoints will
                be estimated. Defaults to 0.9 for the first 90%. Not used if
                `changepoints` is specified.
        """
        ## General
        self.name = "NeuralProphet"
        self.verbose = verbose
        self.n_forecasts = n_forecasts

        ## Data Preprocessing
        self.normalize_y = normalize_y
        self.data_freq = data_freq
        if self.data_freq != 'D':
            # TODO: implement other frequency handling than daily.
            print("NOTICE: Parts of code may break if using other than daily data.")
        self.impute_missing = impute_missing
        self.impute_limit_linear = 5
        self.impute_rolling = 20

        ## Training
        self.train_config = AttrDict({  # TODO allow to be passed in init
            "lr": learning_rate,
            "lr_decay": 0.98,
            "epochs": 40,
            "batch": 128,
            "est_sparsity": ar_sparsity,  # 0 = fully sparse, 1 = not sparse
            "lambda_delay": 10,  # delays start of regularization by lambda_delay epochs
            "reg_lambda_trend": None,
            "trend_reg_threshold": None,
            "reg_lambda_season": None
        })
        self.loss_func_name = loss_func
        if loss_func.lower() in ['huber','smoothl1', 'smoothl1loss']:
            self.loss_fn = torch.nn.SmoothL1Loss()
        elif loss_func.lower() in ['mae', 'l1', 'l1loss']:
            self.loss_fn = torch.nn.L1Loss()
        elif loss_func.lower() in ['mse', 'mseloss', 'l2', 'l2loss']:
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise NotImplementedError("Loss function {} not found".format(loss_func))
        self.metrics = metrics.MetricsCollection(
            metrics=[
                metrics.LossMetric(self.loss_fn),
                metrics.MAE(),
                # metrics.MSE(),
            ],
            value_metrics=[
                # metrics.ValueMetric("Loss"),
                metrics.ValueMetric("RegLoss"),
            ]
        )

        ## AR
        self.n_lags = n_lags
        if n_lags == 0 and n_forecasts > 1:
            self.n_forecasts = 1
            print("NOTICE: changing n_forecasts to 1. Without lags, "
                  "the forecast can be computed for any future time, independent of present values")
        self.model_config = AttrDict({
            "num_hidden_layers": num_hidden_layers,
            "d_hidden": d_hidden,
        })

        ## Trend
        self.n_changepoints = n_changepoints
        self.trend_smoothness = trend_smoothness
        # self.growth = "linear" # OG Prophet Trend related, only linear currently implemented
        # if self.growth != 'linear':
        #     raise NotImplementedError
        if self.n_changepoints > 0 and self.trend_smoothness > 0:
            print("NOTICE: A numeric value greater than 0 for continuous_trend is interpreted as"
                  "the trend changepoint regularization strength. Please note that this feature is experimental.")
            self.train_config.reg_lambda_trend = 0.01*self.trend_smoothness
            if trend_threshold is not None and trend_threshold is not False:
                if trend_threshold == 'auto' or trend_threshold is True:
                    self.train_config.trend_reg_threshold = 3.0 / (3 + (1 + self.trend_smoothness) * np.sqrt(self.n_changepoints))
                else:
                    self.train_config.trend_reg_threshold = trend_threshold

        ## Seasonality
        self.season_config = AttrDict({})
        self.season_config.type = 'fourier'  # Currently no other seasonality_type
        self.season_config.mode = seasonality_mode
        self.season_config.periods = OrderedDict({ # defaults
            "yearly": AttrDict({'resolution': 6, 'period': 365.25, 'arg': yearly_seasonality}),
            "weekly": AttrDict({'resolution': 4, 'period': 7, 'arg': weekly_seasonality,}),
            "daily": AttrDict({'resolution': 6, 'period': 1, 'arg': daily_seasonality,}),
        })
        if seasonality_reg is not None:
            print("NOTICE: A Regularization strength for the seasonal Fourier Terms was set."
                  "Please note that this feature is experimental.")
            self.train_config.reg_lambda_season = 0.1 * seasonality_reg

        ## Events
        self.events_config = None
        self.country_holidays_config = None

        ## Extra Regressors
        self.covar_config = None
        self.regressors_config = None

        ## Set during _train()
        self.fitted = False
        self.history = None
        self.data_params = None
        self.optimizer = None
        self.scheduler = None
        self.model = None

        ## set during prediction
        self.future_periods = None
        ## later set by user (optional)
        self.forecast_in_focus = None
        self.true_ar_weights = None

    def _init_model(self):
        """Build Pytorch model with configured hyperparamters.

        Returns:
            TimeNet model
        """
        self.model = time_net.TimeNet(
            n_forecasts=self.n_forecasts,
            n_lags=self.n_lags,
            n_changepoints=self.n_changepoints,
            trend_smoothness=self.trend_smoothness,
            num_hidden_layers=self.model_config.num_hidden_layers,
            d_hidden=self.model_config.d_hidden,
            season_dims=utils.season_config_to_model_dims(self.season_config),
            season_mode=self.season_config.mode if self.season_config is not None else None,
            covar_config=self.covar_config,
            regressors_dims=utils.regressors_config_to_model_dims(self.regressors_config),
            events_dims=utils.events_config_to_model_dims(self.events_config, self.country_holidays_config),
        )
        if self.verbose:
            print(self.model)
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
            verbose=self.verbose,
            covar_config=self.covar_config,
            regressors_config=self.regressors_config,
        )

    def _auto_learning_rate(self, multiplier=1.0):
        """Computes a reasonable guess for a learning rate based on estimated model complexity.

        Args:
            multiplier (float): multiplier for learning rate guesstimate

        Returns:
            learning rate guesstimate
        """
        model_complexity = 10 * np.sqrt(self.n_lags * self.n_forecasts)
        model_complexity += np.log(1 + self.n_changepoints)
        if self.season_config is not None:
            model_complexity += np.log(1 + sum([p.resolution for name, p in self.season_config.periods.items()]))
        model_complexity = max(1.0, model_complexity)
        if self.verbose: print("model_complexity", model_complexity)
        return multiplier / model_complexity

    def _handle_missing_data(self, df, predicting=False, allow_missing_dates='auto'):
        """Checks, auto-imputes and normalizes new data

        Args:
            df (pd.DataFrame): raw data with columns 'ds' and 'y'
            predicting (bool): allow NA values in 'y' of forecast series or 'y' to miss completely
            allow_missing_dates (bool): do not fill missing dates
                (only possible if no lags defined.)

        Returns:
            pre-processed df
        """
        if allow_missing_dates == 'auto':
            allow_missing_dates = self.n_lags == 0
        elif allow_missing_dates:
            assert self.n_lags == 0
        if not allow_missing_dates:
            df, missing_dates = df_utils.add_missing_dates_nan(df, freq=self.data_freq)
            if missing_dates > 0:
                if self.impute_missing:
                    if self.verbose:
                        print("NOTICE: {} missing dates were added.".format(missing_dates))
                else:
                    raise ValueError("Missing dates found. "
                                     "Please preprocess data manually or set impute_missing to True.")
        ## impute missing values
        data_columns = []
        if not (predicting and self.n_lags == 0):
            data_columns.append('y')
        if self.covar_config is not None:
            data_columns.extend(self.covar_config.keys())
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
                    else:
                        df, remaining_na = df_utils.fill_linear_then_rolling_avg(
                            df, column=column, allow_missing_dates=allow_missing_dates,
                            limit_linear=self.impute_limit_linear, rolling=self.impute_rolling, freq=self.data_freq)
                    if self.verbose:
                        print("NOTICE: {} NaN values in column {} were auto-imputed."
                              .format(sum_na - remaining_na, column))
                    if remaining_na > 0:
                        raise ValueError("More than {} consecutive missing values encountered in column {}. "
                                         "Please preprocess data manually."
                                         .format(2*self.impute_limit_linear + self.impute_rolling, column))
                else:
                    raise ValueError("Missing values found. "
                                     "Please preprocess data manually or set impute_missing to True.")
        return df

    def _validate_column_name(self, name, check_events=True, check_seasonalities=True, check_regressors=True):
        """Validates the name of a seasonality, event, or regressor.

        Args:
            name (str):
            check_events (bool):  check if name already used for event
            check_seasonalities (bool):  check if name already used for seasonality
            check_regressors (bool): check if name already used for regressor
        """
        reserved_names = [
            'trend', 'additive_terms', 'daily', 'weekly', 'yearly',
            'events', 'holidays', 'zeros', 'extra_regressors_additive', 'yhat',
            'extra_regressors_multiplicative', 'multiplicative_terms',
        ]
        rn_l = [n + '_lower' for n in reserved_names]
        rn_u = [n + '_upper' for n in reserved_names]
        reserved_names.extend(rn_l)
        reserved_names.extend(rn_u)
        reserved_names.extend(['ds', 'y', 'cap', 'floor', 'y_scaled', 'cap_scaled'])
        if name in reserved_names:
            raise ValueError('Name {name!r} is reserved.'.format(name=name))
        if check_events and self.events_config is not None:
            if name in self.events_config.keys():
                raise ValueError('Name {name!r} already used for an event.'
                                 .format(name=name))
        if check_events and self.country_holidays_config is not None:
            if name in self.country_holidays_config["holiday_names"]:
                raise ValueError('Name {name!r} is a holiday name in {country_holidays}.'
                                 .format(name=name, country_holidays=self.country_holidays_config["country"]))
        if check_seasonalities and self.season_config is not None:
            if name in self.season_config.periods:
                raise ValueError('Name {name!r} already used for a seasonality.'
                                 .format(name=name))
        if check_regressors and self.covar_config is not None:
            if name in self.covar_config:
                raise ValueError('Name {name!r} already used for an added regressor.'
                                 .format(name=name))
        if check_regressors and self.regressors_config is not None:
            if name in self.regressors_config.keys():
                raise ValueError('Name {name!r} already used for an added regressor.'
                                 .format(name=name))

    def _init_train_loader(self, df):
        """Executes data preparation steps and initiates training procedure.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with training data

        Returns:
            torch DataLoader
        """
        ## compute data parameters
        self.data_params = df_utils.init_data_params(
            df, normalize_y=self.normalize_y, covariates_config=self.covar_config, regressor_config=self.regressors_config,
            events_config=self.events_config, verbose=self.verbose)
        df = df_utils.normalize(df, self.data_params)
        self.history = df.copy(deep=True)
        self.season_config = utils.set_auto_seasonalities(
            dates=self.history['ds'], season_config=self.season_config, verbose=self.verbose)
        if self.country_holidays_config is not None:
            self.country_holidays_config["holiday_names"] = utils.get_holidays_from_country(self.country_holidays_config["country"], df['ds'])
        dataset = self._create_dataset(df, predict_mode=False)  # needs to be called after set_auto_seasonalities
        loader = DataLoader(dataset, batch_size=self.train_config["batch"], shuffle=True)
        self.model = self._init_model()  # needs to be called after set_auto_seasonalities
        self.train_config.lr = self._auto_learning_rate(multiplier=self.train_config.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_config.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.train_config.lr_decay)
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
        if self.n_lags > 0: # slowly increase regularization until lambda_delay epoch
            reg_lambda_ar = utils.get_regularization_lambda(
                self.train_config.est_sparsity, self.train_config.lambda_delay, e)
        for inputs, targets in loader:
            # Run forward calculation
            predicted = self.model.forward(inputs)
            # Compute loss.
            loss = self.loss_fn(predicted, targets)
            # Regularize.
            loss, reg_loss = self._add_batch_regualarizations(loss, reg_lambda_ar)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.metrics.update(predicted=predicted.detach(), target=targets.detach(),
                                values={"Loss": loss, "RegLoss": reg_loss})
        self.scheduler.step()
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
        l_trend = self.train_config.reg_lambda_trend
        if self.model.n_changepoints > 0 and l_trend is not None and l_trend > 0:
            reg_trend = utils.reg_func_trend(
                weights=self.model.get_trend_deltas,
                threshold=self.train_config.trend_reg_threshold,
            )
            reg_loss += l_trend * reg_trend
            loss += l_trend * reg_trend

        # Regularize seasonality: sparsify fourier term coefficients
        l_season = self.train_config.reg_lambda_season
        if self.model.season_dims is not None and l_season is not None and l_season > 0:
            for name in self.model.season_params.keys():
                reg_season = utils.reg_func_season(self.model.season_params[name])
                reg_loss += l_season * reg_season
                loss += l_season * reg_season

        # Regularize holidays: sparsify holiday features coefficients
        if self.events_config is not None or self.country_holidays_config is not None:
            pass

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

    def _train(self, df, df_val=None):
        """Execute model training procedure for a configured number of epochs.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with training data
            df_val (pd.DataFrame): containing column 'ds', 'y' with validation data
        Returns:
            df with metrics
        """
        loader = self._init_train_loader(df)
        val = df_val is not None
        ## Metrics
        if self.forecast_in_focus is not None:
            self.metrics.add_specific_target(target_pos=self.forecast_in_focus - 1)
        if self.normalize_y:
            self.metrics.set_shift_scale((self.data_params['y'].shift, self.data_params['y'].scale))
        if val:
            val_loader = self._init_val_loader(df_val)
            val_metrics = metrics.MetricsCollection([m.new() for m in self.metrics.batch_metrics])

        ## Run
        start = time.time()
        for e in range(self.train_config.epochs):
            self.metrics.reset()
            if val: val_metrics.reset()
            epoch_metrics = self._train_epoch(e, loader)
            if val: val_epoch_metrics = self._evaluate_epoch(val_loader, val_metrics)
            else: val_epoch_metrics = None
            if self.verbose:
                utils.print_epoch_metrics(epoch_metrics, e=e, val_metrics=val_epoch_metrics)
        ## Metrics
        if self.verbose:
            print("Train Time: {:8.3f}".format(time.time() - start))
            print("Total Batches: ", self.metrics.total_updates)
        metrics_df = self.metrics.get_stored_as_df()
        if val:
            metrics_df_val = val_metrics.get_stored_as_df()
            for col in metrics_df_val.columns:
                metrics_df["{}_val".format(col)] = metrics_df_val[col]
        return metrics_df

    def _eval_true_ar(self, verbose=False):
        assert self.n_lags > 0
        if self.forecast_in_focus is None:
            if self.n_lags > 1:
                raise ValueError("Please define forecast_lag for sTPE computation")
            forecast_pos = 1
        else:
            forecast_pos = self.forecast_in_focus
        weights = self.model.ar_weights.detach().numpy()
        weights = weights[forecast_pos - 1, :][::-1]
        sTPE = utils.symmetric_total_percentage_error(self.true_ar_weights, weights)
        if verbose:
            print("AR parameters: ", self.true_ar_weights, "\n", "Model weights: ", weights)
        return sTPE

    def _evaluate(self, loader, verbose=None):
        """Evaluates model performance.

        Args:
            loader (torch DataLoader):  instantiated Validation Dataloader (with TimeDataset)
        Returns:
            df with evaluation metrics
        """
        if self.fitted is False: raise Exception('Model object needs to be fit first.')
        if verbose is None: verbose = self.verbose
        val_metrics = metrics.MetricsCollection([m.new() for m in self.metrics.batch_metrics])
        if self.forecast_in_focus is not None:
            val_metrics.add_specific_target(target_pos=self.forecast_in_focus - 1)
        ## Run
        val_metrics_dict = self._evaluate_epoch(loader, val_metrics)

        if self.true_ar_weights is not None:
            val_metrics_dict["sTPE"] = self._eval_true_ar(verbose=verbose)
        if verbose:
            print("Validation metrics:")
            utils.print_epoch_metrics(val_metrics_dict)
        val_metrics_df = val_metrics.get_stored_as_df()
        return val_metrics_df

    def split_df(self, df, valid_p=0.2, inputs_overbleed=True, verbose=None):
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
            verbose=self.verbose if verbose is None else verbose,
        )
        return df_train, df_val

    def fit(self, df, validate_each_epoch=False, valid_p=0.2):
        """Train, and potentially evaluate model.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with all data
            validate_each_epoch (bool): whether to evaluate performance after each training epoch
            valid_p (float): fraction of data to hold out from training for model evaluation
        Returns:
            metrics with training and potentially evaluation metrics
        """
        if self.fitted is True:
            raise Exception('Model object can only be fit once. Instantiate a new object.')
        df = df_utils.check_dataframe(df, check_y=True, covariates=self.covar_config, regressors=self.regressors_config,
                                      events=self.events_config)
        df = self._handle_missing_data(df)
        if validate_each_epoch:
            df_train, df_val = df_utils.split_df(df, n_lags=self.n_lags, n_forecasts=self.n_forecasts, valid_p=valid_p)
            metrics_df = self._train(df_train, df_val)
        else:
            metrics_df = self._train(df)
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
            raise Exception('Model needs to be fit first.')
        df = df_utils.check_dataframe(df, check_y=True, covariates=self.covar_config, events=self.events_config)
        df = self._handle_missing_data(df)
        loader = self._init_val_loader(df)
        val_metrics_df = self._evaluate(loader)
        return val_metrics_df

    def compose_prediction_df(self, df, events_df=None, regressors_df=None, future_periods=None, n_historic_predictions=0):
        assert n_historic_predictions >= 0
        if future_periods is not None:
            assert future_periods >= 0
            if future_periods == 0 and n_historic_predictions == 0:
                raise ValueError("Set either history or future to contain more than zero values.")

        # check for external regressors known in future
        if self.regressors_config is not None and future_periods is not None:
            if regressors_df is None:
                raise ValueError("Future values of all user specified regressors not provided")
            else:
                for regressor in self.regressors_config.keys():
                    if regressor not in regressors_df.columns:
                        raise ValueError("Future values of user specified regressor {} not provided".format(regressor))


        n_lags = 0 if self.n_lags is None else self.n_lags

        if len(df) < n_lags:
            raise ValueError("Insufficient data for a prediction")
        elif len(df) < n_lags + n_historic_predictions:
            print("Warning: insufficient data for {} historic forecasts, reduced to {}.".format(
                n_historic_predictions, len(df) - n_lags))
            n_historic_predictions = len(df) - n_lags
        df = df[-(n_lags + n_historic_predictions):]

        if len(df) > 0:
            if len(df.columns) == 1 and 'ds' in df:
                assert n_lags == 0
                df = df_utils.check_dataframe(df, check_y=False)
            else:
                df = df_utils.check_dataframe(df, check_y=n_lags > 0, covariates=self.covar_config, events=self.events_config)
                df = self._handle_missing_data(df, predicting=True)
            df = df_utils.normalize(df, self.data_params)

        # future data
        # check for external events known in future
        if self.events_config is not None and future_periods is not None and events_df is None:
            print("NOTICE: Future values not supplied for user specified events. "
                  "All events being treated as not occurring in future")

        if future_periods is None:
            if n_lags > 0:
                future_periods = self.n_forecasts
            else:
                future_periods = 1

        if n_lags > 0:
            if future_periods > 0 and future_periods != self.n_forecasts:
                future_periods = self.n_forecasts
                print("NOTICE: Number of forecast steps is defined by n_forecasts. "
                      "Adjusted to {}.".format(self.n_forecasts))

        if future_periods > 0:
            future_df = df_utils.make_future_df(
                df, periods=future_periods, freq=self.data_freq,
                events_config=self.events_config, events_df=events_df,
                regressor_config=self.regressors_config, regressors_df=regressors_df)
            future_df = df_utils.normalize(future_df, self.data_params)
            if len(df) > 0:
                df = df.append(future_df)
            else:
                df = future_df
        df.reset_index(drop=True, inplace=True)
        return df

    def predict(self, df):
        """Runs the model to make predictions.

        and compute stats (MSE, MAE)
        Args:
            df (pandas DataFrame): Dataframe with columns 'ds' datestamps, 'y' time series values and
                other external variables

        Returns:
            df_forecast (pandas DataFrame): columns 'ds', 'y', 'trend' and ['yhat<i>']
        """
        #TODO: Implement data sanity checks?
        if self.fitted is False:
            raise Exception('Model has not been fit.')
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

        scale_y, shift_y = self.data_params['y'].scale, self.data_params['y'].shift
        predicted = predicted * scale_y + shift_y
        multiplicative_components = [
            name for name in components.keys() if ('season' in name and self.season_config.mode == 'multiplicative')
        ]
        for name, value in components.items():
            if name not in multiplicative_components:
                components[name] = value * scale_y

        cols = ['ds', 'y']  # cols to keep from df
        df_forecast = pd.concat((df[cols],), axis=1)

        # create a line for each forecast_lag
        # 'yhat<i>' is the forecast for 'y' at 'ds' from i steps ago.
        for i in range(self.n_forecasts):
            forecast_lag = i + 1
            forecast = predicted[:, forecast_lag - 1]
            pad_before = self.n_lags + forecast_lag - 1
            pad_after = self.n_forecasts - forecast_lag
            yhat = np.concatenate(([None] * pad_before, forecast, [None] * pad_after))
            df_forecast['yhat{}'.format(i + 1)] = yhat
            df_forecast['residual{}'.format(i + 1)] = yhat - df_forecast['y']

        lagged_components = ['ar', ]
        if self.covar_config is not None:
            for name in self.covar_config.keys():
                lagged_components.append('lagged_regressor_{}'.format(name))
        for comp in lagged_components:
            if comp in components:
                for i in range(self.n_forecasts):
                    forecast_lag = i + 1
                    forecast = components[comp][:, forecast_lag - 1]
                    pad_before = self.n_lags + forecast_lag - 1
                    pad_after = self.n_forecasts - forecast_lag
                    yhat = np.concatenate(([None] * pad_before, forecast, [None] * pad_after))
                    df_forecast['{}{}'.format(comp, i + 1)] = yhat

        # # OR create a line for each foreacast
        # # 'yhat<i>' is the forecast given at i steps before the end of data.
        # n_history = only_last_n - 1
        # for i in range(n_history + 1):
        #     forecast_age = i
        #     forecast = predicted[-1 - forecast_age, :]
        #     pad_before = self.n_lags + n_history - forecast_age
        #     pad_after = forecast_age
        #     yhat = np.concatenate(([None] * pad_before, forecast, [None] * pad_after))
        #     df_forecast['yhat{}'.format(i + 1)] = yhat
        #     df_forecast['residual{}'.format(i + 1)] = yhat - df_forecast['y']
        #
        # lagged_components = ['ar', ]
        # if self.covar_config is not None:
        #     for name in self.covar_config.keys():
        #         lagged_components.append('covar_{}'.format(name))
        # for comp in lagged_components:
        #     if comp in components:
        #         for i in range(n_history + 1):
        #             forecast_age = i
        #             forecast = components[comp][-1 - forecast_age, :]
        #             pad_before = self.n_lags + n_history - forecast_age
        #             pad_after = forecast_age
        #             yhat = np.concatenate(([None] * pad_before, forecast, [None] * pad_after))
        #             df_forecast['{}{}'.format(comp, i + 1)] = yhat

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
        t = torch.from_numpy(np.expand_dims(df['t'].values, 1))
        trend = self.model.trend(t).squeeze().detach().numpy()
        trend = trend * self.data_params['y'].scale
        return pd.DataFrame({'ds': df['ds'], 'trend': trend})

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
                predicted[name] = predicted[name] * self.data_params['y'].scale
        return pd.DataFrame({'ds': df['ds'], **predicted})

    def set_true_ar_for_eval(self, true_ar_weights):
        """configures model to evaluate closeness of AR weights to true weights.

        Args:
            true_ar_weights (np.array): True AR-parameters, if known.
        """
        self.true_ar_weights = true_ar_weights

    def set_forecast_in_focus(self, forecast_number=None):
        """Set which forecast step to focus on for metrics evaluation and plotting.

        Args:
            forecast_number (int): i-th step ahead forecast to use for performance statistics evaluation.
                Can also be None.
        """
        if forecast_number is not None:
            assert forecast_number <= self.n_forecasts
        self.forecast_in_focus = forecast_number
        return self

    def add_lagged_regressor(self, name, regularization=None, normalize='auto', only_last_value=False):
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
        # Note: disabled custom n_lags to make code simpler.
        # if n_lags is None:
        #     n_lags = self.n_lags
        # elif self.n_lags < n_lags:
        #         raise ValueError("Exogenous regressors can only be of same or lower order than autoregression.")
        if regularization is not None:
            if regularization < 0: raise ValueError('regularization must be >= 0')
            if regularization == 0: regularization = None
        self._validate_column_name(name)

        if self.covar_config is None: self.covar_config = OrderedDict({})
        self.covar_config[name] = AttrDict({
            "reg_lambda": regularization,
            "normalize": normalize,
            "as_scalar": only_last_value,
        })
        return self

    def add_future_regressor(self, name, regularization=None, normalize='auto', mode="additive"):
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
            if regularization < 0: raise ValueError('regularization must be >= 0')
            if regularization == 0: regularization = None
        self._validate_column_name(name)

        if self.regressors_config is None:
            self.regressors_config = OrderedDict({})
        self.regressors_config[name] = AttrDict({
            "reg_lambda": regularization,
            "normalize": normalize,
            "mode": mode
        })
        return self

    def add_events(self, events, lower_window=0, upper_window=0, regularization=None, mode='additive'):
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
            if regularization < 0: raise ValueError('regularization must be >= 0')
            if regularization == 0: regularization = None

        if not isinstance(events, list):
            events = [events]

        for event_name in events:
            self._validate_column_name(event_name)
            self.events_config[event_name] = AttrDict({
                "lower_window": lower_window,
                "upper_window": upper_window,
                "reg_lambda": regularization,
                "mode": mode
            })
        return self

    def add_country_holidays(self, country_name, lower_window=0, upper_window=0, regularization=None, mode='additive'):
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
            if regularization < 0: raise ValueError('regularization must be >= 0')
            if regularization == 0: regularization = None

        if self.country_holidays_config is None:
            self.country_holidays_config = OrderedDict({})

        self.country_holidays_config["country"] = country_name
        self.country_holidays_config["lower_window"] = lower_window
        self.country_holidays_config["upper_window"] = upper_window
        self.country_holidays_config["reg_lambda"] = regularization
        self.country_holidays_config["holiday_names"] = utils.get_holidays_from_country(country_name)
        self.country_holidays_config["mode"] = mode
        return self

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
            raise Exception("The events configs should be added to the NeuralProphet object (add_events fn)"
                            "before creating the data with events features")
        else:
            df = df_utils.convert_events_to_features(df, events_config=self.events_config, events_df=events_df)

        df.reset_index(drop=True, inplace=True)
        return df

    def plot(self, fcst, ax=None, xlabel='ds', ylabel='y', figsize=(10, 6)):
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
            num_forecasts = sum(fcst['yhat1'].notna())
            if num_forecasts < self.n_forecasts:
                print("Notice: too few forecasts to plot a line per forecast step."
                      "Plotting a line per forecast origin instead.")
                return self.plot_last_forecast(
                    fcst, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize,
                    include_previous_forecasts=num_forecasts - 1, plot_history_data=True)
        return plotting.plot(
            fcst=fcst, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize,
            highlight_forecast=self.forecast_in_focus
        )

    def plot_last_forecast(self, fcst, ax=None, xlabel='ds', ylabel='y', figsize=(10, 6),
                           include_previous_forecasts=0, plot_history_data=None):
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
            fcst = fcst[-(include_previous_forecasts + self.n_forecasts + self.n_lags):]
        elif plot_history_data is False:
            fcst = fcst[-(include_previous_forecasts + self.n_forecasts):]
        elif plot_history_data is True:
            fcst = fcst
        fcst = utils.fcst_df_to_last_forecast(fcst, n_last=1 + include_previous_forecasts)
        return plotting.plot(
            fcst=fcst, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize,
            highlight_forecast=self.forecast_in_focus, line_per_origin=True,
        )

    def plot_components(self, fcst, figsize=(10, 6)):
        """Plot the NeuralProphet forecast components.

        Args:
            fcst (pd.DataFrame): output of self.predict
            figsize (tuple):   width, height in inches. default: (10, 6)
            crop_last_n (int): number of samples to plot (combined future and past)
                None (default) includes entire history. ignored for seasonality.
        Returns:
            A matplotlib figure.
        """
        return plotting.plot_components(
            m=self,
            fcst=fcst,
            figsize=figsize,
            forecast_in_focus=self.forecast_in_focus,
        )

    def plot_parameters(self, weekly_start=0, yearly_start=0, figsize=(10, 6)):
        """Plot the NeuralProphet forecast components.

        Args:
            weekly_start (int): specifying the start day of the weekly seasonality plot.
                0 (default) starts the week on Sunday. 1 shifts by 1 day to Monday, and so on.
            yearly_start (int): specifying the start day of the yearly seasonality plot.
                0 (default) starts the year on Jan 1. 1 shifts by 1 day to Jan 2, and so on.
            figsize (tuple):   width, height in inches. default: (10, 6)
        Returns:
            A matplotlib figure.
        """
        return plotting.plot_parameters(
            m=self,
            forecast_in_focus=self.forecast_in_focus,
            weekly_start=weekly_start,
            yearly_start=yearly_start,
            figsize=figsize,
        )


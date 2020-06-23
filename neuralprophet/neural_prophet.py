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
            holidays (pd.DataFrame):  with columns holiday (string) and ds (date type)
        """
        ## General
        self.name = "NeuralProphet"
        self.verbose = verbose
        self.n_forecasts = max(1, n_forecasts)

        ## Data Preprocessing
        self.normalize_y = normalize_y
        self.data_freq = data_freq
        if self.data_freq != 'D':
            # TODO: implement other frequency handling than daily.
            print("NOTICE: Parts of code may break if using other than daily data.")
        self.impute_missing = impute_missing

        ## Training
        self.train_config = AttrDict({  # TODO allow to be passed in init
            "lr": learning_rate,
            "lr_decay": 0.98,
            "epochs": 50,
            "batch": 128,
            "est_sparsity": ar_sparsity,  # 0 = fully sparse, 1 = not sparse
            "lambda_delay": 10,  # delays start of regularization by lambda_delay epochs
            "reg_lambda_trend": None,
            "trend_reg_threshold": None,
            "reg_lambda_season": None,
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
        # self.growth = "linear" # Prophet Trend related, only linear currently implemented
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
        self.holidays = None
        self.country_holidays = None

        ## Extra Regressors
        self.covar_config = None

        ## Set during _train()
        self.fitted = False
        self.history = None
        self.data_params = None
        self.optimizer = None
        self.scheduler = None
        self.model = None

        ## later set by user
        self.forecast_in_focus = None

    def _init_model(self):
        """Build Pytorch model with configured hyperparamters.

        Returns:
            TimeNet model
        """
        return time_net.TimeNet(
            n_forecasts=self.n_forecasts,
            n_lags=self.n_lags,
            n_changepoints=self.n_changepoints,
            trend_smoothness=self.trend_smoothness,
            num_hidden_layers=self.model_config.num_hidden_layers,
            d_hidden=self.model_config.d_hidden,
            season_dims=utils.season_config_to_model_dims(self.season_config),
            season_mode=self.season_config.mode if self.season_config is not None else None,
            covar_config=self.covar_config,
        )

    def _create_dataset(self, df, predict_mode=False, season_config=None, n_lags=None, n_forecasts=None, verbose=None):
        """Construct dataset from dataframe.

        (Configured Hyperparameters can be overridden by explicitly supplying them.
        Useful to predict a single model component.)
        Args:
            df (pd.DataFrame): containing original and normalized columns 'ds', 'y', 't', 'y_scaled'
            predict_mode (bool): False (default) includes target values.
                True does not include targets but includes entire dataset as input
            season_config (AttrDict): configuration for seasonalities.
            n_lags (int): number of lagged values of series to include. Aka AR-order
            n_forecasts (int): number of steps to forecast into future.
            verbose (bool): whether to print status updates
        Returns:
            TimeDataset
        """
        return time_dataset.TimeDataset(
            df,
            season_config=self.season_config if season_config is None else season_config,
            n_lags=self.n_lags if n_lags is None else n_lags,
            n_forecasts=self.n_forecasts if n_forecasts is None else n_forecasts,
            predict_mode=predict_mode,
            verbose=self.verbose if verbose is None else verbose,
            covar_config=self.covar_config,
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

    def _prep_new_data(self, df, predicting=False, training=False, only_ds=False, allow_missing_dates='auto'):
        """Checks, auto-imputes and normalizes new data

        Args:
            df (pd.DataFrame): raw data with columns 'ds' and 'y'
            predicting (bool): allow NA values in 'y' of forecast series or 'y' to miss completely
            training (bool): fit data params
            only_ds (bool): only ds column
            allow_missing_dates (bool): do not fill missing dates
                (only possible if no lags defined.)

        Returns:
            pre-processed df
        """
        ## Check data sanity
        if predicting and self.n_lags == 0:
            only_ds = True # in some cases when we are predicting we only have 'ds' as input
        regressors_to_check = None
        if not only_ds and self.covar_config is not None:
            regressors_to_check = self.covar_config.keys()
        df = df_utils.check_dataframe(df, check_y=(not only_ds), covariates=regressors_to_check)
        ## add missing dates
        # TODO before commit: uncomment
        # if allow_missing_dates == 'auto':
        #     allow_missing_dates = (self.n_lags == 0 or only_ds)
        # elif allow_missing_dates: assert self.n_lags == 0
        # if allow_missing_dates is False:
        #     df, missing_dates = df_utils.add_missing_dates_nan(df, freq=self.data_freq)
        #     if missing_dates > 0:
        #         if self.impute_missing:
        #             self.verbose: print("NOTICE: {} missing dates were added.".format(missing_dates))
        #         else:
        #             raise ValueError("Missing values found. Please preprocess data manually or set impute_missing to True.")
        ## impute missing values
        data_columns = []
        if not only_ds:
            data_columns.append('y')
            if self.covar_config is not None:
                data_columns.extend(self.covar_config.keys())
        for column in data_columns:
            sum_na = sum(df[column].isnull())
            if sum_na > 0:
                if self.impute_missing is True:
                    df = df_utils.fill_small_linear_large_trend(
                        df, column=column, allow_missing_dates=allow_missing_dates, freq=self.data_freq)
                    print("NOTICE: {} NaN values in column {} were auto-imputed.".format(sum_na, column))
                else:
                    raise ValueError("Missing values found. Please preprocess data manually or set impute_missing to True.")
        ## compute data parameters
        if training:
            assert (self.data_params is None)
            self.data_params = df_utils.init_data_params(
                df, normalize_y=self.normalize_y, covariates_config=self.covar_config)
            if self.verbose:
                print("Data Parameters (shift, scale):",[(k, (v.shift, v.scale)) for k, v in self.data_params.items()])
        df = df_utils.normalize(df, self.data_params)
        return df

    def _prep_data_predict(self, df=None, periods=0, n_history=None, only_ds=False):
        """
        Prepares data for prediction without knowing the true targets.

        Used for model extrapolation into unknown future.
        Args:
            df (pandas DataFrame): Dataframe with columns 'ds' datestamps and 'y' time series values
            periods (int): number of future steps to predict
            n_history (): number of historic/training data steps to include in forecast

        Returns:
            dataset (torch Dataset): Dataset prepared for prediction
            df (pandas DataFrame): input df preprocessed, extended into future, and normalized
        """
        assert (self.data_params is not None)
        if df is None:
            df = self.history.copy()
        else:
            df = self._prep_new_data(df, predicting=True, only_ds=only_ds)

        if periods > 0:
            future_df = df_utils.make_future_df(df, periods=periods, freq=self.data_freq)
            future_df['ds'] = df_utils.normalize(pd.DataFrame({'ds': future_df['ds']}), self.data_params)

        if n_history is None:
            df = df
        elif n_history > 0 or self.n_lags > 0:
            df = df[-(self.n_lags + n_history):]

        if periods > 0:
            if n_history is None or n_history > 0 or self.n_lags > 0:
                df = df.append(future_df)
            else:
                df = future_df

        df.reset_index(drop=True, inplace=True)
        dataset = self._create_dataset(df, predict_mode=True)
        return dataset, df

    def _validate_column_name(self, name, check_holidays=True, check_seasonalities=True, check_regressors=True):
        """Validates the name of a seasonality, holiday, or regressor.

        Args:
            name (str):
            check_holidays (bool):  check if name already used for holiday
            check_seasonalities (bool):  check if name already used for seasonality
            check_regressors (bool): check if name already used for regressor
        """
        if '_delim_' in name:
            raise ValueError('Name cannot contain "_delim_"')
        reserved_names = [
            'trend', 'additive_terms', 'daily', 'weekly', 'yearly',
            'holidays', 'zeros', 'extra_regressors_additive', 'yhat',
            'extra_regressors_multiplicative', 'multiplicative_terms',
        ]
        rn_l = [n + '_lower' for n in reserved_names]
        rn_u = [n + '_upper' for n in reserved_names]
        reserved_names.extend(rn_l)
        reserved_names.extend(rn_u)
        reserved_names.extend(['ds', 'y', 'cap', 'floor', 'y_scaled', 'cap_scaled'])
        if name in reserved_names:
            raise ValueError('Name {name!r} is reserved.'.format(name=name))
        if check_holidays and self.holidays is not None:
            if name in self.holidays['holiday'].unique():
                raise ValueError('Name {name!r} already used for a holiday.'
                                 .format(name=name))
        if check_holidays and self.country_holidays is not None:
            if name in get_holiday_names(self.country_holidays):
                raise ValueError('Name {name!r} is a holiday name in {country_holidays}.'
                                 .format(name=name, country_holidays=self.country_holidays))
        if check_seasonalities and self.season_config is not None:
            if name in self.season_config.periods:
                raise ValueError('Name {name!r} already used for a seasonality.'
                                 .format(name=name))
        if check_regressors and self.covar_config is not None:
            if name in self.covar_config:
                raise ValueError('Name {name!r} already used for an added regressor.'
                                 .format(name=name))

    def _init_train_loader(self, df):
        """Executes data preparation steps and initiates training procedure.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with training data

        Returns:
            torch DataLoader
        """
        if self.fitted is True: raise Exception('Model object can only be fit once. Instantiate a new object.')
        df = self._prep_new_data(df, training=True)
        self.history = df.copy(deep=True)
        # self.history_dates = pd.to_datetime(df['ds']).sort_values()
        self.season_config = utils.set_auto_seasonalities(
            dates=df['ds'], season_config=self.season_config, verbose=self.verbose)
        self.model = self._init_model()
        if self.verbose: print(self.model)
        self.train_config.lr = self._auto_learning_rate(multiplier=self.train_config.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_config.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.train_config.lr_decay)
        dataset = self._create_dataset(df, predict_mode=False)
        loader = DataLoader(dataset, batch_size=self.train_config["batch"], shuffle=True)
        return loader

    def _init_val_loader(self, df):
        """Executes data preparation steps and initiates evaluation procedure.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with validation data

        Returns:
            torch DataLoader
        """
        assert (self.data_params is not None)
        df = self._prep_new_data(df)
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
            self.metrics.update(predicted=predicted, target=targets,
                                values={"Loss": loss, "RegLoss": reg_loss,})
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
                val_metrics.update(predicted=predicted, target=targets)
            val_metrics = val_metrics.compute(save=True)
        return val_metrics

    def _train(self, loader):
        """Execute model training procedure for a configured number of epochs.

        Args:
            loader (torch DataLoader):  instantiated Training Dataloader (with TimeDataset)
        Returns:
            df with metrics
        """
        if self.forecast_in_focus is not None:
            self.metrics.add_specific_target(target_pos=self.forecast_in_focus - 1)
        start = time.time()
        for e in range(self.train_config.epochs):
            self.metrics.reset()
            epoch_metrics = self._train_epoch(e, loader)
            if self.verbose: utils.print_epoch_metrics(epoch_metrics, e=e)
        self.fitted = True
        if self.verbose:
            print("Train Time: {:8.3f}".format(time.time() - start))
            print("Total Batches: ", self.metrics.total_updates)
        train_metrics = self.metrics.get_stored_as_df()
        return train_metrics

    def _train_and_eval(self, train_loader, val_loader):
        """Train model and evaluate after each epoch.

        Args:
            train_loader (torch DataLoader):  instantiated Training Dataloader (with TimeDataset)
            val_loader (torch DataLoader):  instantiated Validation Dataloader (with TimeDataset)
        Returns:
            df with metrics
        """
        val_metrics = metrics.MetricsCollection([m.new() for m in self.metrics.batch_metrics])
        if self.forecast_in_focus is not None:
            self.metrics.add_specific_target(target_pos=self.forecast_in_focus - 1)
            val_metrics.add_specific_target(target_pos=self.forecast_in_focus - 1)
        start = time.time()
        for e in range(self.train_config.epochs):
            self.metrics.reset()
            train_epoch_metrics = self._train_epoch(e, train_loader)
            val_epoch_metrics = self._evaluate_epoch(val_loader, val_metrics)
            if self.verbose: utils.print_epoch_metrics(train_epoch_metrics, val_metrics=val_epoch_metrics, e=e)
        self.fitted = True
        if self.verbose:
            print("Train Time: {:8.3f}".format(time.time() - start))
            print("Total Batches: ", self.metrics.total_updates)
        metrics_df = self.metrics.get_stored_as_df()
        metrics_df_val = val_metrics.get_stored_as_df()
        for col in metrics_df_val.columns:
            metrics_df["{}_val".format(col)] = metrics_df_val[col]
        return metrics_df

    def _evaluate(self, loader, true_ar=None, verbose=None):
        """Evaluates model performance.

        Args:
            loader (torch DataLoader):  instantiated Validation Dataloader (with TimeDataset)
            true_ar (np.array): True AR-parameters, if known.
        Returns:
            df with evaluation metrics
        """
        if self.fitted is False: raise Exception('Model object needs to be fit first.')
        if verbose is None: verbose = self.verbose
        val_metrics = metrics.MetricsCollection([m.new() for m in self.metrics.batch_metrics])
        if self.forecast_in_focus is not None:
            val_metrics.add_specific_target(target_pos=self.forecast_in_focus - 1)

        val_metrics_dict = self._evaluate_epoch(loader, val_metrics)

        if true_ar is not None and true_ar is not False:
            assert self.n_lags > 0
            if self.forecast_in_focus is None:
                if self.n_lags > 1: raise ValueError("Please define forecast_lag for sTPE computation")
                forecast_pos = 1
            else: forecast_pos = self.forecast_in_focus
            weights = self.model.ar_weights.detach().numpy()
            weights = weights[forecast_pos - 1, :][::-1]
            val_metrics_dict["sTPE"] = utils.symmetric_total_percentage_error(true_ar, weights)
            if verbose:
                print("AR parameters: ", true_ar, "\n", "Model weights: ",weights)
        if verbose:
            print("Validation metrics:")
            utils.print_epoch_metrics(val_metrics_dict)
        val_metrics_df = val_metrics.get_stored_as_df()
        return val_metrics_df

    def split_df(self, df, valid_p=0.2, inputs_overbleed=True, verbose=None):
        """Splits timeseries df into train and validation sets.

        Convenience function. See documentation on df_utils.split_df."""
        df = df_utils.check_dataframe(df, check_y=False)
        df_train, df_val = df_utils.split_df(
            df,
            n_lags=self.n_lags,
            n_forecasts=self.n_forecasts,
            valid_p=valid_p,
            inputs_overbleed=inputs_overbleed,
            verbose=self.verbose if verbose is None else verbose,
        )
        return df_train, df_val

    def fit(self, df, test_each_epoch=False, valid_p=0.2):
        """Train, and potentially evaluate model.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with all data
            test_each_epoch (bool): whether to evaluate performance after each training epoch
            valid_p (float): fraction of data to hold out from training for model evaluation
        Returns:
            metrics with training and potentially evaluation metrics
        """
        if test_each_epoch:
            df_train, df_val = self.split_df(df, valid_p=valid_p)
            train_loader = self._init_train_loader(df_train)
            val_loader = self._init_val_loader(df_val)
            return self._train_and_eval(train_loader, val_loader)
        else:
            train_loader = self._init_train_loader(df)
            return self._train(train_loader)

    def test(self, df, true_ar=None):
        """Evaluate model on holdout data.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with holdout data
            true_ar (np.array): True AR-parameters, if known.
        Returns:
            df with evaluation metrics
        """
        loader = self._init_val_loader(df)
        return self._evaluate(loader, true_ar=true_ar)

    def predict(self, df=None, future_periods=None, n_history=None):
        """
        Runs the model to make predictions.

        TODO: use the forecast at forecast_lag number to show the fit (if multiple forecasts were made)
        and compute stats (MSE, MAE)
        Args:
            future_periods (): number of steps to predict into future.
                if n_lags > 0, must be equal to n_forecasts
            df (pandas DataFrame): Dataframe with columns 'ds' datestamps and 'y' time series values
                if no df is provided, shows predictions over the data history.
            n_history (): number of historic/training data steps to include in forecast
                if n_history is > n_forecasts, a line is plotted for each i-th step ahead forecast
                instead of a line for each forecast.

        Returns:
            df_forecast (pandas DataFrame): columns 'ds', 'y', 'trend' and ['yhat<i>']
                if n_history is > n_forecasts, 'yhat<i>' is the forecast for 'y' at 'ds' from i steps ago.
                if n_history is <= n_forecasts, 'yhat<i>' is the forecast given at i steps before the end of data.

        """
        if self.history is None:
            raise Exception('Model has not been fit.')
        if future_periods is None:
            future_periods = self.n_forecasts
        if self.n_lags > 0:
            if future_periods < self.n_forecasts:
                future_periods = self.n_forecasts
                print("NOTICE: parameter future_periods set to n_forecasts Autoregression is present.")
            elif future_periods > self.n_forecasts:
                future_periods = self.n_forecasts
                print("NOTICE: parameter future_periods set to n_forecasts Autoregression is present.")
                print("Unrolling of AR forecasts into the future beyond n_forecasts is not implemented.")
        dataset, df = self._prep_data_predict(df, periods=future_periods, n_history=n_history)
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
        for name, value in components.items():
            if not ('season' in name and self.season_config.mode == 'multiplicative'):
                components[name] = value * scale_y

        # trend = self.predict_trend(df['ds'].copy(deep=True))
        # cols = ['ds', 'y', 'trend'] #cols to keep from df
        cols = ['ds', 'y'] #cols to keep from df
        df_forecast = pd.concat((df[cols],), axis=1)

        if n_history is not None and n_history <= self.n_forecasts:
            # create a line for each foreacast
            for i in range(n_history+1):
                forecast_age = i
                forecast = predicted[-1 -forecast_age, :]
                pad_before = self.n_lags + n_history - forecast_age
                pad_after = forecast_age
                yhat = np.concatenate(([None] * pad_before, forecast, [None] * pad_after))
                df_forecast['yhat{}'.format(i + 1)] = yhat
        else:
            # create a line for each forecast_lag
            for i in range(self.n_forecasts):
                forecast_lag = i + 1
                forecast = predicted[:, forecast_lag - 1]
                pad_before = self.n_lags + forecast_lag - 1
                pad_after = self.n_forecasts - forecast_lag
                yhat = np.concatenate(([None] * pad_before, forecast, [None] * pad_after))
                df_forecast['yhat{}'.format(i+1)] = yhat

        lagged_components = ['ar', ]
        for comp in lagged_components:
            if comp in components:
                for i in range(self.n_forecasts):
                    forecast_lag = i + 1
                    forecast = components['ar'][:, forecast_lag - 1]
                    pad_before = self.n_lags + forecast_lag - 1
                    pad_after = self.n_forecasts - forecast_lag
                    yhat = np.concatenate(([None] * pad_before, forecast, [None] * pad_after))
                    df_forecast['ar{}'.format(i+1)] = yhat

        for comp in components:
            if comp not in lagged_components:
                # only for non-lagged components
                forecast_0 = components[comp][0, :]
                forecast_rest = components[comp][1:, self.n_forecasts - 1]
                yhat = np.concatenate(([None]*self.n_lags, forecast_0, forecast_rest))
                df_forecast[comp] = yhat
        return df_forecast

    def predict_trend(self, dates, future_periods=0, n_history=None):
        """Predict only trend component of the model.

        Args:
            dates (pandas DataFrame): containing column 'ds', prediction dates
            future_periods (): number of steps to predict into future.
            n_history (): number of historic/training data steps to include in forecast
                None defaults to entire history

        Returns:
            numpy Vector with trend on prediction dates.

        """
        print("DEPRECATED: predict_trend, "
              "use predict instead and retrieve trend component from forecast df")
        df_ds = pd.DataFrame({'ds': dates, })
        _, df_ds = self._prep_data_predict(df_ds, periods=future_periods, n_history=n_history, only_ds=True)
        t = torch.from_numpy(np.expand_dims(df_ds['t'].values, 1))
        trend = self.model.trend(t).squeeze().detach().numpy()
        trend = trend * self.data_params['y'].scale
        return trend

    def predict_seasonal_components(self, df):
        """Predict seasonality components

        Args:
            df (pd.DataFrame): containing column 'ds', prediction dates

        Returns:
            pd.Dataframe with seasonal components. with columns of name <seasonality component name>

        """
        print("DEPRECATED: predict_seasonal_components, "
              "use predict instead and retrieve season component from forecast df")
        df = self._prep_new_data(df)
        dataset = self._create_dataset(df, predict_mode=True, n_lags=0, n_forecasts=1, verbose=False)
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
        return pd.DataFrame(predicted)

    def get_last_forecasts(self, n_last_forecasts=1, df=None, future_periods=None,):
        """
        Computes the n last forecasts into the future, at the end of known data.

        Args:
            n_last_forecasts (): how many forecasts to show.
                if more than 1, forecasts that miss the n - 1 last data samples as inputs are included.
                if n_last_forecasts is > n_forecasts, a line is plotted for each i-th step ahead forecast
                    instead of a line for each forecast.
            df (): see self.predict()
            future_periods (): see self.predict()

        Returns:
            see self.predict()

        """
        return self.predict(df=df, future_periods=future_periods, n_history=n_last_forecasts-1)

    def set_forecast_in_focus(self, forecast_number=None):
        """Set which forecast step to focus on for metrics evaluation and plotting.

        Args:
            forecast_number (int): i-th step ahead forecast to use for performance statistics evaluation.
                Can also be None.
        """
        if forecast_number is not None:
            assert forecast_number <= self.n_forecasts
        self.forecast_in_focus = forecast_number

    def add_covariate(self, name, regularization=None, normalize='auto'):
        """Add a covariate time series as an additional lagged regressor to be used for fitting and predicting.

        The dataframe passed to `fit` and `predict` will have a column with the specified name to be used as
        a lagged regressor. When normalize=True, the covariate will be normalized unless it is binary.

        Args:
            name (string):  name of the regressor.
            regularization (float): optional  scale for regularization strength
            normalize (bool): optional, specify whether this regressor will be
                normalized prior to fitting.
                if 'auto', binary regressors will not be normalized.

        Returns:
            NeuralProphet object
        """
        if self.fitted: raise Exception("Covariates must be added prior to model fitting.")
        # Note: disabled custom n_lags to make code simpler.
        # if self.n_lags == 0: raise ValueError("Covariates can only be used with autoregression enabled.")
        # if n_lags is None:
        #     n_lags = self.n_lags
        # elif self.n_lags < n_lags:
        #         raise ValueError("Exogenous regressors can only be of same or lower order than autoregression.")
        if regularization is not None:
            if regularization < 0: raise ValueError('regularization must be > 0')
            if regularization == 0: regularization = None
        self._validate_column_name(name, check_regressors=False)

        if self.covar_config is None: self.covar_config = OrderedDict({})
        self.covar_config[name] = AttrDict({
            "reg_lambda": regularization,
            "normalize": normalize,
        })
        return self

    def plot(self, fcst, ax=None, xlabel='ds', ylabel='y', figsize=(10, 6), crop_last_n=None):
        """Plot the NeuralProphet forecast, including history.

        Args:
            fcst (pd.DataFrame): output of self.predict.
            ax (matplotlib axes): Optional, matplotlib axes on which to plot.
            xlabel (string): label name on X-axis
            ylabel (string): label name on Y-axis
            figsize (tuple):   width, height in inches.
            crop_last_n (int): number of samples to plot (combined future and past)

        Returns:
            A matplotlib figure.
        """
        if crop_last_n is not None:
            fcst = fcst[-crop_last_n:]

        return plotting.plot(
            fcst=fcst, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize,
            highlight_forecast=self.forecast_in_focus
        )

    def plot_components(self, fcst, weekly_start=0, yearly_start=0, figsize=None, crop_last_n=None,):
        """Plot the Prophet forecast components.

        Args:
            fcst (pd.DataFrame): output of self.predict
            weekly_start (int): specifying the start day of the weekly seasonality plot.
                0 (default) starts the week on Sunday. 1 shifts by 1 day to Monday, and so on.
            yearly_start (int): specifying the start day of the yearly seasonality plot.
                0 (default) starts the year on Jan 1. 1 shifts by 1 day to Jan 2, and so on.
            figsize (tuple):   width, height in inches.
            crop_last_n (int): number of samples to plot (combined future and past)
                None (default) includes entire history. ignored for seasonality.
        Returns:
            A matplotlib figure.
        """
        if crop_last_n is not None:
            fcst = fcst[-crop_last_n:]
        return plotting.plot_components(
            m=self,
            fcst=fcst,
            weekly_start=weekly_start,
            yearly_start=yearly_start,
            figsize=figsize,
            ar_coeff_forecast_n=self.forecast_in_focus,
        )

    def plot_last_forecasts(self, n_last_forecasts=1, df=None, future_periods=None,
                            ax=None, xlabel='ds', ylabel='y', figsize=(10, 6)):
        fcst = self.predict(df=df, future_periods=future_periods, n_history=n_last_forecasts-1)
        return self.plot(fcst, highlight_forecast=1, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize)

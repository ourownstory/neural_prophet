import time
from collections import OrderedDict
from attrdict import AttrDict
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
            trend_threshold=True,
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality='auto',
            seasonality_mode='additive',
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
            seasonality_type (str): 'linear', 'fourier' type of seasonality modelling
            freq (str):Data step sizes. Frequency of data recording,
                Any valid frequency for pd.date_range, such as 'D' or 'M'
            impute_missing (bool): whether to automatically impute missing dates/values
                imputation follows a linear method up to 10 missing values, more are filled with trend.
            verbose (bool): Whether to print procedure status updates for debugging/monitoring

        TODO:
            seasonality_smoothness (float): Parameter modulating the strength of the
                seasonality model. Smaller values allow the model to fit larger seasonal
                fluctuations, larger values dampen the seasonality.
                Can be specified for individual seasonalities using add_seasonality.
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
        })
        if loss_func.lower() in ['huber','smoothl1', 'smoothl1loss']:
            self.loss_fn = torch.nn.SmoothL1Loss()
        elif loss_func.lower() in ['mae', 'l1', 'l1loss']:
            self.loss_fn = torch.nn.L1Loss()
        elif loss_func.lower() in ['mse', 'mseloss', 'l2', 'l2loss']:
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise NotImplementedError("Loss function {} not found".format(loss_func))


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
            self.train_config.reg_lambda_trend = self.trend_smoothness * np.sqrt(self.n_changepoints)
            self.train_config.trend_reg_threshold = None
            if trend_threshold is not None and trend_threshold is not False:
                if trend_threshold == 'auto' or trend_threshold is True:
                    self.train_config.trend_reg_threshold = 100.0 / (self.trend_smoothness * self.n_changepoints)
                else:
                    self.train_config.trend_reg_threshold = trend_threshold

        ## Seasonality
        self.season_config = AttrDict({})
        self.season_config.type = 'fourier'  # Currently no other seasonality_type
        self.season_config.mode = seasonality_mode
        self.season_config.periods = OrderedDict({ # defaults
            "yearly": AttrDict({'resolution': 8, 'period': 365.25, 'arg': yearly_seasonality}),
            "weekly": AttrDict({'resolution': 4, 'period': 7, 'arg': weekly_seasonality,}),
            "daily": AttrDict({'resolution': 8, 'period': 1, 'arg': daily_seasonality,}),
        })

        ## Set during _train()
        self.fitted = False
        self.history = None
        self.data_params = None
        self.optimizer = None
        self.scheduler = None
        self.model = None

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

    def _init_train_loader(self, df):
        """Executes data preparation steps and initiates training procedure.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with training data

        Returns:
            torch DataLoader
        """
        if self.fitted is True:
            raise Exception('Model object can only be fit once. Instantiate a new object.')
        else:
            assert (self.data_params is None)
        self.data_params = df_utils.init_data_params(df, normalize_y=self.normalize_y, verbose=self.verbose)
        df = self._prep_new_data(df)
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

    def _train(self, loader):
        """Execute model training procedure for a configured number of epochs.

        Args:
            loader (torch DataLoader):  instantiated Training Dataloader (with TimeDataset)

        Returns:

        """
        results = AttrDict({
            'epoch_losses': [],
            'epoch_regularizations': [],
        })
        total_batches = 0
        start = time.time()
        for e in range(self.train_config.epochs):
            epoch_loss, epoch_reg, batches = self._train_epoch(e, loader)
            results["epoch_losses"].append(epoch_loss)
            results["epoch_regularizations"].append(epoch_reg)
            total_batches += batches
        results["time_train"] = time.time() - start
        results["loss_train"] = results["epoch_losses"][-1]
        if self.verbose:
            print("Train Time: {:8.4f}".format(results["time_train"]))
            print("Total Number of Batches: ", total_batches)
        return results

    def _train_epoch(self, e, loader):
        """Make one complete iteration over all samples in dataloader and update model after each batch.

        Args:
            e (int): current epoch number
            loader (torch DataLoader): Training Dataloader

        Returns:
            epoch_loss: Average training loss of current training epoch (without regularization losses)
            epoch_reg: Average regularization loss of current training epoch (without training losses)
            num_batches: total number of update steps taken
        """
        # slowly increase regularization until lambda_delay epoch
        reg_lambda_ar = None
        if self.n_lags > 0:
            reg_lambda_ar = utils.get_regularization_lambda(
                self.train_config.est_sparsity, self.train_config.lambda_delay, e)

        num_batches = 0
        current_epoch_losses = []
        current_epoch_reg_losses = []
        for inputs, targets in loader:
            # Run forward calculation
            predicted = self.model.forward(**inputs)

            # Compute loss.
            loss = self.loss_fn(predicted, targets)
            current_epoch_losses.append(loss.data.item())

            # Add regularization of AR weights
            reg_loss_ar = torch.zeros(1, dtype=torch.float, requires_grad=False)
            if reg_lambda_ar is not None:
                reg = utils.regulariziation_function_ar(self.model.ar_weights)
                reg_loss_ar = reg_lambda_ar * reg
                loss += reg_lambda_ar * reg

            # Regularize trend to be smoother
            reg_loss_trend = torch.zeros(1, dtype=torch.float, requires_grad=False)
            if self.train_config.reg_lambda_trend is not None:
                reg = utils.regulariziation_function_trend(
                    weights=self.model.get_trend_deltas,
                    threshold=self.train_config.trend_reg_threshold,
                )
                reg_loss_trend = self.train_config.reg_lambda_trend * reg
                loss += self.train_config.reg_lambda_trend * reg

            current_epoch_reg_losses.append((reg_loss_ar + reg_loss_trend).data.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            num_batches += 1

        self.scheduler.step()
        epoch_loss = np.mean(current_epoch_losses)
        epoch_reg = np.mean(current_epoch_reg_losses)
        if self.verbose:
            print("{}. Epoch Avg Loss: {:10.2f}".format(e + 1, epoch_loss))
        return epoch_loss, epoch_reg, num_batches

    def _evaluate(self, df, true_ar=None, forecast_lag=None):
        """Evaluates model performance.

        TODO: Update

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with holdout data
            true_ar (np.array): True AR-parameters, if known.
            forecast_lag (int): i-th step ahead forecast to use for performance statistics evaluation.

        Returns:
            dict with evaluation results and statistics
        """
        if self.fitted is False:
            raise Exception('Model object needs to be fit first.')
        assert (self.data_params is not None)
        df = self._prep_new_data(df)
        dataset = self._create_dataset(df, predict_mode=False)
        loader = DataLoader(dataset, batch_size=min(1024, len(dataset)), shuffle=False, drop_last=False)
        results = AttrDict({})
        losses = list()
        targets_vectors = list()
        predicted_vectors = list()
        batch_index = 0
        for inputs, targets in loader:
            predicted = self.model.forward(**inputs)
            loss = self.loss_fn(predicted, targets)
            losses.append(loss.data.numpy())
            targets_vectors.append(targets.data.numpy())
            predicted_vectors.append(predicted.data.numpy())
            batch_index += 1

        results["loss_val"] = np.mean(np.array(losses))
        predicted = np.concatenate(predicted_vectors)
        actual = np.concatenate(targets_vectors)
        results["predicted"] = predicted
        results["actual"] = actual
        if forecast_lag is None:
            mse = np.mean((predicted - actual) ** 2)
        else:
            assert forecast_lag <= self.n_forecasts
            mse = np.mean((predicted[:, forecast_lag - 1] - actual[:, forecast_lag - 1]) ** 2)
        results["mse_val"] = mse

        if self.n_lags is not None and self.n_lags >= 1:
            weights_rereversed = self.model.ar_weights.detach().numpy()[0, ::-1]
            results["weights"] = weights_rereversed
        if true_ar is not None:
            results["true_ar"] = true_ar
            sTPE = utils.symmetric_total_percentage_error(results["true_ar"], results["weights"])
            results["sTPE (weights)"] = sTPE
            if self.verbose:
                print("sTPE (weights): {:6.3f}".format(results["sTPE (weights)"]))
                print("AR params: ")
                print(results["true_ar"])
                print("Weights: ")
                print(results["weights"])

        if self.verbose:
            print("Validation MSEs{}:".format("" if forecast_lag is None else
                                              " for {}-step ahead".format(forecast_lag)) +
                " {:10.2f}".format(results["mse_val"]))
        return results

    def _prep_new_data(self, df, allow_na=False):
        """Checks, auto-imputes and normalizes new data

        Args:
            df (pd.DataFrame): raw data with columns 'ds' and 'y'

        Returns:
            pre-processed df
        """
        allow_missing_dates = self.n_lags == 0
        if allow_missing_dates is not True:
            df, missing_dates = df_utils.add_missing_dates_nan(df, freq=self.data_freq)
            if self.verbose and missing_dates > 0:
                print("NOTICE: {} missing dates were added.".format(missing_dates))
        sum_na = sum(df['y'].isnull())
        if allow_na is not True and sum_na > 0:
            if self.impute_missing is True:
                df = df_utils.fill_small_linear_large_trend(df, freq=self.data_freq)
                assert sum(df['y'].isnull()) == 0
                print("NOTICE: {} NaN values were auto-imputed.".format(sum_na))
            else:
                raise ValueError("Missing dates found. Please preprocess data manually or set impute_missing to True.")
        df = df_utils.check_dataframe(df)
        df = df_utils.normalize(df, self.data_params)
        return df

    def _prep_data_predict(self, df=None, periods=0, n_history=None):
        """
        Prepares data for prediction without knowing the true targets.

        Used for model extrapolation into unknown future.
        Args:
            df (pandas DataFrame): Dataframe with columns 'ds' datestamps and 'y' time series values
            periods (): number of future steps to predict
            n_history (): number of historic/training data steps to include in forecast

        Returns:
            dataset (torch Dataset): Dataset prepared for prediction
            df (pandas DataFrame): input df preprocessed, extended into future, and normalized
        """
        assert (self.data_params is not None)
        if df is None:
            df = self.history.copy()
        else:
            df = self._prep_new_data(df, allow_na=True)
        # print(periods, n_history, self.n_lags, self.n_forecasts)
        if periods > 0:
            future_df = df_utils.make_future_df(df, periods=periods, freq=self.data_freq)
            future_df = df_utils.normalize(future_df, self.data_params)
        if n_history is not None:
            if not (n_history == 0 and self.n_lags == 0):
                df = df[-(self.n_lags + n_history):]
                # print(df)
        if periods > 0:
            if n_history == 0 and self.n_lags == 0:
                df = future_df
            else:
                df = df.append(future_df)
        df.reset_index(drop=True, inplace=True)
        dataset = self._create_dataset(df, predict_mode=True)
        return dataset, df

    def fit(self, df):
        """Fit model on training data.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with training data

        Returns:
            dict with training results
        """
        loader = self._init_train_loader(df)
        self._train(loader)

    def test(self, df, true_ar=None):
        """Evaluate model on holdout data.

        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with holdout data

        Returns:
            dict with evaluation results and statistics
        """
        self._evaluate(df, true_ar=true_ar)

    def train_eval(self, df, valid_p=0.2, true_ar=None):
        """Train and evaluate model.

        Utility function, useful to compare model performance over a range of hyperparameters
        Args:
            df (pd.DataFrame): containing column 'ds', 'y' with all data
            valid_p (int): fraction of datato hold out from training for model evaluation
            true_ar (np.array): True AR-parameters, if known.

        Returns:
            TODO
        """
        df = self._prep_new_data(df)
        df_train, df_val = df_utils.split_df(df, self.n_lags, self.n_forecasts, valid_p, inputs_overbleed=True, verbose=self.verbose)
        train_loader = self._init_train_loader(df)
        results_train = self._train(train_loader)
        results_val = self._evaluate(df_val, true_ar=true_ar)
        raise NotImplementedError
        # return results

    def predict(self, future_periods=None, df=None, n_history=None):
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
        # targets_vectors = list()
        for inputs, _ in loader:
            predicted = self.model.forward(**inputs)
            predicted_vectors.append(predicted.data.numpy())
            # targets_vectors.append(_.data.numpy())

        predicted = np.concatenate(predicted_vectors)
        predicted = predicted * self.data_params.y_scale + self.data_params.y_shift
        # print(df)
        trend = self.predict_trend(df)
        # print(len(trend))
        df.loc[:, 'trend'] = trend
        cols = ['ds', 'y', 'trend'] #cols to keep from df
        df_forecast = pd.concat((df[cols],), axis=1)

        # just for debugging - to check if we got all indices right:
        # actual = np.concatenate(targets_vectors)
        # actual = actual * self.data_params.y_scale + self.data_params.y_shift
        # actual = actual[:, forecast_lag-1]
        # df2['actual'] = np.concatenate(([None]*(self.n_lags + forecast_lag - 1),
        #                        actual,
        #                        [None]*(self.n_forecasts - forecast_lag)))

        if n_history is not None and n_history <= self.n_forecasts:
            # create a line for each foreacast
            for i in range(n_history+1):
                forecast_age = i
                forecast = predicted[-1 -forecast_age, :]
                yhat = np.concatenate(([None] * (self.n_lags + n_history - forecast_age),
                                       forecast,
                                       [None] * forecast_age))
                df_forecast['yhat{}'.format(i + 1)] = yhat
        else:
            # create a line for each forecast_lag
            for i in range(self.n_forecasts):
                forecast_lag = i + 1
                forecast = predicted[:, forecast_lag - 1]
                yhat = np.concatenate(([None] * (self.n_lags + forecast_lag - 1),
                                       forecast,
                                       [None] * (self.n_forecasts - forecast_lag)))
                df_forecast['yhat{}'.format(i+1)] = yhat
        return df_forecast

    def predict_trend(self, df, future_periods=0, n_history=None):
        """Predict only trend component of the model.

        Args:
            df (pandas DataFrame): containing column 'ds', prediction dates
            future_periods (): number of steps to predict into future.
            n_history (): number of historic/training data steps to include in forecast
                None defaults to entire history

        Returns:
            numpy Vector with trend on prediction dates.

        """
        _, df = self._prep_data_predict(df, periods=future_periods, n_history=n_history)
        t = torch.from_numpy(np.expand_dims(df['t'].values, 1))
        trend = self.model.trend(t).squeeze().detach().numpy()
        trend = trend * self.data_params.y_scale + self.data_params.y_shift
        return trend

    def predict_seasonal_components(self, df):
        """Predict seasonality components

        Args:
            df (pd.DataFrame): containing column 'ds', prediction dates

        Returns:
            pd.Dataframe with seasonal components. with columns of name <seasonality component name>

        """
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
                predicted[name] = predicted[name] * self.data_params.y_scale + self.data_params.y_shift

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

    def plot(self, fcst, highlight_forecast=None, ax=None, xlabel='ds', ylabel='y', figsize=(10, 6), crop_last_n=None):
        """Plot the NeuralProphet forecast, including history.

        Args:
            fcst (pd.DataFrame): output of self.predict.
            highlight_forecast (int): which yhat<i> forecasts to highlight
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
        if highlight_forecast is not None and highlight_forecast > self.n_forecasts:
            highlight_forecast = self.n_forecasts
            print("NOTICE: highlight_forecast > n_forecasts given. "
                "highlight_forecast reduced to n_forecasts")
        return plotting.plot(
            fcst=fcst, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize,
            highlight_forecast=highlight_forecast
        )

    def plot_components(self, fcst, weekly_start=0, yearly_start=0, figsize=None, crop_last_n=None, ar_coeff_forecast_n=None,):
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
            ar_coeff_forecast_n (int): n-th step ahead forecast AR-coefficients to plot

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
            ar_coeff_forecast_n=ar_coeff_forecast_n,
        )

    def plot_last_forecasts(self, n_last_forecasts=1, df=None, future_periods=None,
                            ax=None, xlabel='ds', ylabel='y', figsize=(10, 6)):
        fcst = self.predict(df=df, future_periods=future_periods, n_history=n_last_forecasts-1)
        return self.plot(fcst, highlight_forecast=1, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize)

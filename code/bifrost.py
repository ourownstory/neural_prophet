import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from torch import optim
from attrdict import AttrDict
from collections import OrderedDict
import time

from code.make_dataset import check_dataframe, split_df, init_data_params, normalize, \
    tabularize_univariate_datetime, TimeDataset
from code.model import TimeNet
import code.utils as utils
import code.plotting as plotting


class Bifrost:
    """Bifrost forecaster.

    Parameters
    ----------
    n_forecasts: int, Number of steps ahead of prediction time step to forecast.
    n_lags: int, Previous time series steps to include in auto-regression. Aka AR-order
    n_changepoints: int, Number of potential changepoints to include.
        TODO: Not used if input `changepoints` is supplied. If `changepoints` is not supplied,
        then n_changepoints potential changepoints are selected uniformly from
        the first `changepoint_range` proportion of the history.
    learnign_rate: Multiplier for learning rate. Try values ~0.001-10.
    normalize_y: Bool, Whether to normalize the time series before modelling it.
    num_hidden_layers: int, number of hidden layer to include in AR-Net. defaults to 0.
    d_hidden: int, dimension of hidden layers of the AR-Net. Ignored if num_hidden_layers == 0.
    ar_sparsity: float, [0-1], how much sparsity to enduce in the AR-coefficients.
        Should be around (# nonzero components) / (AR order), eg. 3/100 = 0.03
    trend_smoothness: Parameter modulating the flexibility of the automatic changepoint selection.
        Large values (~1-100) will limit the variability of changepoints.
        Small values (~0.001-1.0) will allow changepoints to change faster.
        default: 0 will fully fit a trend to each segment.
        -1 will allow discontinuous trend (overfitting danger)
    yearly_seasonality: Fit yearly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    weekly_seasonality: Fit weekly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    daily_seasonality: Fit daily seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    seasonality_mode: 'additive' (default) or 'multiplicative'.
    seasonality_type 'linear', 'fourier'
    TODO: seasonality_smoothness: Parameter modulating the strength of the
        seasonality model. Smaller values allow the model to fit larger seasonal
        fluctuations, larger values dampen the seasonality.
        Can be specified for individual seasonalities using add_seasonality.
    TODO: changepoints: List of dates at which to include potential changepoints. If
        not specified, potential changepoints are selected automatically.
    TODO: changepoint_range: Proportion of history in which trend changepoints will
        be estimated. Defaults to 0.9 for the first 90%. Not used if
        `changepoints` is specified.
    TODO: holidays: pd.DataFrame with columns holiday (string) and ds (date type)

    verbose: Whether to print procedure status updates for debugging/monitoring
    """
    def __init__(
            self,
            n_forecasts=1,
            n_lags=0,
            n_changepoints=5,
            learnign_rate=1.0,
            normalize_y=True,
            num_hidden_layers=0,
            d_hidden=None,
            ar_sparsity=None,
            trend_smoothness=0,
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality='auto',
            seasonality_mode='additive',
            seasonality_type='linear',
            verbose=False,
    ):
        ## General
        self.name = "Bifrost"
        self.verbose = verbose
        self.n_forecasts = n_forecasts if n_forecasts > 0 else 1
        self.normalize_y = normalize_y

        ## Training
        self.train_config = AttrDict({  # TODO allow to be passed in init
            "lr": learnign_rate,
            "lr_decay": 0.98,
            "epochs": 50,
            "batch": 128,
            "est_sparsity": ar_sparsity,  # 0 = fully sparse, 1 = not sparse
            "lambda_delay": 10,  # delays start of regularization by lambda_delay epochs
            "reg_lambda_trend": None,
        })
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = torch.nn.SmoothL1Loss()

        ## AR
        self.n_lags = n_lags
        if n_lags == 0 and n_forecasts > 1:
            n_forecasts = 1
            print("NOTICE: changing n_forecasts to 1. Without lags, "
                  "the forecast can be computed for any future time, independent of present values")
        self.model_config = AttrDict({
            "num_hidden_layers": num_hidden_layers,
            "d_hidden": d_hidden,
        })

        ## Trend
        self.n_changepoints = n_changepoints
        self.trend_smoothness = trend_smoothness
        self.growth = "linear" # Prophet Trend related, only linear currently implemented
        if self.n_changepoints > 0 and self.trend_smoothness > 0:
            print("NOTICE: A numeric value greater than 0 for continuous_trend is interpreted as"
                  "the trend changepoint regularization strength. Please note that this feature is experimental.")
            self.train_config.reg_lambda_trend = self.trend_smoothness / np.sqrt(self.n_changepoints)
            self.train_config.trend_reg_threshold = 100.0 / (self.trend_smoothness + self.n_changepoints)

        ## Seasonality
        self.season_config = AttrDict({})
        self.season_config.type = seasonality_type
        self.season_config.mode = seasonality_mode
        self.season_config.periods = OrderedDict({ # defaults
            "yearly": AttrDict({'resolution': 12, 'period': 365.25, 'arg': yearly_seasonality}),
            "weekly": AttrDict({'resolution': 7, 'period': 7, 'arg': weekly_seasonality,}),
            "daily": AttrDict({'resolution': 12, 'period': 1, 'arg': daily_seasonality,}),
        })

        ## Set during _train
        self.fitted = False
        self.history = None
        self.data_params = None
        self.optimizer = None
        self.scheduler = None
        self.model = None

    def _init_model(self):
        """builds Pytorch model with configured hyperparamters."""
        return TimeNet(
            n_forecasts=self.n_forecasts,
            n_lags=self.n_lags,
            n_changepoints=self.n_changepoints,
            trend_smoothness=self.trend_smoothness,
            num_hidden_layers=self.model_config.num_hidden_layers,
            d_hidden=self.model_config.d_hidden,
            season_dims=utils.season_config_to_model_dims(self.season_config),
            season_mode=self.season_config.mode,
        )

    def _create_dataset(self, df, predict_mode=False, season_config=None, n_lags=None, n_forecasts=None, verbose=None):
        """
        Constructs dataset from dataframe. Defaults to training mode.
        Configured Hyperparameters can be overridden by explicitly supplying them.
        (Useful to predict a single model component.)
        returns TimeDataset
        """
        return TimeDataset(
            *tabularize_univariate_datetime(
                df=df,
                season_config=self.season_config if season_config is None else season_config,
                n_lags=self.n_lags if n_lags is None else n_lags,
                n_forecasts=self.n_forecasts if n_forecasts is None else n_forecasts,
                predict_mode=predict_mode,
                verbose=self.verbose if verbose is None else verbose,
            )
        )

    def _auto_learning_rate(self, multiplier=1.0):
        """computes a reasonable guess for a learning rate based on estimated model complexity
        returns learning rate"""
        model_complexity = max(1.0,
                               10 * np.sqrt(self.n_lags * self.n_forecasts)
                               + np.log(1 + self.n_changepoints)
                               + np.log(1 + sum([p.resolution for name, p in self.season_config.periods.items()]))
                               )
        if self.verbose: print("model_complexity", model_complexity)
        return multiplier / model_complexity

    def _init_train_loader(self, df):
        """executes data preparatioin steps and initiation of training procedure.
        returns training DataLoader"""
        if self.fitted is True:
            raise Exception('Model object can only be fit once. Instantiate a new object.')
        else:
            assert (self.data_params is None)
        df = check_dataframe(df)
        self.data_params = init_data_params(df, normalize_y=self.normalize_y, verbose=self.verbose)
        df = normalize(df, self.data_params)
        self.season_config = utils.set_auto_seasonalities(
            dates=df['ds'], season_config=self.season_config, verbose=self.verbose)
        # self.history_dates = pd.to_datetime(df['ds']).sort_values()
        self.history = df.copy(deep=True)

        self.model = self._init_model()
        if self.verbose: print(self.model)
        self.train_config.lr = self._auto_learning_rate(multiplier=self.train_config.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_config.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.train_config.lr_decay)
        dataset = self._create_dataset(df, predict_mode=False)
        loader = DataLoader(dataset, batch_size=self.train_config["batch"], shuffle=True)
        return loader

    def _train(self, loader):
        """
        Execute model training procedure for a configured number of epochs.

        Parameters
        ----------
        loader: instantiated training Dataloader (with TimeDataset)
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
        results["loss_train"] = epoch_losses[-1]
        if self.verbose:
            print("Train Time: {:8.4f}".format(results["time_train"]))
            print("Total Number of Batches: ", total_batches)
        return results

    def _train_epoch(self, e, loader):
        """Make one complete iteration over all samples in dataloader in batches
        and update model after each batch."""
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
                reg_loss_ar = reg_lambda_ar * (reg_loss_ar + torch.mean(reg)).squeeze()
                loss += reg_loss_ar

            # Regularize trend to be smoother
            reg_loss_trend = torch.zeros(1, dtype=torch.float, requires_grad=False)
            if self.train_config.reg_lambda_trend is not None:
                reg = utils.regulariziation_function_trend(
                    weights=self.model.get_trend_deltas,
                    threshold=self.train_config.trend_reg_threshold,
                )
                reg_loss_trend = self.train_config.reg_lambda_trend * torch.sum(reg)
                loss += self.train_config.reg_lambda_trend * torch.sum(reg)

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
        """
        TODO: Update
        """
        if self.fitted is False:
            raise Exception('Model object needs to be fit first.')
        assert (self.data_params is not None)
        df = check_dataframe(df)
        df = normalize(df, self.data_params)
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

    def _prep_data_predict(self, df=None, periods=0, freq='D', n_history=None):
        """Prepares data for prediction without knowing the true targets.
        Used for model extrapolation into unknown future."""
        assert (self.data_params is not None)
        if df is None:
            df = self.history.copy()
        if n_history is not None:
            df = df[-(self.n_lags + n_history - 1):]
        if periods > 0:
            df = self._extend_df_to_future(df, periods=periods, freq=freq)
        df = normalize(df, self.data_params)

        dataset = self._create_dataset(df, predict_mode=True)
        return dataset, df

    def _extend_df_to_future(self, df, periods, freq):
        """extends df periods steps into future."""
        df = check_dataframe(df)
        history_dates = pd.to_datetime(df['ds']).sort_values()
        future_df = utils.make_future_dataframe(history_dates, periods, freq, include_history=False)
        future_df["y"] = None
        df2 = df.append(future_df)
        return df2

    def predict(self, future_periods=None, df=None, freq='D', n_history=None):
        """
        runs the model to make predictions.
        if no df is provided, shows predictions over the data history.
        TODO: uses the forecast at forecast_lag number to show the fit (if multiple forecasts were made)
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
        dataset, df = self._prep_data_predict(df, periods=future_periods, freq=freq, n_history=n_history)
        loader = DataLoader(dataset, batch_size=min(1024, len(df)), shuffle=False, drop_last=False)

        predicted_vectors = list()
        # targets_vectors = list()
        for inputs, _ in loader:
            predicted = self.model.forward(**inputs)
            predicted_vectors.append(predicted.data.numpy())
            # targets_vectors.append(_.data.numpy())

        predicted = np.concatenate(predicted_vectors)
        predicted = predicted * self.data_params.y_scale + self.data_params.y_shift

        df['trend'] = self.predict_trend(df)
        cols = ['ds', 'y', 'trend'] #cols to keep from df
        df2 = pd.concat((df[cols],), axis=1)

        # just for debugging - to check if we got all indices right:
        # actual = np.concatenate(targets_vectors)
        # actual = actual * self.data_params.y_scale + self.data_params.y_shift
        # actual = actual[:, forecast_lag-1]
        # df2['actual'] = np.concatenate(([None]*(self.n_lags + forecast_lag - 1),
        #                        actual,
        #                        [None]*(self.n_forecasts - forecast_lag)))

        if n_history is not None and n_history <= self.n_forecasts:
            # create a line for each foreacast
            for i in range(n_history):
                forecast_age = i
                forecast = predicted[-1 -forecast_age, :]
                yhat = np.concatenate(([None] * (self.n_lags + n_history - forecast_age - 1),
                                       forecast,
                                       [None] * forecast_age))
                df2['yhat{}'.format(i + 1)] = yhat
        else:
            # create a line for each forecast_lag
            for i in range(self.n_forecasts):
                forecast_lag = i + 1
                forecast = predicted[:, forecast_lag - 1]
                yhat = np.concatenate(([None] * (self.n_lags + forecast_lag - 1),
                                       forecast,
                                       [None] * (self.n_forecasts - forecast_lag)))
                df2['yhat{}'.format(i+1)] = yhat
        return df2

    def get_last_forecasts(self, n_last_forecasts=1, df=None, future_periods=None, freq='D'):
        return self.predict(df=df, future_periods=future_periods, freq=freq, n_history=n_last_forecasts)

    def predict_trend(self, df):
        """Predict trend using the prophet model.

        Parameters
        ----------
        df: Prediction dataframe.

        Returns
        -------
        Vector with trend on prediction dates.
        """
        if self.growth != 'linear':
            raise NotImplementedError
        t =torch.from_numpy(np.expand_dims(df['t'].values, 1))
        trend = self.model.trend(t).detach().numpy()
        return trend * self.data_params.y_scale + self.data_params.y_shift

    def plot(self, fcst, highlight_forecast=1, ax=None, xlabel='ds', ylabel='y', figsize=(10, 6), crop_last_n=None):
        """Plot the Prophet forecast.

        Parameters
        ----------
        fcst: pd.DataFrame output of self.predict.
        ax: Optional matplotlib axes on which to plot.
        uncertainty: Optional boolean to plot uncertainty intervals.
        plot_cap: Optional boolean indicating if the capacity should be shown
            in the figure, if available.
        xlabel: Optional label name on X-axis
        ylabel: Optional label name on Y-axis
        figsize: Optional tuple width, height in inches.

        Returns
        -------
        A matplotlib figure.
        """
        if crop_last_n is not None:
            fcst = fcst[-crop_last_n:]
        if highlight_forecast > self.n_forecasts:
            highlight_forecast = self.n_forecasts
            print("NOTICE: highlight_forecast > n_forecasts given. "
                "highlight_forecast reduced to n_forecasts")
        return plotting.plot(
            fcst=fcst, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize,
            highlight_forecast=highlight_forecast
        )

    def plot_components(self,
                        fcst,
                        weekly_start=0,
                        yearly_start=0,
                        figsize=None,
                        crop_last_n=None
                        ):
        """Plot the Prophet forecast components.

        Will plot whichever are available of: trend, holidays, weekly
        seasonality, and yearly seasonality.

        Parameters
        ----------
        fcst: pd.DataFrame output of self.predict.
        uncertainty: Optional boolean to plot uncertainty intervals.
        plot_cap: Optional boolean indicating if the capacity should be shown
            in the figure, if available.
        weekly_start: Optional int specifying the start day of the weekly
            seasonality plot. 0 (default) starts the week on Sunday. 1 shifts
            by 1 day to Monday, and so on.
        yearly_start: Optional int specifying the start day of the yearly
            seasonality plot. 0 (default) starts the year on Jan 1. 1 shifts
            by 1 day to Jan 2, and so on.
        figsize: Optional tuple width, height in inches.

        Returns
        -------
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
        )

    def predict_seasonal_components(self, df):
        """Predict seasonality components

        Parameters
        ----------
        df: Prediction dataframe.

        Returns
        -------
        Dataframe with seasonal components. with columns of name <seasonality component name>
        """
        df = check_dataframe(df)
        df = normalize(df, self.data_params)
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

    def fit(self, df):
        loader = self._init_train_loader(df)
        self._train(loader)

    def test(self, df):
        self._evaluate(df)

    def train_eval(self, df, valid_p=0.2, true_ar=None):
        df_train, df_val = split_df(check_dataframe(df), self.n_lags, self.n_forecasts, valid_p, inputs_overbleed=True, verbose=self.verbose)
        train_loader = self._init_train_loader(df)
        results_train = self._train(train_loader)
        results_val = self._evaluate(df_val, true_ar)
        raise NotImplementedError
        # return results


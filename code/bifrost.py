import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from torch import optim
from attrdict import AttrDict
import time

from code.make_dataset import check_dataframe, split_df, init_data_params, normalize, \
    tabularize_univariate_datetime, TimeDataset
from code.model import TimeNet
import code.utils as utils
import code.plotting as plotting


class Bifrost:
    def __init__(self, n_forecasts=1, n_lags=0, n_changepoints=0, num_hidden_layers=0, d_hidden=None,
                 normalize_y=True, ar_sparsity=None, trend_smoothness=0, verbose=False):
        self.name = "ar-net"
        self.verbose = verbose
        self.n_lags = n_lags
        if n_lags == 0 and n_forecasts > 1:
            n_forecasts = 1
            print("NOTICE: changing n_forecasts to 1. Without lags, "
                  "the forecast can be computed for any future time, independent of present values")
        assert n_forecasts >= 1
        self.n_forecasts = n_forecasts
        self.n_changepoints = n_changepoints
        self.normalize_y = normalize_y

        model_complexity =  1 + 10*np.sqrt(n_lags*n_forecasts) + np.log(1 + n_changepoints)
        if verbose: print("model_complexity", model_complexity)
        self.train_config = AttrDict({# TODO allow to be passed in init
            "lr": 1.0 / model_complexity,
            "lr_decay": 0.9,
            "epochs": 40,
            "batch": 16,
            "est_sparsity": ar_sparsity,  # 0 = fully sparse, 1 = not sparse
            "lambda_delay": 10,  # delays start of regularization by lambda_delay epochs
            "reg_lambda_trend": None,
        })
        if self.n_changepoints > 0 and trend_smoothness > 0:
            print("NOTICE: A numeric value greater than 0 for continuous_trend is interpreted as"
                  "the trend changepoint regularization strength. Please note that this might lead to instability."
                  "If training does not converge or becomes NAN, this might be the cause.")
            self.train_config.reg_lambda_trend = 0.1 * trend_smoothness / np.sqrt(self.n_changepoints)
            self.train_config.trend_reg_threshold = 100.0 / (trend_smoothness + self.n_changepoints)

        # self.model = DeepNet(
        #     d_inputs=self.n_lags + self.n_changepoints,
        #     d_outputs=self.n_forecasts,
        #     d_hidden=self.d_hidden,
        #     num_hidden_layers=self.num_hidden_layers,
        # )
        self.model = TimeNet(
            n_forecasts=self.n_forecasts,
            n_lags=self.n_lags,
            n_changepoints=self.n_changepoints,
            trend_smoothness=trend_smoothness,
            num_hidden_layers=num_hidden_layers,
            d_hidden=d_hidden,
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_config.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.train_config.lr_decay)

        # self.history_dates = None
        self.history = None
        self.data_params = None
        self.results = None

        # Prophet Trend related
        self.growth = "linear"
        self.params = AttrDict({
            "trend": AttrDict({"k": 1, "m": 0, "deltas": None, "trend_changepoints_t": None})
        })
        if self.verbose:
            print(self.model)
            # print(self.model.layers[0].weight)

    def _prep_data(self, df):
        if df.shape[0] == 0:
            raise ValueError('Dataframe has no rows.')
        df = check_dataframe(df)
        if self.data_params is None:
            # self.history_dates = pd.to_datetime(df['ds']).sort_values()
            self.history = df.copy(deep=True)
            self.data_params = init_data_params(df, normalize_y=self.normalize_y)
            if self.verbose: print(self.data_params)
        df = normalize(df, self.data_params)

        dataset = TimeDataset(*tabularize_univariate_datetime(
            df, n_lags=self.n_lags, n_forecasts=self.n_forecasts, predict_mode=False, verbose=self.verbose
        ))
        # if self.verbose:
            # plt.plot(df.loc[:100, 'y'])
            # plt.plot(df.loc[:100, 'y_scaled'])
            # plt.show()
        return dataset

    def _prep_data_predict(self, df=None, periods=0, freq='D', n_history=None):
        assert (self.data_params is not None)
        if df is None:
            df = self.history.copy()
        if n_history is not None:
            df = df[-(self.n_lags + n_history - 1):]
        if periods > 0:
            df = self._extend_df_to_future(df, periods=periods, freq=freq)
        df = normalize(df, self.data_params)

        dataset = TimeDataset(*tabularize_univariate_datetime(
            df, n_lags=self.n_lags, n_forecasts=self.n_forecasts, predict_mode=True, verbose=self.verbose
        ))
        return dataset, df

    def _extend_df_to_future(self, df, periods, freq):
        df = check_dataframe(df)
        history_dates = pd.to_datetime(df['ds']).sort_values()
        future_df = utils.make_future_dataframe(history_dates, periods, freq, include_history=False)
        future_df["y"] = None
        df2 = df.append(future_df)
        return df2

    def _train(self, df):
        if self.history is not None: # Note: self.data_params should also be None
            raise Exception('Model object can only be fit once. '
                            'Instantiate a new object.')
        assert (self.data_params is None)

        dataset = self._prep_data(df)
        loader = DataLoader(dataset , batch_size=self.train_config["batch"], shuffle=True)
        results = AttrDict({})
        total_batches = 0
        epoch_losses = []
        epoch_regs = []
        start = time.time()
        for e in range(self.train_config.epochs):
            epoch_loss, epoch_reg, batches = self._train_epoch(e, loader)
            epoch_losses.append(epoch_loss)
            epoch_regs.append(epoch_reg)
            total_batches += batches

        duration = time.time() - start
        results["epoch_losses"] = epoch_losses
        results["epoch_regularizations"] = epoch_regs
        results["loss_train"] = epoch_losses[-1]
        results["time_train"] = duration
        if self.verbose:
            print("Train Time: {:8.4f}".format(duration))
            print("Total Number of Batches: ", total_batches)
        return results

    def _train_epoch(self, e, loader):
        # slowly increase regularization until lambda_delay epoch
        reg_lambda_ar = None
        if self.n_lags > 0:
            reg_lambda_ar = utils.get_regularization_lambda(self.train_config.est_sparsity, self.train_config.lambda_delay, e)

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
        if self.history is None:
            raise Exception('Model object needs to be fit first.')
        assert (self.data_params is not None)

        dataset = self._prep_data(df)
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

    def fit(self, df):
        self._train(df)

    def test(self, df):
        self._evaluate(df)

    def train_eval(self, df, valid_p=0.2, true_ar=None):
        df_train, df_val = split_df(check_dataframe(df), self.n_lags, self.n_forecasts, valid_p, inputs_overbleed=True, verbose=self.verbose)
        results_train = self._train(df_train)
        results_val = self._evaluate(df_val, true_ar)
        raise NotImplementedError
        # return results

    def predict(self, future_periods=None, df=None, freq='D', n_history=None):
        # runs the model  to show predictions
        # if no df is provided, shows predictions over the data history
        # uses the forecast at forecast_lag number to show the fit (if multiple forecasts were made)
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
        return plotting.plot(
            fcst=fcst, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize,
            highlight_forecast=highlight_forecast
        )

    def plot_components(self, fcst,
                        # uncertainty=True, plot_cap=True,
                        # weekly_start=0, yearly_start=0,
                        figsize=None, crop_last_n=None):
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
            fcst=fcst,
            # uncertainty=uncertainty, plot_cap=plot_cap,
            # weekly_start=weekly_start, yearly_start=yearly_start,
            figsize=figsize,
        )





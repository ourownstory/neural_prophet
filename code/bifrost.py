import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from torch import optim
from attrdict import AttrDict
import time

from code.make_dataset import check_dataframe, split_df, init_data_params, normalize, \
    tabularize_univariate_datetime, TimeDataset, make_future_dataframe
from code.model import TimeNet
import code.utils as utils
import code.plotting as plotting


class Bifrost:
    def __init__(self, n_forecasts=1, n_lags=0, n_changepoints=0, num_hidden_layers=0, normalize=True, verbose=False):
        self.name = "ar-net"
        self.verbose = verbose
        self.n_lags = n_lags
        if n_lags == 0:
            if n_forecasts > 1:
                print("changing n_forecasts to 1. Without lags, "
                      "the forecast can be computed for any future time, independent of present values")
            n_forecasts = 1
        self.n_forecasts = n_forecasts
        self.n_changepoints = n_changepoints
        self.normalize = normalize
        # self.num_hidden_layers = num_hidden_layers
        # self.d_hidden = 4 * (n_lags + n_forecasts)
        model_complexity =  1.0 + np.sqrt(n_lags*n_forecasts) + np.log(1 + n_changepoints)
        if verbose: print("model_complexity", model_complexity)
        self.train_config = AttrDict({# TODO allow to be passed in init
            "lr": 0.1 / model_complexity,
            "lr_decay": 0.9,
            "epochs": 50,
            "batch": 16,
            "est_sparsity": 0.1,  # 0 = fully sparse, 1 = not sparse
            "lambda_delay": 10,  # delays start of regularization by lambda_delay epochs
        })
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
            "trend": AttrDict({"k": 1, "m": 0, "deltas": None, "trend_changepoints": None})
        })
        if self.verbose:
            print(self.model)
            # print(self.model.layers[0].weight)

    def _prep_data(self, df):
        df = check_dataframe(df)

        if self.data_params is None:
            self.data_params = init_data_params(df, normalize=self.normalize)
            if self.verbose: print(self.data_params)
        df = normalize(df, self.data_params)

        # self.history_dates = pd.to_datetime(df['ds']).sort_values()
        self.history = df.copy(deep=True)

        dataset = TimeDataset(*tabularize_univariate_datetime(
            df, self.n_lags, self.n_forecasts, self.verbose
        ))
        # if self.verbose:
            # plt.plot(df.loc[:100, 'y'])
            # plt.plot(df.loc[:100, 'y_scaled'])
            # plt.show()
        return dataset

    def _prep_data_predict(self, df=None):
        if df is None:
            df = self.history.copy()
        else:
            if df.shape[0] == 0:
                raise ValueError('Dataframe has no rows.')
            df = check_dataframe(df)
            assert (self.data_params is not None)
            df = normalize(df, self.data_params)

        n_forecasts = 0
        dataset = TimeDataset(*tabularize_univariate_datetime(
            df, self.n_lags, n_forecasts, self.verbose
        ))
        return dataset, df


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
        reg_lambda = utils.get_regularization_lambda(self.train_config.est_sparsity, self.train_config.lambda_delay, e)
        num_batches = 0
        current_epoch_losses = []
        current_epoch_reg_losses = []
        for inputs, targets in loader:
            # Run forward calculation
            predicted = self.model.forward(**inputs)

            # Compute loss.
            loss = self.loss_fn(predicted, targets)
            current_epoch_losses.append(loss.data.item())

            # Add regularization
            reg_loss = torch.zeros(1, dtype=torch.float, requires_grad=True)
            if self.n_lags >0 and reg_lambda is not None:
                # Warning: will grab first layer as the weight to be regularized!
                abs_weights = torch.abs(self.model.ar_weights) #TimeNet
                reg = torch.div(2.0, 1.0 + torch.exp(-3.0 * abs_weights.pow(1.0 / 3.0))) - 1.0
                reg_loss = reg_lambda * (reg_loss + torch.mean(reg))
                loss = loss + reg_loss
            current_epoch_reg_losses.append(reg_loss.data.item())

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
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
        loader = DataLoader(dataset, batch_size=min(1024, len(df)), shuffle=False, drop_last=False)
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
                print("sTPE (weights): {:6.3f}".format(stats["sTPE (weights)"]))
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
        return results

    def predict_history(self, forecast_lag=None, multi_forecast=True):
        # runs the model over the data history to show predictions
        # uses the forecast at forecast_lag number to show the fit (if multiple forecasts were made)
        if self.history is None:
            raise Exception('Model has not been fit.')
        if forecast_lag is not None:
            assert forecast_lag <= self.n_forecasts
        df = self.history.copy()

        results = self._evaluate(df, forecast_lag=forecast_lag)
        if forecast_lag is None: forecast_lag = 1
        predicted = results.predicted * self.data_params.y_scale + self.data_params.y_shift
        forecast = predicted[:, forecast_lag-1]

        df['trend'] = self.predict_trend(df)
        cols = ['ds', 'trend'] #cols to keep from df
        df2 = pd.concat((df[cols],), axis=1)
        df2['yhat'] = np.concatenate(([None]*(self.n_lags + forecast_lag - 1),
                               forecast,
                               [None]*(self.n_forecasts - forecast_lag)))

        # just for debugging - to check if we got all indices right:
        # actual = results.actual * self.data_params.y_scale + self.data_params.y_shift
        # actual = actual[:, forecast_lag-1]
        # df2['actual'] = np.concatenate(([None]*(self.n_lags + forecast_lag - 1),
        #                        actual,
        #                        [None]*(self.n_forecasts - forecast_lag)))

        if multi_forecast:
            for i in range(self.n_forecasts):
                forecast_lag = i + 1
                forecast = predicted[:, forecast_lag - 1]

                yhat = np.concatenate(
                    ([None] * (self.n_lags + forecast_lag - 1),
                     forecast,
                     [None] * (self.n_forecasts - forecast_lag))
                )
                df2['yhat{}'.format(i+1)] = yhat

        return df2


    def predict_future(self, df=None):
        # predicts the next n_forecast steps from the last step in the history
        if self.history is None:
            raise Exception('Model has not been fit.')

        dataset, df = self._prep_data_predict(df)
        inputs, targets = dataset[-1]
        predicted = self.model.forward(inputs)
        predicted = predicted * self.data_params.y_scale + self.data_params.y_shift

        df['trend'] = self.predict_trend(df)

        # TODO: un-normalize and plot with inputs and forecasts marked.

        # TODO decompose components for components plots

        # TODO: adapt from Prophet
        # df['trend'] = self.predict_trend(df)
        # seasonal_components = self.predict_seasonal_components(df)
        # if self.uncertainty_samples:
        #     intervals = self.predict_uncertainty(df)
        # else:
        #     intervals = None
        #
        # # Drop columns except ds, cap, floor, and trend
        # cols = ['ds', 'trend']
        # if 'cap' in df:
        #     cols.append('cap')
        # if self.logistic_floor:
        #     cols.append('floor')
        # # Add in forecast components
        # df2 = pd.concat((df[cols], intervals, seasonal_components), axis=1)
        # df2['yhat'] = (
        #     df2['trend'] * (1 + df2['multiplicative_terms'])
        #     + df2['additive_terms']
        # )
        # return df2

    def predict_trend(self, df):
        """Predict trend using the prophet model.

        Parameters
        ----------
        df: Prediction dataframe.

        Returns
        -------
        Vector with trend on prediction dates.
        """
        k = np.squeeze(self.model.trend_k.detach().numpy())
        m = np.squeeze(self.model.trend_m.detach().numpy())
        t = np.array(df['t'])
        if self.n_changepoints > 0:
            deltas = np.squeeze(self.model.trend_deltas.detach().numpy())
            changepoints = np.squeeze(self.model.trend_changepoints.detach().numpy())
        else:
            deltas = None,
            changepoints = None

        trend = utils.piecewise_linear(t, k, m, deltas=deltas, changepoints_t=changepoints)

        if self.growth != 'linear':
            raise NotImplementedError
            # cap = df['cap_scaled']
            # trend = self.piecewise_logistic(
            #     t, cap, deltas, k, m, self.trend_changepoints)

        return trend * self.data_params.y_scale + self.data_params.y_shift

    def __make_future_dataframe(self, periods, freq='D', include_history=True):
        # This only makes sense if no AR is performed. We will instead go another route
        # We will only predict the next n_forecast steps from the last step in the history
        # directly use self.predict()
        history_dates = pd.to_datetime(self.history['ds']).sort_values()
        return make_future_dataframe(history_dates, periods, freq, include_history)

    def plot(self, fcst, ax=None, xlabel='ds', ylabel='y', figsize=(10, 6), max_history=None):
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
        history = self.history
        if max_history is not None:
            history = history[-max_history:]
            fcst = fcst[-max_history:]
        return plotting.plot(
            history=history, fcst=fcst, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize,
            multi_forecast=self.n_forecasts if self.n_forecasts > 1 else None,
        )

    def plot_components(self, fcst,
                        # uncertainty=True, plot_cap=True,
                        # weekly_start=0, yearly_start=0,
                        figsize=None, max_history=None):
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
        history = self.history
        if max_history is not None:
            # history = history[-max_history:]
            fcst = fcst[-max_history:]
        return plotting.plot_components(
            fcst=fcst,
            # uncertainty=uncertainty, plot_cap=plot_cap,
            # weekly_start=weekly_start, yearly_start=yearly_start,
            figsize=figsize,
        )

def test_1():
    df = pd.read_csv('../data/example_air_passengers.csv')
    df.head()
    # print(df.shape)
    seasonality = 12
    train_frac = 0.8
    train_num = int((train_frac * df.shape[0]) // seasonality * seasonality)
    # print(train_num)
    df_train = df.copy(deep=True).iloc[:train_num]
    df_val = df.copy(deep=True).iloc[train_num:]

    m = Bifrost(
        n_lags=seasonality,
        n_forecasts=1,
        verbose=False,
    )

    m = m.fit(df_train)

    m = m.test(df_val)
    for stat, value in m.results.items():
        print(stat, value)

def test_plotting():
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    m = Bifrost(n_lags=60, n_forecasts=30, verbose=True)
    m.fit(df)
    forecast = m.predict_history()
    m.plot(forecast, max_history=500)
    m.plot_components(forecast)
    plt.show()

def test_changepoints():
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    m = Bifrost(n_lags=0, n_changepoints=100, n_forecasts=0, verbose=True)
    m.fit(df)
    forecast = m.predict_history()
    m.plot(forecast)
    m.plot_components(forecast)
    plt.show()



if __name__ == '__main__':
    # test_1()
    # test_plotting()
    test_changepoints()

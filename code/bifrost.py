import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from attrdict import AttrDict
import time

from code.make_dataset import check_dataframe, split_df, init_data_params, normalize, \
    tabularize_univariate_datetime, TimeDataset, make_future_dataframe
import code.utils as utils


class FlatNet(nn.Module):
    '''
    Linear regression
    '''

    def __init__(self, d_inputs, d_outputs):
        # Perform initialization of the pytorch superclass
        super(FlatNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_inputs, d_outputs),
        )

    def forward(self, x):
        return self.layers(x)



class DeepNet(nn.Module):
    '''
    A simple, general purpose, fully connected network
    '''
    def __init__(self, d_inputs, d_outputs, d_hidden=32, num_hidden_layers=0):
        # Perform initialization of the pytorch superclass
        super(DeepNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(d_inputs, d_hidden, bias=True))
            d_inputs = d_hidden
        self.layers.append(nn.Linear(d_inputs, d_outputs, bias=True))

    def forward(self, x):
        '''
        This method defines the network layering and activation functions
        '''
        activation = F.relu
        for i in range(len(self.layers)):
            if i > 0: x = activation(x)
            x = self.layers[i](x)
        return x


class Bifrost:
    def __init__(self, n_lags=1, n_forecasts=1, n_trend=1, num_hidden_layers=0, normalize=True, verbose=False):
        self.name = "ar-net"
        self.verbose = verbose
        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self.n_trend = n_trend
        self.normalize = normalize
        self.num_hidden_layers = num_hidden_layers
        self.d_hidden = 4 * (n_lags + n_forecasts + n_trend)
        self.train_config = AttrDict({# TODO allow to be passed in init
            "lr": 1e-3,
            "lr_decay": 0.95,
            "epochs": 20,
            "batch": 16,
            "est_sparsity": 0.1,  # 0 = fully sparse, 1 = not sparse
            "lambda_delay": 10,  # delays start of regularization by lambda_delay epochs
        })
        self.model = DeepNet(
            d_inputs=self.n_lags + self.n_trend,
            d_outputs=self.n_forecasts,
            d_hidden=self.d_hidden,
            num_hidden_layers=self.num_hidden_layers,
        )
        if self.verbose:
            print(self.model)
            # print(self.model.layers[0].weight)
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
            "trend": AttrDict({"k": 1, "m": 0, "deltas": None, "changepoints_t": None})
        })

    def _prep_data(self, df):
        df = check_dataframe(df)

        if self.data_params is None:
            self.data_params = init_data_params(df, normalize=self.normalize)
            if self.verbose: print(self.data_params)
        df = normalize(df, self.data_params)

        # self.history_dates = pd.to_datetime(df['ds']).sort_values()
        self.history = df.copy(deep=True)

        inputs, input_names, targets = tabularize_univariate_datetime(
            df, self.n_lags, self.n_forecasts, self.n_trend, self.verbose
        )
        dataset = TimeDataset(inputs, input_names, targets)

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
        inputs, input_names, targets = tabularize_univariate_datetime(
            df, self.n_lags, n_forecasts, self.n_trend, self.verbose
        )
        dataset = TimeDataset(inputs, input_names, targets)

        return dataset, df


    def fit(self, df):
        self._train(df)
        return self

    def test(self, df):
        self._evaluate(df)
        return self

    def train_eval(self, df, valid_p=0.2, true_ar=None):
        df_train, df_val = split_df(check_dataframe(df), self.n_lags, self.n_forecasts, valid_p, inputs_overbleed=True, verbose=self.verbose)

        results_train = self._train(df_train)

        results_val = self._evaluate(df_val, true_ar)

        return results

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
            predicted = self.model.forward(inputs)

            # Compute loss.
            loss = self.loss_fn(predicted, targets)
            current_epoch_losses.append(loss.data.item())

            # Add regularization
            reg_loss = torch.zeros(1, dtype=torch.float, requires_grad=True)
            if reg_lambda is not None:
                # Warning: will grab first layer as the weight to be regularized!
                abs_weights = torch.abs(self.model.layers[0].weight)
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
            predicted = self.model.forward(inputs)
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
            weights_rereversed = self.model.layers[0].weight.detach().numpy()[0, ::-1]
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


    def predict_history(self, forecast_lag=1, multi_forecast=True):
        # runs the model over the data history to show predictions
        # uses the forecast at forecast_lag number to show the fit (if multiple forecasts were made)
        if self.history is None:
            raise Exception('Model has not been fit.')

        df = self.history.copy()
        results = self._evaluate(df, forecast_lag=forecast_lag)
        predicted = results.predicted * self.data_params.y_scale + self.data_params.y_shift

        # if forecast_lag is None:
        #     forecast_lag=1
        assert forecast_lag <= self.n_forecasts

        forecast = predicted[:, forecast_lag-1]

        yhat = np.concatenate(
            ([None]*(self.n_lags + forecast_lag - 1),
             forecast,
             [None]*(self.n_forecasts - forecast_lag))
        )
        df2 = pd.concat((df[['ds']],), axis=1)
        df2['yhat'] = yhat

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
        k = np.nanmean(self.params.trend.k)
        m = np.nanmean(self.params.trend.m)
        deltas = np.nanmean(self.params.trend.delta, axis=0)
        changepoints_t = self.params.trend.changepoints_t

        t = np.array(df['t'])
        if self.growth == 'linear':
            trend = utils.piecewise_linear(t, deltas, k, m, changepoints_t)
        else:
            raise NotImplementedError
            # cap = df['cap_scaled']
            # trend = self.piecewise_logistic(
            #     t, cap, deltas, k, m, self.changepoints_t)

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
        return utils.plot(
            history=history, fcst=fcst, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize,
            multi_forecast=self.n_forecasts if self.n_forecasts > 1 else None,
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
        n_trend=1,
        num_hidden_layers=0,
        normalize=True,
        verbose=False,
    )

    m = m.fit(df_train)

    m = m.test(df_val)
    for stat, value in m.results.items():
        print(stat, value)

def test_2():
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    m = Bifrost(verbose=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=12)
    print(future.tail())
    # forecast = m.predict(future)
    # m.plot(forecast)
    # m.plot_components(forecast)
    plt.show()


def test_plot_history():
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    m = Bifrost(n_lags=1, n_forecasts=1, verbose=True)
    m.fit(df)
    forecast = m.predict_history(forecast_lag=1)
    m.plot(forecast, max_history=50)
    plt.show()

def main():
    # test_1()
    # test_2()
    test_plot_history()



if __name__ == '__main__':
    main()

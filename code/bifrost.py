import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from attrdict import AttrDict
import time

from code.make_dataset import check_dataframe, split_df, normalize, tabularize_univariate_datetime, TimeDataset
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
    def __init__(self, n_lags, n_forecasts, n_trend, num_hidden_layers=0, normalize=True, verbose=False):
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
            "lr_decay": 0.9,
            "epochs": 40,
            "batch": 16,
            "est_sparsity": 1,  # 0 = fully sparse, 1 = not sparse
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
        self.data_params = None

    def prep_data(self, df, fit_params=False):
        if self.verbose:
            plt.plot(np.array(df['y'])[:100])
            plt.show()
        if fit_params: self.data_params = None
        else: assert self.data_params is not None
        df, self.data_params = normalize(df, self.data_params, verbose=self.verbose)
        inputs, input_names, targets = tabularize_univariate_datetime(
            df, self.n_lags, self.n_forecasts, self.n_trend, self.verbose)
        dataset = TimeDataset(inputs, input_names, targets)
        return dataset

    def fit(self, df):
        df = check_dataframe(df)
        dataset_train = self.prep_data(df, fit_params=True)
        self.train(dataset_train)
        return self


    def train_eval(self, df, valid_p=0.2):
        df = check_dataframe(df)
        df_train, df_val = split_df(df, self.n_lags, self.n_forecasts, valid_p, inputs_overbleed=True, verbose=self.verbose)
        dataset_train = self.prep_data(df_train, fit_params=True)
        dataset_val = self.prep_data(df_val)
        start = time.time()
        losses, epoch_losses = self.train(dataset_train)
        self.evaluate(dataset_val)
        duration = time.time() - start

        # TODO: adapt code
        if self.verbose:
            print("Time: {:8.4f}".format(duration))
            print("Final train epoch loss: {:10.2f}".format(epoch_losses[-1]))
            print("Test MSEs: {:10.2f}".format(test_mse))

        results = {}
        results["weights"] = weights
        results["predicted"] = predicted
        results["actual"] = actual
        results["test_mse"] = test_mse
        results["losses"] = losses
        results["epoch_losses"] = epoch_losses
        if data["type"] == 'AR':
            stats = utils.compute_stats_ar(results, ar_params=data["ar"], verbose=verbose)
        else:
            raise NotImplementedError
        stats["Time (s)"] = duration
        return results, stats

    def train(self, dataset):
        loader = DataLoader(dataset, batch_size=self.train_config["batch"], shuffle=True)
        losses = list()
        batch_index = 0
        epoch_losses = []
        avg_losses = []
        lambda_value = utils.intelligent_regularization(self.train_config.est_sparsity)

        for e in range(self.train_config.epochs):
            # slowly increase regularization until lambda_delay epoch
            if self.train_config.lambda_delay is not None and e < self.train_config.lambda_delay:
                l_factor = e / (1.0 * self.train_config.lambda_delay)
                # l_factor = (e / (1.0 * lambda_delay))**2
            else:
                l_factor = 1.0

            for inputs, targets in loader:
                loss = utils.train_batch(model=self.model, x=inputs, y=targets, optimizer=self.optimizer,
                                   loss_fn=self.loss_fn, lambda_value=l_factor * lambda_value)
                epoch_losses.append(loss)
                batch_index += 1
            self.scheduler.step()
            losses.extend(epoch_losses)
            avg_loss = np.mean(epoch_losses)
            avg_losses.append(avg_loss)
            epoch_losses = []
            if self.verbose:
                print("{}. Epoch Avg Loss: {:10.2f}".format(e + 1, avg_loss))
        if self.verbose:
            print("Total Batches: ", batch_index)

        return losses, avg_losses

    def evaluate(self, dataset):
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        # Test and get the resulting predicted y values
        # TODO: adapt code
        losses = list()
        y_vectors = list()
        y_predict_vectors = list()
        batch_index = 0
        for x, y in loader:
            y_predict = self.model.forward(x)
            loss = self.loss_fn(y_predict, y)
            losses.append(loss.data.numpy())
            y_vectors.append(y.data.numpy())
            y_predict_vectors.append(y_predict.data.numpy())
            batch_index += 1

        losses = np.array(losses)
        y_predict_vector = np.concatenate(y_predict_vectors)
        mse = np.mean((y_predict_vector - np.concatenate(y_vectors)) ** 2)

        actual = np.concatenate(np.array(dataset.y_data))
        predicted = np.concatenate(y_predict)
        weights_rereversed = self.model.layers[0].weight.detach().numpy()[0, ::-1]

        pass


    def predict(self, df):
        df = check_dataframe(df)
        pass


import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from code.make_dataset import check_dataframe, normalize, tabularize_univariate_datetime, TimeDataset
from code.v01_training import train as run_train

class NN(nn.Module):
    '''
    A simple, general purpose, fully connected network
    '''

    def __init__(self, d_inputs, d_outputs, hidden_layer_dims=None):
        # Perform initialization of the pytorch superclass
        super().__init__()
        self.activation = F.relu
        self.d_inputs = d_inputs
        self.d_outputs = d_outputs
        self.hidden_layer_dims = [] if hidden_layer_dims is None else hidden_layer_dims

        self.layers = []
        d_in = self.d_inputs
        layer_out_dims = self.hidden_layer_dims + [d_outputs]
        for d_out in layer_out_dims:
            self.layers.append(nn.Linear(d_in, d_out, bias=True))
            d_in = d_out

    def forward(self, x):
        '''
        This method defines the network layering and activation functions
        '''
        x = self.layers[0](x)
        for i in range(len(self.layers) - 1):
            x = self.activation(x)
            x = self.layers[i+1](x)
        return x


class Bifrost:
    def __init__(self, n_lags, n_forecasts, n_trend, normalize=True, verbose=False):
        self.name = "ar-net"
        self.verbose = verbose
        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self.n_trend = n_trend
        self.normalize = normalize
        self.loss_fn = nn.MSELoss()
        self.hidden_layers = []  # TODO allow to be passed in init
        self.train_config = {# TODO allow to be passed in init
            "lr": 2e-4,
            "lr_decay": 0.9,
            "epochs": 10,
            "batch": 16,
            "est_sparsity": 1,  # 0 = fully sparse, 1 = not sparse
            "lambda_delay": 10,  # delays start of regularization by lambda_delay epochs
        }

        self.model = None
        self.data_params = None
        self.dataset_train = None
        self.dataset_val = None

    def prep_data(self, df, fit_params=True, valid_p=None):
        df = check_dataframe(df)
        if self.verbose:
            plt.plot(np.array(df['y'])[:100, 0])
            plt.show()

        split_idx = -1
        if valid_p is not None:
            n_samples = len(df) - self.n_lags + 1 - self.n_forecasts
            n_train = n_samples - int(n_samples * valid_p)
            split_idx = n_train + self.n_lags
            if self.verbose:
                print("{} n_train / {} n_samples".format(n_train, n_samples))

        if fit_params: self.data_params = None
        else: assert self.data_params is not None
        df, self.data_params = normalize(df, self.data_params, split_idx, self.verbose)
        inputs, input_names, targets = tabularize_univariate_datetime(df, self.n_lags, self.n_forecasts, self.n_trend, self.verbose)

        if valid_p is None:
            self.dataset_train = TimeDataset(*inputs, targets)
        else:
            # do split
            self.dataset_train = TimeDataset([x.iloc[:split_idx] for x in inputs], input_names, targets.iloc[:split_idx])
            self.dataset_val = TimeDataset([x.iloc[split_idx:] for x in inputs], input_names, targets.iloc[split_idx:])

    def fit(self, df):
        self.prep_data(df, fit_params=True)
        self.train()
        return self


    def train_eval(self, df, valid_p=0.2):
        self.prep_data(df, fit_params=True, valid_p=valid_p)
        self.train()
        self.evaluate()

    def train(self):
        self.model = NN(
            d_inputs=self.n_lags + self.n_trend,
            d_outputs=self.n_forecasts,
            hidden_layer_dims=self.hidden_layers
        )
        data_loader_train = DataLoader(dataset=self.dataset_train, batch_size=self.train_config["batch"], shuffle=True)

        losses, avg_losses = run_train(
            model=self.model,
            loader=data_loader_train,
            loss_fn=self.loss_fn,
            **self.train_config,
            verbose=self.verbose,
        )

    def evaluate(self):
        if self.model is None: raise BrokenPipeError("Model must be fitted first.")
        data_loader_test = DataLoader(dataset=self.dataset_val, batch_size=len(self.dataset_val), shuffle=False)

        # Test and get the resulting predicted y values
        y_predict, test_losses, test_mse = test(model=model, loader=data_loader_test, loss_fn=loss_fn)
        # TODO: adapt code
        actual = np.concatenate(np.array(dataset_test.y_data))
        predicted = np.concatenate(y_predict)
        weights_rereversed = model.layer_1.weight.detach().numpy()[0, ::-1]
        pass


    def predict(self, df):
        pass


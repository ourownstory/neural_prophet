import os
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from neuralprophet import utils, df_utils


class TimeDataset(Dataset):
    """
    Create a PyTorch dataset of a tabularized time-series

    Arguments: (SAME as returns from tabularize_univariate_datetime)
        inputs: ordered dict of model inputs, each of len(df) but with varying dimensions
        targets: targets to be predicted of same length as each of the model inputs,
                    with dimension n_forecasts
    """
    def __init__(self, inputs, targets=None):
        # these are inputs with lenth of len(dataset), but varying dimensionality
        inputs_dtype = {
            "lags": torch.float,
            "time": torch.float,
            "changepoints": torch.bool,
            "seasonalities": torch.float,
        }
        targets_dtype = torch.float
        self.length = inputs["time"].shape[0]

        self.inputs = OrderedDict({})
        for key, data in inputs.items():
            if key == "seasonalities":
                self.inputs[key] = OrderedDict({})
                for name, period_features in inputs[key].items():
                    self.inputs[key][name] = torch.from_numpy(period_features).type(inputs_dtype[key])
            else:
                self.inputs[key] = torch.from_numpy(data).type(inputs_dtype[key])
        self.targets = torch.from_numpy(targets).type(targets_dtype)

    def __getitem__(self, index):
        # return torch.cat([x[index] for x in self.inputs]), self.targets[index]
        # Future TODO: vectorize
        sample = OrderedDict({})
        for key, data in self.inputs.items():
            if key == "seasonalities":
                sample[key] = OrderedDict({})
                for name, period_features in self.inputs[key].items():
                    sample[key][name] = period_features[index]
            else:
                sample[key] = data[index]
        targets = self.targets[index]
        return sample, targets

    def __len__(self):
        return self.length


def tabularize_univariate_datetime(df,
                                   season_config=None,
                                   n_lags=0,
                                   n_forecasts=1,
                                   predict_mode=False,
                                   verbose=False,
                                   ):
    """
    Create a tabular dataset with ar_order lags for supervised forecasting

    Arguments:
        df: Sequence of observations as a Pandas DataFrame with columns 't' and 'y_scaled'
                Note data must be clean and have no gaps
        n_lags: Number of lag observations as input (X).
        n_forecasts: Number of observations as output (y).
    Returns:
        inputs: ordered dict of model inputs, each of len(df) but with varying dimensions
        targets: targets to be predicted of same length as each of the model inputs,
                    with dimension n_forecasts
    """
    n_samples = len(df) - n_lags + 1 - n_forecasts
    series = df.loc[:, 'y_scaled'].values
    # data is stored in OrderedDict
    inputs = OrderedDict({})

    def _stride_time_features_for_forecasts(x):
        # only for case where n_lags > 0
        return np.array([x[n_lags + i: n_lags + i + n_forecasts] for i in range(n_samples)])

    # time is the time at each forecast step
    t = df.loc[:, 't'].values
    if n_lags == 0:
        assert n_forecasts == 1
        time = np.expand_dims(t, 1)
    else:
        time = _stride_time_features_for_forecasts(t)
    inputs["time"] = time

    if n_lags > 0:
        lags = np.array([series[i: i + n_lags] for i in range(n_samples)])
        inputs["lags"] = lags

    if season_config is not None:
        seasonalities = utils.seasonal_features_from_dates(df['ds'], season_config)
        for name, features in seasonalities.items():
            if n_lags == 0:
                seasonalities[name] = np.expand_dims(features, axis=1)
            else:
                # stride into num_forecast at dim=1 for each sample, just like we did with time
                seasonalities[name] = _stride_time_features_for_forecasts(features)
        inputs["seasonalities"] = seasonalities

    if predict_mode:
        # targets = np.empty((time.shape[0], 1))
        targets = np.empty_like(time)

    else:
        targets = [series[i + n_lags: i + n_lags + n_forecasts] for i in range(n_samples)]
        targets = np.array(targets)

    if verbose:
        print("Tabularized inputs:")
        for key, value in inputs.items():
            if key == "seasonalities":
                for name, period_features in value.items():
                    print(name, period_features.shape)
            else:
                print(key, "shape: ", value.shape)
    return inputs, targets

def test(verbose=True):
    data_path = os.path.join(os.getcwd(), 'data')
    # data_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
    data_name = 'example_air_passengers.csv'

    ## manually load any file that stores a time series, for example:
    df_in = pd.read_csv(os.path.join(data_path, data_name), index_col=False)
    if verbose:
        print(df_in.shape)

    n_lags = 3
    n_forecasts = 1
    valid_p = 0.2
    df_train, df_val = split_df(df_in, n_lags, n_forecasts, valid_p, inputs_overbleed=True, verbose=verbose)

    ## create a tabularized dataset from time series
    df = check_dataframe(df_train)
    data_params = init_data_params(df)
    df = df_utils.normalize(df, data_params)
    inputs, targets = tabularize_univariate_datetime(
        df,
        n_lags=n_lags,
        n_forecasts=n_forecasts,
        verbose=verbose,
    )
    if verbose:
        print("tabularized inputs")
        for inp, values in inputs.items():
            print(inp, values.shape)
        print("targets", targets.shape)


if __name__ == '__main__':
    test()



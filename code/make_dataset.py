import pandas as pd
import numpy as np
import os
from torch.utils.data.dataset import Dataset
import torch


class TimeDataset(Dataset):
    def __init__(self, inputs, input_names, targets):
        inputs_dtype = {
            "lags": torch.FloatTensor,
            "trend": torch.FloatTensor,
        }
        targets_dtype = torch.FloatTensor
        self.length = targets.shape[0]
        self.inputs = [torch.from_numpy(data).type(inputs_dtype[key])
                       for key, data in zip(input_names, inputs)]
        self.targets = torch.from_numpy(targets).type(targets_dtype)

    def __getitem__(self, index):
        return torch.cat([x[index] for x in self.inputs]), self.targets[index]

    def __len__(self):
        return self.length


def split_df(df, n_lags, n_forecasts, valid_p=0.2, inputs_overbleed=True, verbose=False):
    n_samples = len(df) - n_lags + 1 - n_forecasts
    n_train = n_samples - int(n_samples * valid_p)
    if verbose: print("{} n_train / {} n_samples".format(n_train, n_samples))
    split_idx_train = n_train + n_lags
    split_idx_val = split_idx_train - n_lags if inputs_overbleed else split_idx_train
    df_train = df.copy(deep=True).iloc[:split_idx_train].reset_index(drop=True)
    df_val = df.copy(deep=True).iloc[split_idx_val:].reset_index(drop=True)
    return df_train, df_val


def normalize(df, data_params=None, split_idx=-1, verbose=False):
    if data_params is None:
        data_params = {}
        data_params["t_start"] = np.min(df['ds'].iloc[:split_idx])
        data_params["t_scale"] = np.max(df['ds'].iloc[:split_idx]) - data_params["t_start"]
        data_params["y_shift"] = np.mean(df['y'].iloc[:split_idx].values)
        data_params["y_scale"] = np.std(df['y'].iloc[:split_idx].values)
    if verbose: print(data_params)

    df.loc[:, 'ds'] = (df.loc[:, 'ds'] - data_params["t_start"]) / data_params["t_scale"]
    df.loc[:, 'y'] = (df.loc[:, 'y'] - data_params["y_shift"]) / data_params["y_scale"]

    return df, data_params


def tabularize_univariate_datetime(df, n_lags, n_forecasts=1, n_trend=1, verbose=False):
    """
    Create a tabular dataset with ar_order lags for supervised forecasting
        Adds a time index and scales y. Creates auxiliary columns 't',
        'y_scaled'. These columns are used during both fitting and predicting.
    Arguments:
        series: Sequence of observations as a Pandas DataFrame with columns 'ds' and 'y'
                Note data must be clean and have no gaps
        n_lags: Number of lag observations as input (X).
        n_forecasts: Number of observations as output (y).
    Returns:
        df: Pandas DataFrame  of input lags and forecast values (as nested lists)
            shape (n_samples, 2).
            Cols: "x": list(n_lags)
            Cols: "y": list(n_forecasts)
    """
    n_samples = len(df) - n_lags + 1 - n_forecasts

    time = df.loc[:, 'ds'].iloc[n_lags-1:-n_forecasts].values
    # time = pd.DataFrame(time)
    time = np.expand_dims(time, axis=1)

    # lags = pd.DataFrame(
    #     [df.loc[:, 'y'].iloc[i: i + n_lags].values for i in range(n_samples)]
    # )
    # targets = pd.DataFrame(
    #     [df.loc[:, 'y'].iloc[i + n_lags: i + n_lags + n_forecasts].values for i in range(n_samples)]
    # )
    series = df.loc[:, 'y'].values
    lags = np.array([series[i: i + n_lags] for i in range(n_samples)])
    targets = np.array([series[i + n_lags: i + n_lags + n_forecasts] for i in range(n_samples)])
    if verbose:
        print("time_idx.shape", time.shape)
        print("input.shape", lags.shape)
        print("target.shape", targets.shape)
    # df = pd.concat([time, lags, targets], axis=1)
    # df.columns = ["t"] + ["input_{}".format(num) for num in list(range(len(lags.columns)))] + \
    #              ["target_{}".format(num) for num in list(range(len(targets.columns)))]
    # return df
    inputs = [lags]
    input_names = ["lags"]
    if n_trend == 1:
        inputs += [time]
        input_names += ["trend"]
    elif n_trend > 1:
        raise NotImplementedError
    return inputs, input_names, targets


def check_dataframe(df):
    """Prepare dataframe for fitting or predicting.
    ----------
    df: pd.DataFrame with columns ds, y.
    Returns
    -------
    pd.DataFrame prepared for fitting or predicting.
    """
    # TODO: check that no gaps exist

    # prophet based
    if ('ds' not in df) or ('y' not in df):
        raise ValueError(
            'Dataframe must have columns "ds" and "y" with the dates and '
            'values respectively.'
        )
    # check y column
    if df['y'].isnull().any():
        raise ValueError('Dataframe contains NaN values in y.')
    df.loc[:, 'y'] = pd.to_numeric(df['y'])
    if np.isinf(df.loc[:, 'y'].values).any():
        raise ValueError('Found infinity in column y.')
    # check ds column
    if df['ds'].dtype == np.int64:
        df.loc[:, 'ds'] = df.loc[:, 'ds'].astype(str)
    df.loc[:, 'ds'] = pd.to_datetime(df.loc[:, 'ds'])
    if df['ds'].dt.tz is not None:
        raise ValueError(
            'Column ds has timezone specified, which is not supported. '
            'Remove timezone.'
        )
    if df.loc[:, 'ds'].isnull().any():
        raise ValueError('Found NaN in column ds.')

    if df.index.name == 'ds':
        df.index.name = None
    df = df.sort_values('ds')
    df = df.reset_index(drop=True)

    return df


def main():
    verbose = True
    data_path = os.path.join(os.getcwd(), 'data')
    # data_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
    data_name = 'example_air_passengers.csv'

    ## manually load any file that stores a time series, for example:
    df_in = pd.read_csv(os.path.join(data_path, data_name), index_col=False)

    n_lags = 3
    n_forecasts = 1
    valid_p = 0.2

    if verbose:
        print(df_in.shape)

    ## create a tabularized dataset from time series
    df_in = check_dataframe(df_in)
    df_in, data_params = normalize(df_in, verbose=verbose)
    df = tabularize_univariate_datetime(
        df_in,
        n_lags=n_lags,
        n_forecasts=n_forecasts,
        verbose=verbose,
    )

    if verbose:
        print("tabularized df")
        print(df.shape)
        print(df.head())


if __name__ == '__main__':
    main()



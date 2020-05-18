import pandas as pd
import numpy as np
import os
from torch.utils.data.dataset import Dataset
import torch
from attrdict import AttrDict


class TimeDataset(Dataset):
    def __init__(self, inputs, input_names, targets):
        inputs_dtype = {
            "lags": torch.FloatTensor,
            "trend": torch.FloatTensor,
        }
        targets_dtype = torch.FloatTensor
        self.length = inputs[0].shape[0]
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


def init_data_params(df, normalize=True, split_idx=-1):
    data_params = AttrDict({})
    data_params.t_start = df['ds'].min()
    data_params.t_scale = df['ds'].max() - data_params.t_start
    # data_params.t_start = np.min(df['ds'].iloc[:split_idx])
    # data_params.t_scale = np.max(df['ds'].iloc[:split_idx]) - data_params.t_start

    # Note: unlike Prophet, we do a z normalization,
    # Prophet does shift by min and scale by max.
    # if 'y' in df:
    data_params.y_shift = np.mean(df['y'].values) if normalize else 0.0
    data_params.y_scale = np.std(df['y'].values) if normalize else 1.0
    # data_params.y_shift = np.mean(df['y'].iloc[:split_idx].values) if normalize else 0.0
    # data_params.y_scale = np.std(df['y'].iloc[:split_idx].values) if normalize else 1.0

    # Future TODO: logistic/limited growth?
    # if self.growth == 'logistic' and 'floor' in df:
    #     self.logistic_floor = True
    #     floor = df['floor']
    # else:
    #     floor = 0.
    # self.y_scale = (df['y'] - floor).abs().max()
    # if self.y_scale == 0:
    #     self.y_scale = 1

    # Future TODO: extra regressors
    # for name, props in self.extra_regressors.items():
    #     standardize = props['standardize']
    #     n_vals = len(df[name].unique())
    #     if n_vals < 2:
    #         standardize = False
    #     if standardize == 'auto':
    #         if set(df[name].unique()) == set([1, 0]):
    #             standardize = False  # Don't standardize binary variables.
    #         else:
    #             standardize = True
    #     if standardize:
    #         mu = df[name].mean()
    #         std = df[name].std()
    #         self.extra_regressors[name]['mu'] = mu
    #         self.extra_regressors[name]['std'] = std

    return data_params


def normalize(df, data_params):
    # TODO: adopt Prophet code
    """Initialize model scales.

    Sets model scaling factors using df.

    Parameters
    ----------
    initialize_scales: Boolean set the scales or not.
    df: pd.DataFrame for setting scales.
    """
    # Future TODO: logistic/limited growth?
    # if self.logistic_floor:
    #     if 'floor' not in df:
    #         raise ValueError('Expected column "floor".')
    # else:
    #     df['floor'] = 0
    # if self.growth == 'logistic':
    #     if 'cap' not in df:
    #         raise ValueError(
    #             'Capacities must be supplied for logistic growth in '
    #             'column "cap"'
    #         )
    #     if (df['cap'] <= df['floor']).any():
    #         raise ValueError(
    #             'cap must be greater than floor (which defaults to 0).'
    #         )
    #     df['cap_scaled'] = (df['cap'] - df['floor']) / self.y_scale


    # Future TODO: extra regressors
    # for name, props in self.extra_regressors.items():
    #     df[name] = ((df[name] - props['mu']) / props['std'])

    df['t'] = (df['ds'] - data_params.t_start) / data_params.t_scale
    # if 'y' in df:
    df['y_scaled'] = (df['y'].values - data_params.y_shift) / data_params.y_scale

    return df


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

    time = df.loc[:, 't'].iloc[n_lags-1:-n_forecasts].values
    # time = pd.DataFrame(time)
    time = np.expand_dims(time, axis=1)

    # lags = pd.DataFrame(
    #     [df.loc[:, 'y'].iloc[i: i + n_lags].values for i in range(n_samples)]
    # )
    # targets = pd.DataFrame(
    #     [df.loc[:, 'y'].iloc[i + n_lags: i + n_lags + n_forecasts].values for i in range(n_samples)]
    # )
    series = df.loc[:, 'y_scaled'].values
    lags = np.array([series[i: i + n_lags] for i in range(n_samples)])
    if n_forecasts > 0:
        targets = [series[i + n_lags: i + n_lags + n_forecasts] for i in range(n_samples)]
    else:
        targets = [[None] * n_samples]
    targets = np.array(targets)
    # if verbose:
    #     print("time_idx.shape", time.shape)
    #     print("input.shape", lags.shape)
    #     print("target.shape", targets.shape)

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
    # TODO: Future: handle mising
    # prophet based
    if ('ds' not in df) or ('y' not in df):
        raise ValueError(
            'Dataframe must have columns "ds" and "y" with the dates and '
            'values respectively.'
        )

    # check y column: soft
    history = df[df['y'].notnull()].copy()
    if history.shape[0] < 2:
        raise ValueError('Dataframe has less than 2 non-NaN rows.')
    # check y column: hard
    if df['y'].isnull().any():
        raise ValueError('Dataframe contains NaN values in y.')
    df.loc[:, 'y'] = pd.to_numeric(df['y'])
    if np.isinf(df.loc[:, 'y'].values).any():
        raise ValueError('Found infinity in column y.')

    # check ds column
    if df['ds'].isnull().any():
        raise ValueError('Found NaN in column ds.')
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

    # TODO: adopt Prophet code for extra regressors and seasonality
    # for name in self.extra_regressors:
    #     if name not in df:
    #         raise ValueError(
    #             'Regressor {name!r} missing from dataframe'
    #             .format(name=name)
    #         )
    #     df[name] = pd.to_numeric(df[name])
    #     if df[name].isnull().any():
    #         raise ValueError(
    #             'Found NaN in column {name!r}'.format(name=name)
    #         )
    # for props in self.seasonalities.values():
    #     condition_name = props['condition_name']
    #     if condition_name is not None:
    #         if condition_name not in df:
    #             raise ValueError(
    #                 'Condition {condition_name!r} missing from dataframe'
    #                 .format(condition_name=condition_name)
    #             )
    #         if not df[condition_name].isin([True, False]).all():
    #             raise ValueError(
    #                 'Found non-boolean in column {condition_name!r}'
    #                 .format(condition_name=condition_name)
    #             )
    #         df[condition_name] = df[condition_name].astype('bool')

    if df.index.name == 'ds':
        df.index.name = None
    df = df.sort_values('ds')
    df = df.reset_index(drop=True)

    return df


def make_future_dataframe(history_dates, periods, freq='D', include_history=True):
    """Simulate the trend using the extrapolated generative model.

    Parameters
    ----------
    periods: Int number of periods to forecast forward.
    freq: Any valid frequency for pd.date_range, such as 'D' or 'M'.
    include_history: Boolean to include the historical dates in the data
        frame for predictions.

    Returns
    -------
    pd.Dataframe that extends forward from the end of self.history for the
    requested number of periods.
    """
    if history_dates is None:
        raise Exception('Model has not been fit.')
    last_date = history_dates.max()
    dates = pd.date_range(
        start=last_date,
        periods=periods + 1,  # An extra in case we include start
        freq=freq)
    dates = dates[dates > last_date]  # Drop start if equals last_date
    dates = dates[:periods]  # Return correct number of periods

    if include_history:
        dates = np.concatenate((np.array(history_dates), dates))

    return pd.DataFrame({'ds': dates})

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



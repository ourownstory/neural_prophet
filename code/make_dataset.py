import pandas as pd
import numpy as np
import os
from torch.utils.data.dataset import Dataset
import torch
from attrdict import AttrDict
from collections import OrderedDict
from itertools import chain

import code.utils as utils


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


def init_data_params(df, normalize_y=True, split_idx=None, verbose=False):
    """Initialize data scales.

    Sets data scaling factors using df.

    Arguments:
        df: pd.DataFrame to compute normalization parameters from.
        normalize_y: Boolean whether to scale the time series 'y'
        split_idx: if supplied, params are only computed with data up to this point
    Returns:
        data_params: AttrDict of scaling values (t_start, t_scale, [y_shift, y_scale])
    """
    data_params = AttrDict({})
    if split_idx is None:
        # default case, use full dataset
        data_params.t_start = df['ds'].min()
        data_params.t_scale = df['ds'].max() - data_params.t_start
        # Note: unlike Prophet, we do a z normalization,
        # Prophet does shift by min and scale by max.
        if 'y' in df:
            data_params.y_shift = np.mean(df['y'].values) if normalize_y else 0.0
            data_params.y_scale = np.std(df['y'].values) if normalize_y else 1.0
    else:
        # currently never called
        data_params.t_start = np.min(df['ds'].iloc[:split_idx])
        data_params.t_scale = np.max(df['ds'].iloc[:split_idx]) - data_params.t_start
        if 'y' in df:
            data_params.y_shift = np.mean(df['y'].iloc[:split_idx].values) if normalize_y else 0.0
            data_params.y_scale = np.std(df['y'].iloc[:split_idx].values) if normalize_y else 1.0

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
    if verbose: print(data_params)
    return data_params


def normalize(df, data_params):
    # TODO: adopt Prophet code
    """Apply data scales.

    Applies data scaling factors to df using data_params.

    Arguments:
        data_params: AttrDict  of scaling values (t_start, t_scale, [y_shift, y_scale],
                as returned by init_data_params
        df: pd.DataFrame
    Returns:
        df: pd.DataFrame
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

    # if self.verbose:
        # plt.plot(df.loc[:100, 'y'])
        # plt.plot(df.loc[:100, 'y_scaled'])
        # plt.show()
    return df


def check_dataframe(df):
    """Prepare dataframe for fitting or predicting.
    Only performs basic data sanity checks and ordering.
    ----------
    df: pd.DataFrame with columns ds, y.
    Returns
    -------
    pd.DataFrame prepared for fitting or predicting.
    """
    # TODO: Future: handle mising
    # prophet based
    if df.shape[0] == 0:
        raise ValueError('Dataframe has no rows.')
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

    ## TODO: adopt Prophet code for extra regressors
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
    ## Future TODO: allow conditions for seasonality
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


def split_df(df, n_lags, n_forecasts, valid_p=0.2, inputs_overbleed=True, verbose=False):
    n_samples = len(df) - n_lags + 1 - n_forecasts
    n_train = n_samples - int(n_samples * valid_p)
    if verbose: print("{} n_train / {} n_samples".format(n_train, n_samples))
    split_idx_train = n_train + n_lags
    split_idx_val = split_idx_train - n_lags if inputs_overbleed else split_idx_train
    df_train = df.copy(deep=True).iloc[:split_idx_train].reset_index(drop=True)
    df_val = df.copy(deep=True).iloc[split_idx_val:].reset_index(drop=True)
    return df_train, df_val


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
    df = normalize(df, data_params)
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



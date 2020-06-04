from attrdict import AttrDict
import pandas as pd
import numpy as np


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


def extend_df_to_future(df, periods, freq):
    """
    Extends df periods number steps into future.

    Args:
        df (pandas DataFrame): Dataframe with columns 'ds' datestamps and 'y' time series values
        periods (): number of future steps to predict
        freq (): Data step sizes. Frequency of data recording, [ 'Y', 'M', 'D', 'h', 'min', 'sec]
            TODO: implement missing

    Returns:
        df2 (pandas DataFrame): input df with 'ds' extended into future, and 'y' set to None
    """
    df = check_dataframe(df)
    history_dates = pd.to_datetime(df['ds']).sort_values()
    future_df = utils.make_future_dataframe(history_dates, periods, freq, include_history=False)
    future_df["y"] = None
    df2 = df.append(future_df)
    return df2

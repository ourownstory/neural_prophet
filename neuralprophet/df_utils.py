from attrdict import AttrDict
import pandas as pd
import numpy as np
import datetime


def init_data_params(df, normalize_y=True, split_idx=None, verbose=False):
    """Initialize data scaling values.

    Args:
        df (pd.DataFrame): Time series to compute normalization parameters from.
        normalize_y (bool): whether to scale the time series 'y'
        split_idx (int): if supplied, params are only computed with data up to this point
        verbose (bool):

    Returns:
        data_params: AttrDict of scaling values (t_start, t_scale, [y_shift, y_scale])
    """

    if df['ds'].dtype == np.int64:
        df.loc[:, 'ds'] = df.loc[:, 'ds'].astype(str)
    df.loc[:, 'ds'] = pd.to_datetime(df.loc[:, 'ds'])

    data_params = AttrDict({})
    # default case, use full dataset
    data_params.t_start = df['ds'].min()
    data_params.t_scale = df['ds'].max() - data_params.t_start
    # Note: unlike Prophet, we do a z normalization,
    # Prophet does shift by min and scale by max.
    if 'y' in df:
        data_params.y_shift = np.mean(df['y'].values) if normalize_y else 0.0
        data_params.y_scale = np.std(df['y'].values) if normalize_y else 1.0

    # TODO: delete?
    # if split_idx is not None:
    #     # currently never called
    #     data_params.t_start = np.min(df['ds'].iloc[:split_idx])
    #     data_params.t_scale = np.max(df['ds'].iloc[:split_idx]) - data_params.t_start
    #     if 'y' in df:
    #         data_params.y_shift = np.mean(df['y'].iloc[:split_idx].values) if normalize_y else 0.0
    #         data_params.y_scale = np.std(df['y'].iloc[:split_idx].values) if normalize_y else 1.0

    # Future TODO: extra regressors
    """
    for name, props in self.extra_regressors.items():
        standardize = props['standardize']
        n_vals = len(df[name].unique())
        if n_vals < 2:
            standardize = False
        if standardize == 'auto':
            if set(df[name].unique()) == set([1, 0]):
                standardize = False  # Don't standardize binary variables.
            else:
                standardize = True
        if standardize:
            mu = df[name].mean()
            std = df[name].std()
            self.extra_regressors[name]['mu'] = mu
            self.extra_regressors[name]['std'] = std
    """

    if verbose: print(data_params)
    return data_params


def normalize(df, data_params):
    """Apply data scales.

    Applies data scaling factors to df using data_params.

    Args:
        df (pd.DataFrame): with columns 'ds', 'y'
        data_params(AttrDict): scaling values,as returned by init_data_params
            (t_start, t_scale, [y_shift, y_scale])
    Returns:
        df: pd.DataFrame, normalized
    """
    # Future TODO: logistic/limited growth?
    """
    if self.logistic_floor:
        if 'floor' not in df:
            raise ValueError('Expected column "floor".')
    else:
        df['floor'] = 0
    if self.growth == 'logistic':
        if 'cap' not in df:
            raise ValueError(
                'Capacities must be supplied for logistic growth in '
                'column "cap"'
            )
        if (df['cap'] <= df['floor']).any():
            raise ValueError(
                'cap must be greater than floor (which defaults to 0).'
            )
        df['cap_scaled'] = (df['cap'] - df['floor']) / self.y_scale
    """

    # Future TODO: extra regressors
    """
    for name, props in self.extra_regressors.items():
        df[name] = ((df[name] - props['mu']) / props['std'])
    """

    df.loc[:, 't'] = (df['ds'] - data_params.t_start) / data_params.t_scale
    if 'y' in df:
        df['y_scaled'] = np.empty_like(df['y'])
        not_na = df['y'].notna()
        df.loc[not_na, 'y_scaled'] = (df.loc[not_na,'y'].values - data_params.y_shift) / data_params.y_scale
    return df


def check_dataframe(df):
    """Performs basic data sanity checks and ordering

    Prepare dataframe for fitting or predicting.
    Note: contains many lines from OG Prophet

    Args:
        df (pd.DataFrame): with columns ds, y.

    Returns:
        pd.DataFrame prepared for fitting or predicting.
    """

    # TODO: Find mysterious error "A value is trying to be set on a copy of a slice from a DataFrame"
    if df.shape[0] == 0:
        raise ValueError('Dataframe has no rows.')
    if ('ds' not in df) or ('y' not in df):
        raise ValueError(
            'Dataframe must have columns "ds" and "y" with the dates and '
            'values respectively.'
        )
    # check y column: soft
    history = df.loc[df.loc[:, 'y'].notnull()].copy()
    if history.shape[0] < 2:
        raise ValueError('Dataframe has less than 2 non-NaN rows.')
    df.loc[:, 'y'] = pd.to_numeric(df.loc[:, 'y'])
    if np.isinf(df.loc[:, 'y'].values).any():
        raise ValueError('Found infinity in column y.')

    # check ds column
    if df.loc[:, 'ds'].isnull().any():
        raise ValueError('Found NaN in column ds.')
    if df['ds'].dtype == np.int64:
        df.loc[:, 'ds'] = df.loc[:, 'ds'].astype(str)
    df.loc[:, 'ds'] = pd.to_datetime(df.loc[:, 'ds'])
    if df['ds'].dt.tz is not None:
        raise ValueError('Column ds has timezone specified, which is not supported. Remove timezone.')

    if df.loc[:, 'ds'].isnull().any():
        raise ValueError('Found NaN in column ds.')

    ## TODO: extra regressors
    """
    for name in self.extra_regressors:
        if name not in df:
            raise ValueError(
                'Regressor {name!r} missing from dataframe'
                .format(name=name)
            )
        df[name] = pd.to_numeric(df[name])
        if df[name].isnull().any():
            raise ValueError(
                'Found NaN in column {name!r}'.format(name=name)
            )    
    """

    if df.index.name == 'ds':
        df.index.name = None
    df = df.sort_values('ds')
    df = df.reset_index(drop=True)
    return df


def split_df(df, n_lags, n_forecasts, valid_p=0.2, inputs_overbleed=True, verbose=False):
    """Splits timeseries df into train and validation sets.

    Args:
        df (pd.DataFrame): data
        n_lags (int): identical to NeuralProhet
        n_forecasts (int): identical to NeuralProhet
        valid_p (float): fraction of data to use for holdout validation set
        inputs_overbleed (bool): Whether to allow last training targets to be first validation inputs (never targets)
        verbose (bool):

    Returns:
        df_train (pd.DataFrame):  training data
        df_val (pd.DataFrame): validation data
    """
    n_samples = len(df) - n_lags + 2 - 2*n_forecasts
    n_samples = n_samples if inputs_overbleed else n_samples - n_lags
    n_train = n_samples - int(n_samples * valid_p)

    split_idx_train = n_train + n_lags + n_forecasts - 1
    split_idx_val = split_idx_train - n_lags if inputs_overbleed else split_idx_train
    df_train = df.copy(deep=True).iloc[:split_idx_train].reset_index(drop=True)
    df_val = df.copy(deep=True).iloc[split_idx_val:].reset_index(drop=True)
    if verbose: print("{} n_train\n{} n_eval".format(n_train, n_samples - n_train))
    # if verbose: print("{} len train\n{} len eval".format(len(df_train), len(df_val)))
    # if verbose: print(df_train[-5:])
    # if verbose: print(df_val[:5])
    return df_train, df_val


def make_future_df(df, periods, freq):
    """Extends df periods number steps into future.

    Args:
        df (pandas DataFrame): Dataframe with columns 'ds' datestamps and 'y' time series values
        periods (int): number of future steps to predict
        freq (str): Data step sizes. Frequency of data recording,
            Any valid frequency for pd.date_range, such as 'D' or 'M'

    Returns:
        df2 (pd.DataFrame): input df with 'ds' extended into future, and 'y' set to None
    """
    df = check_dataframe(df.copy(deep=True))
    history_dates = pd.to_datetime(df['ds']).sort_values()

    # Note: Identical to OG Prophet:
    last_date = history_dates.max()
    future_dates = pd.date_range(
        start=last_date,
        periods=periods + 1,  # An extra in case we include start
        freq=freq)
    future_dates = future_dates[future_dates > last_date]  # Drop start if equals last_date
    future_dates = future_dates[:periods]  # Return correct number of periods
    future_df = pd.DataFrame({'ds': future_dates})
    future_df["y"] = None
    # future_df["y"] = np.empty(len(future_dates), dtype=float)
    future_df.reset_index(drop=True, inplace=True)
    return future_df


def add_missing_dates_nan(df, freq='D'):
    """Fills missing datetimes in 'ds', with NaN for 'y'

    Args:
        df (pd.Dataframe): with column 'ds'  datetimes
        freq (str):Data step sizes. Frequency of data recording,
            Any valid frequency for pd.date_range, such as 'D' or 'M'
        verbose (bool):

    Returns:
        dataframe without date-gaps but nan-values
    """
    if df['ds'].dtype == np.int64:
        df.loc[:, 'ds'] = df.loc[:, 'ds'].astype(str)
    df.loc[:, 'ds'] = pd.to_datetime(df.loc[:, 'ds'])

    data_len = len(df)
    r = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq=freq)
    df_all = df.set_index('ds').reindex(r).rename_axis('ds').reset_index()
    num_added = len(df_all) - data_len
    return df_all, num_added


def impute_missing_with_trend(df_all, n_changepoints=5, trend_smoothness=0, freq='D'):
    """Fills missing values with trend.

    Args:
        df_all (pd.Dataframe): with column 'ds'  datetimes and 'y' (including NaN)
        n_changepoints (int): see NeuralProphet
        trend_smoothness (float): see NeuralProphet
        freq (str):  see NeuralProphet
        verbose (bool):

    Returns:
        filled df
    """
    from neuralprophet.neural_prophet import NeuralProphet
    m_trend = NeuralProphet(
        n_forecasts=1,
        n_lags=0,
        n_changepoints=n_changepoints,
        verbose=False,
        trend_smoothness=trend_smoothness,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        data_freq=freq,
    )
    is_na = pd.isna(df_all['y'])
    df = df_all.copy(deep=True)
    df = df.dropna()
    # print(sum(is_na), sum(pd.isna(df['y'])))
    m_trend.fit(df)
    trend = m_trend.predict_trend(df_all, future_periods=0)
    df_all.loc[is_na, 'y'] = trend[is_na]
    return df_all


def fill_small_linear_large_trend(df, limit_linear=5, n_changepoints=5, trend_smoothness=0, freq='D'):
    """Adds missing dates, fills missing values with linear imputation or trend.

    Args:
        df (pd.Dataframe): with column 'ds'  datetimes and 'y' (potentially including NaN)
        limit_linear (int): maximum number of missing values to impute.
            Note: because imputation is done in both directions, this value is effectively doubled.
        n_changepoints (int): see NeuralProphet
        trend_smoothness (float): see NeuralProphet
        freq (str):  see NeuralProphet
        verbose (bool):

    Returns:
        filled df
    """
    # detect missing dates
    df_all, _ = add_missing_dates_nan(df, freq=freq)
    # impute small gaps linearly:
    df_all.loc[:, 'y'] = df_all['y'].interpolate(method='linear', limit=limit_linear, limit_direction='both')
    # fill remaining gaps with trend
    df_all = impute_missing_with_trend(df_all, n_changepoints=n_changepoints, trend_smoothness=trend_smoothness, freq=freq)
    return df_all


def test_impute(verbose=True):
    """Debugging data preprocessing"""
    from matplotlib import pyplot as plt
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')

    df_filled = fill_small_linear_large_trend(df, freq='D', verbose=verbose)

    if verbose:
        df_plot, _ = add_missing_dates_nan(df.copy(deep=True), freq='D', verbose=verbose)
        print("sum(pd.isna(df['y']))", sum(pd.isna(df_plot['y'])))
        df_plot = df_plot.loc[:350]
        fig1 = plt.plot(df_plot['ds'], df_plot['y'], 'b-')
        fig1 = plt.plot(df_plot['ds'], df_plot['y'], 'b.')

        df_plot = df_filled.copy(deep=True)
        print("sum(pd.isna(df_filled['y']))", sum(pd.isna(df_plot['y'])))
        df_plot = df_plot.loc[:350]
        fig2 = plt.plot(df_plot['ds'], df_plot['y'], 'kx')
        plt.show()


if __name__ == '__main__':
    """
    just used for debugging purposes. 
    should implement proper tests at some point in the future.
    (some test methods might already be deprecated)
    """
    test_impute(True)

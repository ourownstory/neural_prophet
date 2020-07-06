from attrdict import AttrDict
from collections import OrderedDict
import pandas as pd
import numpy as np
import datetime


def init_data_params(df, normalize_y=True, covariates_config=None, holidays_config=None, verbose=False):
    """Initialize data scaling values.

    Note: We do a z normalization on the target series 'y',
        unlike OG Prophet, which does shift by min and scale by max.
    Args:
        df (pd.DataFrame): Time series to compute normalization parameters from.
        normalize_y (bool): whether to scale the time series 'y'
        covariates_config (OrderedDict): extra regressors with sub_parameters
            normalize (bool)
        holidays_config (OrderedDict): user specified holidays configs
        verbose (bool):

    Returns:
        data_params (OrderedDict): scaling values
            with AttrDict entries containing 'shift' and 'scale' parameters
    """
    data_params = OrderedDict({})

    if df['ds'].dtype == np.int64:
        df.loc[:, 'ds'] = df.loc[:, 'ds'].astype(str)
    df.loc[:, 'ds'] = pd.to_datetime(df.loc[:, 'ds'])
    data_params['ds'] = AttrDict({})
    data_params['ds'].shift = df['ds'].min()
    data_params['ds'].scale = df['ds'].max() - data_params['ds'].shift

    if 'y' in df:
        data_params['y'] = AttrDict({"shift": 0, "scale": 1})
        if normalize_y:
            data_params['y'].shift = np.mean(df['y'].values)
            data_params['y'].scale = np.std(df['y'].values)

    if covariates_config is not None:
        for covar in covariates_config.keys():
            if covar not in df.columns:
                raise ValueError("Covariate {} not found in DataFrame.".format(covar))
            if covariates_config[covar].normalize == 'auto':
                # unique = set(df[covar].unique())
                # if unique == {1, 0} or unique == {1.0, 0.0} or unique == {-1, 1} or unique == {-1.0, 1.0} or unique == {True, False}:
                if set(df[covar].unique()) in ({True, False}, {1, 0}, {1.0, 0.0}, {-1, 1}, {-1.0, 1.0}):
                    covariates_config[covar].normalize = False  # Don't standardize binary variables.
                else:
                    covariates_config[covar].normalize = True
            data_params[covar] = AttrDict({"shift": 0, "scale": 1})
            if covariates_config[covar].normalize:
                data_params[covar].shift = np.mean(df[covar].values)
                data_params[covar].scale = np.std(df[covar].values)

    if holidays_config is not None:
        for holiday in holidays_config.keys():
            if holiday not in df.columns:
                raise ValueError("Holiday {} not found in DataFrame.".format(holiday))
            data_params[holiday] = AttrDict({"shift": 0, "scale": 1})
    if verbose:
        print("Data Parameters (shift, scale):", [(k, (v.shift, v.scale)) for k, v in data_params.items()])
    return data_params


def normalize(df, data_params):
    """Apply data scales.

    Applies data scaling factors to df using data_params.

    Args:
        df (pd.DataFrame): with columns 'ds', 'y', (and potentially more regressors)
        data_params (OrderedDict): scaling values,as returned by init_data_params
            with AttrDict entries containing 'shift' and 'scale' parameters
    Returns:
        df: pd.DataFrame, normalized
    """
    for name in df.columns:
        if name not in data_params.keys(): raise ValueError('Unexpected column {} in data'.format(name))
        new_name = name
        if name == 'ds': new_name = 't'
        if name == 'y': new_name = 'y_scaled'
        df[new_name] = df[name].sub(data_params[name].shift)
        df[new_name] = df[new_name].div(data_params[name].scale)
    return df


def check_dataframe(df, check_y=True, covariates=None, holidays=None):
    """Performs basic data sanity checks and ordering

    Prepare dataframe for fitting or predicting.
    Args:
        df (pd.DataFrame): with columns ds
        check_y (bool): if df must have series values
            set to True if training or predicting with autoregression
        covariates (list or dict): other column names
        holidays (list or dict): holiday column names

    Returns:
        pd.DataFrame
    """
    if df.shape[0] == 0:
        raise ValueError('Dataframe has no rows.')

    if 'ds' not in df:
        raise ValueError('Dataframe must have columns "ds" with the dates.')
    if df.loc[:, 'ds'].isnull().any():
        raise ValueError('Found NaN in column ds.')
    if df['ds'].dtype == np.int64:
        df.loc[:, 'ds'] = df.loc[:, 'ds'].astype(str)
    df.loc[:, 'ds'] = pd.to_datetime(df.loc[:, 'ds'])
    if df['ds'].dt.tz is not None:
        raise ValueError('Column ds has timezone specified, which is not supported. Remove timezone.')

    columns = []
    if check_y:
        columns.append('y')
    if covariates is not None:
        if type(covariates) is list:
            columns.extend(covariates)
        else:  # treat as dict
            columns.extend(covariates.keys())
    if holidays is not None:
        if type(holidays) is list:
            columns.extend(holidays)
        else:  # treat as dict
            columns.extend(holidays.keys())
    for name in columns:
        if name not in df:
            raise ValueError('Column {name!r} missing from dataframe'.format(name=name))
        if df.loc[df.loc[:, name].notnull()].shape[0] < 1:
            raise ValueError('Dataframe column {name!r} only has NaN rows.'.format(name=name))
        df.loc[:, name] = pd.to_numeric(df.loc[:, name])
        if np.isinf(df.loc[:, name].values).any():
            # raise ValueError('Found infinity in column {name!r}.'.format(name=name))
            df.loc[:, name] = df[name].replace([np.inf, -np.inf], np.nan)
        # if df[name].isnull().any():
        #     raise ValueError('Found NaN in column {name!r}'.format(name=name))
        if df.loc[df.loc[:, name].notnull()].shape[0] < 1:
            raise ValueError('Dataframe column {name!r} only has NaN rows.'.format(name=name))

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
    return df_train, df_val


def make_future_df(df, periods, freq, holidays=None):
    """Extends df periods number steps into future.

    Args:
        df (pandas DataFrame): Dataframe with columns 'ds' datestamps and 'y' time series values
        periods (int): number of future steps to predict
        freq (str): Data step sizes. Frequency of data recording,
            Any valid frequency for pd.date_range, such as 'D' or 'M'
        holidays (OrderedDict): User specified holidays configs

    Returns:
        df2 (pd.DataFrame): input df with 'ds' extended into future, and 'y' set to None
    """
    history_dates = pd.to_datetime(df['ds'].copy(deep=True)).sort_values()

    # Note: Identical to OG Prophet:
    last_date = history_dates.max()
    future_dates = pd.date_range(
        start=last_date,
        periods=periods + 1,  # An extra in case we include start
        freq=freq)
    future_dates = future_dates[future_dates > last_date]  # Drop start if equals last_date
    future_dates = future_dates[:periods]  # Return correct number of periods
    future_df = pd.DataFrame({'ds': future_dates})
    for column in df.columns:
        if holidays is not None and column in holidays.keys():
            future_df[column] = df[column].iloc[-periods: ].values
        elif column != 'ds':
            future_df[column] = None
            # future_df[column] = np.empty(len(future_dates), dtype=float)
    future_df.reset_index(drop=True, inplace=True)
    return future_df


def add_missing_dates_nan(df, freq='D'):
    """Fills missing datetimes in 'ds', with NaN for all other columns

    Args:
        df (pd.Dataframe): with column 'ds'  datetimes
        freq (str):Data step sizes. Frequency of data recording,
            Any valid frequency for pd.date_range, such as 'D' or 'M'

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


def impute_missing_with_trend(df_all, column, n_changepoints=5, trend_smoothness=0, freq='D'):
    """Fills missing values with trend.

    Args:
        df_all (pd.Dataframe): with column 'ds'  datetimes and column (including NaN)
        column (str): name of column to be imputed
        n_changepoints (int): see NeuralProphet
        trend_smoothness (float): see NeuralProphet
        freq (str):  see NeuralProphet

    Returns:
        filled df
    """
    print("WARING: Imputing missing with Trend may lead to instability.")
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
        impute_missing=False,
    )
    is_na = pd.isna(df_all[column])
    df = pd.DataFrame({'ds': df_all['ds'].copy(deep=True), 'y': df_all[column].copy(deep=True), })
    # print(sum(is_na), sum(pd.isna(df['y'])))
    m_trend.fit(df.copy(deep=True).dropna())
    fcst = m_trend.predict(df=df, future_periods=0)
    trend = fcst['trend']
    # trend = m_trend.predict_trend(dates=df_all['ds'].copy(deep=True))
    df_all.loc[is_na, column] = trend[is_na]
    return df_all


def impute_missing_with_rolling_avg(df_all, column, n_changepoints=5, trend_smoothness=0, freq='D'):
    """Fills missing values with trend.

    Args:
        df_all (pd.Dataframe): with column 'ds'  datetimes and column (including NaN)
        column (str): name of column to be imputed
        n_changepoints (int): see NeuralProphet
        trend_smoothness (float): see NeuralProphet
        freq (str):  see NeuralProphet

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
        impute_missing=False,
    )
    is_na = pd.isna(df_all[column])
    df = pd.DataFrame({'ds': df_all['ds'].copy(deep=True), 'y': df_all[column].copy(deep=True), })
    # print(sum(is_na), sum(pd.isna(df['y'])))
    m_trend.fit(df.copy(deep=True).dropna())
    fcst = m_trend.predict(df=df, future_periods=0)
    trend = fcst['trend']
    # trend = m_trend.predict_trend(dates=df_all['ds'].copy(deep=True))
    df_all.loc[is_na, column] = trend[is_na]
    return df_all


def fill_small_linear_large_trend(df, column, allow_missing_dates=False, limit_linear=5, n_changepoints=5, trend_smoothness=0, freq='D'):
    """Adds missing dates, fills missing values with linear imputation or trend.

    Args:
        df (pd.Dataframe): with column 'ds'  datetimes and column (potentially including NaN)
        column (str): column name to be filled in.
        allow_missing_dates (bool): whether to fill in missing dates
        limit_linear (int): maximum number of missing values to impute.
            Note: because imputation is done in both directions, this value is effectively doubled.
        n_changepoints (int): resolution of trend to be filled in
        trend_smoothness (float): see NeuralProphet
        freq (str):  see NeuralProphet

    Returns:
        filled df
    """
    if allow_missing_dates is True:
        df_all = df
    else:
        # detect missing dates
        df_all, _ = add_missing_dates_nan(df, freq=freq)
    # impute small gaps linearly:
    df_all.loc[:, column] = df_all[column].interpolate(method='linear', limit=limit_linear, limit_direction='both')
    # fill remaining gaps with trend
    df_all = impute_missing_with_trend(
        df_all, column=column, n_changepoints=n_changepoints, trend_smoothness=trend_smoothness, freq=freq)
    remaining_na = sum(df_all[column].isnull())
    return df_all, remaining_na


def fill_linear_then_rolling_avg(df, column, allow_missing_dates=False, limit_linear=5, rolling=20, freq='D'):
    """Adds missing dates, fills missing values with linear imputation or trend.

    Args:
        df (pd.Dataframe): with column 'ds'  datetimes and column (potentially including NaN)
        column (str): column name to be filled in.
        allow_missing_dates (bool): whether to fill in missing dates
        limit_linear (int): maximum number of missing values to impute.
            Note: because imputation is done in both directions, this value is effectively doubled.
        rolling (int): maximal number of missing values to impute.
            Note: window width is rolling + 2*limit_linear
        freq (str):  see NeuralProphet

    Returns:
        filled df
    """
    if allow_missing_dates is False:
        df, _ = add_missing_dates_nan(df, freq=freq)
    # impute small gaps linearly:
    df.loc[:, column] = df[column].interpolate(method='linear', limit=limit_linear, limit_direction='both')
    # fill remaining gaps with rolling avg
    is_na = pd.isna(df[column])
    rolling_avg = df[column].rolling(rolling + 2*limit_linear, min_periods=2*limit_linear, center=True).mean()
    df.loc[is_na, column] = rolling_avg[is_na]
    remaining_na = sum(df[column].isnull())
    return df, remaining_na


def test_impute(verbose=True):
    """Debugging data preprocessing"""
    from matplotlib import pyplot as plt
    allow_missing_dates = False

    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    name = 'test'
    df[name] = df['y'].values

    if not allow_missing_dates: df_na, _ = add_missing_dates_nan(df.copy(deep=True), freq='D')
    else: df_na = df.copy(deep=True)
    to_fill = pd.isna(df_na['y'])
    print("sum(to_fill)", sum(to_fill))

    # df_filled = fill_small_linear_large_trend(df.copy(deep=True), column=name, allow_missing_dates=allow_missing_dates)
    df_filled = fill_linear_then_rolling_avg(df.copy(deep=True), column=name, allow_missing_dates=allow_missing_dates)
    print("sum(pd.isna(df_filled[name]))", sum(pd.isna(df_filled[name])))

    if verbose:
        if not allow_missing_dates: df, _ = add_missing_dates_nan(df)
        df = df.loc[200:250]
        fig1 = plt.plot(df['ds'], df[name], 'b-')
        fig1 = plt.plot(df['ds'], df[name], 'b.')

        df_filled = df_filled.loc[200:250]
        # fig2 = plt.plot(df_filled['ds'], df_filled[name], 'kx')
        fig2 = plt.plot(df_filled['ds'][to_fill], df_filled[name][to_fill], 'kx')
        plt.show()


if __name__ == '__main__':
    """
    just used for debugging purposes. 
    should implement proper tests at some point in the future.
    (some test methods might already be deprecated)
    """
    test_impute(True)

from dataclasses import dataclass
from collections import OrderedDict
import pandas as pd
import numpy as np
import logging
import math

log = logging.getLogger("nprophet.df_utils")


@dataclass
class ShiftScale:
    shift: float = 0.0
    scale: float = 1.0


def init_data_params(df, normalize, covariates_config=None, regressor_config=None, events_config=None):
    """Initialize data scaling values.

    Note: We do a z normalization on the target series 'y',
        unlike OG Prophet, which does shift by min and scale by max.
    Args:
        df (pd.DataFrame): Time series to compute normalization parameters from.
        normalize (str): Type of normalization to apply to the time series.
            options: ['soft', 'off', 'minmax, 'standardize']
            default: 'soft' scales minimum to 0.1 and the 90th quantile to 0.9
        covariates_config (OrderedDict): extra regressors with sub_parameters
            normalize (bool)
        regressor_config (OrderedDict): extra regressors (with known future values)
            with sub_parameters normalize (bool)
        events_config (OrderedDict): user specified events configs

    Returns:
        data_params (OrderedDict): scaling values
            with ShiftScale entries containing 'shift' and 'scale' parameters
    """
    data_params = OrderedDict({})

    if df["ds"].dtype == np.int64:
        df.loc[:, "ds"] = df.loc[:, "ds"].astype(str)
    df.loc[:, "ds"] = pd.to_datetime(df.loc[:, "ds"])
    data_params["ds"] = ShiftScale(
        shift=df["ds"].min(),
        scale=df["ds"].max() - df["ds"].min(),
    )

    if "y" in df:
        data_params["y"] = get_normalization_params(
            array=df["y"].values,
            norm_type=normalize,
        )

    if covariates_config is not None:
        for covar in covariates_config.keys():
            if covar not in df.columns:
                raise ValueError("Covariate {} not found in DataFrame.".format(covar))
            data_params[covar] = get_normalization_params(
                array=df[covar].values,
                norm_type=covariates_config[covar].normalize,
            )

    if regressor_config is not None:
        for reg in regressor_config.keys():
            if reg not in df.columns:
                raise ValueError("Regressor {} not found in DataFrame.".format(reg))
            data_params[reg] = get_normalization_params(
                array=df[reg].values,
                norm_type=regressor_config[reg].normalize,
            )
    if events_config is not None:
        for event in events_config.keys():
            if event not in df.columns:
                raise ValueError("Event {} not found in DataFrame.".format(event))
            data_params[event] = ShiftScale()
    log.debug("Data Parameters (shift, scale): {}".format([(k, (v.shift, v.scale)) for k, v in data_params.items()]))
    return data_params


def auto_normalization_setting(array):
    if len(np.unique(array)) < 2:
        log.error("encountered variable with one unique value")
        raise ValueError
    # elif set(series.unique()) in ({True, False}, {1, 0}, {1.0, 0.0}, {-1, 1}, {-1.0, 1.0}):
    elif len(np.unique(array)) == 2:
        return "minmax"  # Don't standardize binary variables.
    else:
        return "soft"  # default setting


def get_normalization_params(array, norm_type):
    if norm_type == "auto":
        norm_type = auto_normalization_setting(array)
    shift = 0.0
    scale = 1.0
    if norm_type == "soft":
        lowest = np.min(array)
        q90 = np.quantile(array, 0.9, interpolation="higher")
        width = q90 - lowest
        if math.isclose(width, 0):
            width = (np.max(array) - lowest) / 1.25
        shift = lowest - 0.125 * width
        scale = 1.25 * width
    elif norm_type == "minmax":
        shift = np.min(array)
        scale = np.max(array) - np.min(array)
    elif norm_type == "standardize":
        shift = np.mean(array)
        scale = np.std(array)
    elif norm_type != "off":
        log.error("Normalization {} not defined.".format(norm_type))
    return ShiftScale(shift, scale)


def normalize(df, data_params):
    """Apply data scales.

    Applies data scaling factors to df using data_params.

    Args:
        df (pd.DataFrame): with columns 'ds', 'y', (and potentially more regressors)
        data_params (OrderedDict): scaling values,as returned by init_data_params
            with ShiftScale entries containing 'shift' and 'scale' parameters
    Returns:
        df: pd.DataFrame, normalized
    """
    for name in df.columns:
        if name not in data_params.keys():
            raise ValueError("Unexpected column {} in data".format(name))
        new_name = name
        if name == "ds":
            new_name = "t"
        if name == "y":
            new_name = "y_scaled"
        df[new_name] = df[name].sub(data_params[name].shift).div(data_params[name].scale)
    return df


def check_dataframe(df, check_y=True, covariates=None, regressors=None, events=None):
    """Performs basic data sanity checks and ordering

    Prepare dataframe for fitting or predicting.
    Args:
        df (pd.DataFrame): with columns ds
        check_y (bool): if df must have series values
            set to True if training or predicting with autoregression
        covariates (list or dict): covariate column names
        regressors (list or dict): regressor column names
        events (list or dict): event column names

    Returns:
        pd.DataFrame
    """
    if df.shape[0] == 0:
        raise ValueError("Dataframe has no rows.")

    if "ds" not in df:
        raise ValueError('Dataframe must have columns "ds" with the dates.')
    if df.loc[:, "ds"].isnull().any():
        raise ValueError("Found NaN in column ds.")
    if df["ds"].dtype == np.int64:
        df.loc[:, "ds"] = df.loc[:, "ds"].astype(str)
    if not np.issubdtype(df["ds"].dtype, np.datetime64):
        df.loc[:, "ds"] = pd.to_datetime(df.loc[:, "ds"])
    if df["ds"].dt.tz is not None:
        raise ValueError("Column ds has timezone specified, which is not supported. Remove timezone.")

    columns = []
    if check_y:
        columns.append("y")
    if covariates is not None:
        if type(covariates) is list:
            columns.extend(covariates)
        else:  # treat as dict
            columns.extend(covariates.keys())
    if regressors is not None:
        if type(regressors) is list:
            columns.extend(regressors)
        else:  # treat as dict
            columns.extend(regressors.keys())
    if events is not None:
        if type(events) is list:
            columns.extend(events)
        else:  # treat as dict
            columns.extend(events.keys())
    for name in columns:
        if name not in df:
            raise ValueError("Column {name!r} missing from dataframe".format(name=name))
        if df.loc[df.loc[:, name].notnull()].shape[0] < 1:
            raise ValueError("Dataframe column {name!r} only has NaN rows.".format(name=name))
        if not np.issubdtype(df[name].dtype, np.number):
            df.loc[:, name] = pd.to_numeric(df.loc[:, name])
        if np.isinf(df.loc[:, name].values).any():
            df.loc[:, name] = df[name].replace([np.inf, -np.inf], np.nan)
        if df.loc[df.loc[:, name].notnull()].shape[0] < 1:
            raise ValueError("Dataframe column {name!r} only has NaN rows.".format(name=name))

    if df.index.name == "ds":
        df.index.name = None
    df = df.sort_values("ds")
    df = df.reset_index(drop=True)
    return df


def split_df(df, n_lags, n_forecasts, valid_p=0.2, inputs_overbleed=True):
    """Splits timeseries df into train and validation sets.

    Args:
        df (pd.DataFrame): data
        n_lags (int): identical to NeuralProhet
        n_forecasts (int): identical to NeuralProhet
        valid_p (float): fraction of data to use for holdout validation set
        inputs_overbleed (bool): Whether to allow last training targets to be first validation inputs (never targets)

    Returns:
        df_train (pd.DataFrame):  training data
        df_val (pd.DataFrame): validation data
    """
    n_samples = len(df) - n_lags + 2 - 2 * n_forecasts
    n_samples = n_samples if inputs_overbleed else n_samples - n_lags
    n_train = n_samples - int(n_samples * valid_p)

    split_idx_train = n_train + n_lags + n_forecasts - 1
    split_idx_val = split_idx_train - n_lags if inputs_overbleed else split_idx_train
    df_train = df.copy(deep=True).iloc[:split_idx_train].reset_index(drop=True)
    df_val = df.copy(deep=True).iloc[split_idx_val:].reset_index(drop=True)
    log.debug("{} n_train, {} n_eval".format(n_train, n_samples - n_train))
    return df_train, df_val


def make_future_df(
    df_columns, last_date, periods, freq, events_config=None, events_df=None, regressor_config=None, regressors_df=None
):
    """Extends df periods number steps into future.

    Args:
        df_columns (pandas DataFrame): Dataframe columns
        last_date: (pandas Datetime): last history date
        periods (int): number of future steps to predict
        freq (str): Data step sizes. Frequency of data recording,
            Any valid frequency for pd.date_range, such as 'D' or 'M'
        events_config (OrderedDict): User specified events configs
        events_df (pd.DataFrame): containing column 'ds' and 'event'
        regressor_config (OrderedDict): configuration for user specified regressors,
        regressors_df (pd.DataFrame): containing column 'ds' and one column for each of the external regressors
    Returns:
        df2 (pd.DataFrame): input df with 'ds' extended into future, and 'y' set to None
    """
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)  # An extra in case we include start
    future_dates = future_dates[future_dates > last_date]  # Drop start if equals last_date
    future_dates = future_dates[:periods]  # Return correct number of periods
    future_df = pd.DataFrame({"ds": future_dates})
    # set the events features
    if events_config is not None:
        future_df = convert_events_to_features(future_df, events_config=events_config, events_df=events_df)
    # set the regressors features
    if regressor_config is not None:
        for regressor in regressors_df:
            # Todo: iterate over regressor_config instead
            future_df[regressor] = regressors_df[regressor]
    for column in df_columns:
        if column not in future_df.columns:
            if column != "t" and column != "y_scaled":
                future_df[column] = None
    future_df.reset_index(drop=True, inplace=True)
    return future_df


def convert_events_to_features(df, events_config, events_df):
    """
    Converts events information into binary features of the df

    Args:
        df (pandas DataFrame): Dataframe with columns 'ds' datestamps and 'y' time series values
        events_config (OrderedDict): User specified events configs
        events_df (pd.DataFrame): containing column 'ds' and 'event'

    Returns:
        df (pd.DataFrame): input df with columns for user_specified features
    """

    for event in events_config.keys():
        event_feature = pd.Series([0.0] * df.shape[0])
        dates = events_df[events_df.event == event].ds
        event_feature[df.ds.isin(dates)] = 1.0
        df[event] = event_feature
    return df


def add_missing_dates_nan(df, freq="D"):
    """Fills missing datetimes in 'ds', with NaN for all other columns

    Args:
        df (pd.Dataframe): with column 'ds'  datetimes
        freq (str):Data step sizes. Frequency of data recording,
            Any valid frequency for pd.date_range, such as 'D' or 'M'

    Returns:
        dataframe without date-gaps but nan-values
    """
    if df["ds"].dtype == np.int64:
        df.loc[:, "ds"] = df.loc[:, "ds"].astype(str)
    df.loc[:, "ds"] = pd.to_datetime(df.loc[:, "ds"])

    data_len = len(df)
    r = pd.date_range(start=df["ds"].min(), end=df["ds"].max(), freq=freq)
    df_all = df.set_index("ds").reindex(r).rename_axis("ds").reset_index()
    num_added = len(df_all) - data_len
    return df_all, num_added


def impute_missing_with_trend(df_all, column, n_changepoints=5, trend_reg=0, freq="D"):
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
    log.error("Imputing missing with Trend may lead to instability.")
    from neuralprophet.forecaster import NeuralProphet

    m_trend = NeuralProphet(
        n_forecasts=1,
        n_lags=0,
        n_changepoints=n_changepoints,
        trend_reg=trend_reg,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        impute_missing=False,
    )
    is_na = pd.isna(df_all[column])
    df = pd.DataFrame(
        {
            "ds": df_all["ds"].copy(deep=True),
            "y": df_all[column].copy(deep=True),
        }
    )
    m_trend.fit(
        df.copy(deep=True).dropna(),
        freq=freq,
    )
    fcst = m_trend.predict(df=df)
    trend = fcst["trend"]
    df_all.loc[is_na, column] = trend[is_na]
    return df_all


def impute_missing_with_rolling_avg(df_all, column, n_changepoints=5, trend_reg=0, freq="D"):
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
    from neuralprophet.forecaster import NeuralProphet

    m_trend = NeuralProphet(
        n_forecasts=1,
        n_lags=0,
        n_changepoints=n_changepoints,
        trend_reg=trend_reg,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        impute_missing=False,
    )
    is_na = pd.isna(df_all[column])
    df = pd.DataFrame(
        {
            "ds": df_all["ds"].copy(deep=True),
            "y": df_all[column].copy(deep=True),
        }
    )
    m_trend.fit(
        df.copy(deep=True).dropna(),
        freq=freq,
    )
    fcst = m_trend.predict(df=df)
    trend = fcst["trend"]
    df_all.loc[is_na, column] = trend[is_na]
    return df_all


def fill_small_linear_large_trend(
    df, column, allow_missing_dates=False, limit_linear=5, n_changepoints=5, trend_reg=0, freq="D"
):
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
    df_all.loc[:, column] = df_all[column].interpolate(method="linear", limit=limit_linear, limit_direction="both")
    # fill remaining gaps with trend
    df_all = impute_missing_with_trend(
        df_all, column=column, n_changepoints=n_changepoints, trend_reg=trend_reg, freq=freq
    )
    remaining_na = sum(df_all[column].isnull())
    return df_all, remaining_na


def fill_linear_then_rolling_avg(df, column, allow_missing_dates=False, limit_linear=5, rolling=20, freq="D"):
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
    df.loc[:, column] = df[column].interpolate(method="linear", limit=limit_linear, limit_direction="both")
    # fill remaining gaps with rolling avg
    is_na = pd.isna(df[column])
    rolling_avg = df[column].rolling(rolling + 2 * limit_linear, min_periods=2 * limit_linear, center=True).mean()
    df.loc[is_na, column] = rolling_avg[is_na]
    remaining_na = sum(df[column].isnull())
    return df, remaining_na

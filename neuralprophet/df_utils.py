from dataclasses import dataclass
from collections import OrderedDict
import pandas as pd
import numpy as np
import logging
import math

log = logging.getLogger("NP.df_utils")


@dataclass
class ShiftScale:
    shift: float = 0.0
    scale: float = 1.0


def copy_list(df_list):
    df_list_copy = [df.copy(deep=True) for df in df_list]
    return df_list_copy


def create_df_list(df):
    if isinstance(df, list):
        df_list = copy_list(df)
    else:
        df_list = [df.copy(deep=True)]
    return df_list


def join_dataframes(df_list):
    """Join list of dataframes preserving the episodes so it can be recovered later.

    Args:
        df_list (list of df (pd.DataFrame): containing column 'ds', 'y' with training data)

    Returns:
        df_joined: Dataframe with concatenated episodes
        episodes: list containing episodes of each timestamp
    """
    cont = 0
    episodes = []
    for i in df_list:
        s = ["Ep" + str(cont)]
        episodes = episodes + s * len(i)
        cont += 1
    df_joined = pd.concat(df_list)
    return df_joined, episodes


def recover_dataframes(df_joined, episodes):
    """Recover list of dataframes accordingly to Episodes.

    Args:
        df_joined (pd.DataFrame): Dataframe concatenated containing column 'ds', 'y' with training data
        episodes: List containing the episodes from each timestamp

    Returns:
        DF: Original dataframe before concatenation
    """
    df_joined.insert(0, "eps", episodes)
    df_list = [x for _, x in df_joined.groupby("eps")]
    df_list = [x.drop(["eps"], axis=1) for x in df_list]
    return df_list


def data_params_definition(df, normalize, covariates_config=None, regressor_config=None, events_config=None):
    """Initialize data scaling values.

    Note: We do a z normalization on the target series 'y',
        unlike OG Prophet, which does shift by min and scale by max.
    Args:
        df (pd.DataFrame): Time series to compute normalization parameters from.
        normalize (str): Type of normalization to apply to the time series.
            options: [ 'off', 'minmax, 'standardize', 'soft', 'soft1']
            default: 'soft', unless the time series is binary, in which case 'minmax' is applied.
                'off' bypasses data normalization
                'minmax' scales the minimum value to 0.0 and the maximum value to 1.0
                'standardize' zero-centers and divides by the standard deviation
                'soft' scales the minimum value to 0.0 and the 95th quantile to 1.0
                'soft1' scales the minimum value to 0.1 and the 90th quantile to 0.9
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
    return data_params


def init_data_params(
    df, normalize, covariates_config=None, regressor_config=None, events_config=None, local_modeling=False
):
    """Initialize data scaling values.

    Note: We do a z normalization on the target series 'y',
        unlike OG Prophet, which does shift by min and scale by max.
    Args:
        df (pd.DataFrame or list of pd.Dataframe): Time series to compute normalization parameters from.
        normalize (str): Type of normalization to apply to the time series.
            options: ['soft', 'off', 'minmax, 'standardize']
            default: 'soft' scales minimum to 0.1 and the 90th quantile to 0.9
        covariates_config (OrderedDict): extra regressors with sub_parameters
            normalize (bool)
        regressor_config (OrderedDict): extra regressors (with known future values)
            with sub_parameters normalize (bool)
        events_config (OrderedDict): user specified events configs
        local_modeling (bool): when set to true each episode from list of dataframes will be considered
        locally (i.e. seasonality, data_params, normalization) - not fully implemented yet.

    Returns:
        data_params (OrderedDict or list of OrderedDict): scaling values
            with ShiftScale entries containing 'shift' and 'scale' parameters
    """

    if isinstance(df, list):
        df_list = copy_list(df)
        if local_modeling:
            # Local Normalization
            data_params = list()
            for df in df_list:
                data_params.append(
                    data_params_definition(df, normalize, covariates_config, regressor_config, events_config)
                )
                log.debug(
                    "Global Modeling - Local Normalization - Data Parameters (shift, scale): {}".format(
                        [(k, (v.shift, v.scale)) for k, v in data_params[-1].items()]
                    )
                )
                log.warning(
                    "Local normalization will be implemented in the future - list of data_params may break the code"
                )
        else:
            # Global Normalization
            df, _ = join_dataframes(df_list)
            data_params = data_params_definition(df, normalize, covariates_config, regressor_config, events_config)
            log.debug(
                "Global Modeling - Global Normalization - Data Parameters (shift, scale): {}".format(
                    [(k, (v.shift, v.scale)) for k, v in data_params.items()]
                )
            )
    else:
        data_params = data_params_definition(df, normalize, covariates_config, regressor_config, events_config)
        log.debug(
            "Data Parameters (shift, scale): {}".format([(k, (v.shift, v.scale)) for k, v in data_params.items()])
        )
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
        q95 = np.quantile(array, 0.95, interpolation="higher")
        width = q95 - lowest
        if math.isclose(width, 0):
            width = np.max(array) - lowest
        shift = lowest
        scale = width
    elif norm_type == "soft1":
        lowest = np.min(array)
        q90 = np.quantile(array, 0.9, interpolation="higher")
        width = q90 - lowest
        if math.isclose(width, 0):
            width = (np.max(array) - lowest) / 1.25
        shift = lowest - 0.125 * width
        scale = 1.25 * width
    elif norm_type == "minmax":
        shift = np.min(array)
        scale = np.max(array) - shift
    elif norm_type == "standardize":
        shift = np.mean(array)
        scale = np.std(array)
    elif norm_type != "off":
        log.error("Normalization {} not defined.".format(norm_type))
    return ShiftScale(shift, scale)


def _normalization(df, data_params):
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


def normalize(df, data_params, local_modeling=False):
    """Apply data scales.

    Applies data scaling factors to df using data_params.

    Args:
        df (pd.DataFrame or list of pd.Dataframe): with columns 'ds', 'y', (and potentially more regressors)
        data_params (OrderedDict): scaling values,as returned by init_data_params
            with ShiftScale entries containing 'shift' and 'scale' parameters
        local_modeling (bool): when set to true each episode from list of dataframes will be considered
        locally (i.e. seasonality, data_params, normalization) - not fully implemented yet.
    Returns:
        df: pd.DataFrame or list of pd.DataFrame, normalized
    """

    if isinstance(df, list):
        df_list = copy_list(df)
        if local_modeling:
            # Local Normalization
            log.warning(
                "Local normalization will be implemented in the future - list of data_params may break the code"
            )
            df_list_norm = list()
            for df, df_data_params in zip(df_list, data_params):
                df_list_norm.append(_normalization(df, df_data_params))
            df = df_list_norm
        else:
            # Global Normalization
            df_joined, episodes = join_dataframes(df_list)
            df = _normalization(df_joined, data_params)
            df = recover_dataframes(df, episodes)
    else:
        df = _normalization(df, data_params)
    return df


def _check_dataframe(df, check_y, covariates, regressors, events):
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

    # FIX Issue #53: Data: fail with specific error message when data contains duplicate date entries.
    if len(df.ds.unique()) != len(df.ds):
        raise ValueError("Column ds has duplicate values. Please remove duplicates.")
    # END FIX

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


def check_dataframe(df, check_y=True, covariates=None, regressors=None, events=None):
    """Performs basic data sanity checks and ordering

    Prepare dataframe for fitting or predicting.
    Args:
        df (pd.DataFrame or list of pd.DataFrame): with columns ds
        check_y (bool): if df must have series values
            set to True if training or predicting with autoregression
        covariates (list or dict): covariate column names
        regressors (list or dict): regressor column names
        events (list or dict): event column names

    Returns:
        pd.DataFrame or list of pd.DataFrame
    """
    df_list = create_df_list(df)
    checked_df = list()
    for df in df_list:
        checked_df.append(_check_dataframe(df, check_y, covariates, regressors, events))
    df = checked_df
    return df[0] if len(df) == 1 else df


def crossvalidation_split_df(df, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct=0.0):
    """Splits data in k folds for crossvalidation.

    Args:
        df (pd.DataFrame): data
        n_lags (int): identical to NeuralProhet
        n_forecasts (int): identical to NeuralProhet
        k (int): number of CV folds
        fold_pct (float): percentage of overall samples to be in each fold
        fold_overlap_pct (float): percentage of overlap between the validation folds.
            default: 0.0

    Returns:
        list of k tuples [(df_train, df_val), ...] where:
            df_train (pd.DataFrame):  training data
            df_val (pd.DataFrame): validation data
    """
    if n_lags == 0:
        assert n_forecasts == 1
    total_samples = len(df) - n_lags + 2 - (2 * n_forecasts)
    samples_fold = max(1, int(fold_pct * total_samples))
    samples_overlap = int(fold_overlap_pct * samples_fold)
    assert samples_overlap < samples_fold
    min_train = total_samples - samples_fold - (k - 1) * (samples_fold - samples_overlap)
    assert min_train >= samples_fold
    folds = []
    df_fold = df.copy(deep=True)
    for i in range(k, 0, -1):
        df_train, df_val = split_df(df_fold, n_lags, n_forecasts, valid_p=samples_fold, inputs_overbleed=True)
        folds.append((df_train, df_val))
        split_idx = len(df_fold) - samples_fold + samples_overlap
        df_fold = df_fold.iloc[:split_idx].reset_index(drop=True)
    folds = folds[::-1]
    return folds


def double_crossvalidation_split_df(df, n_lags, n_forecasts, k, valid_pct, test_pct):
    """Splits data in two sets of k folds for crossvalidation on validation and test data.

    Args:
        df (pd.DataFrame): data
        n_lags (int): identical to NeuralProhet
        n_forecasts (int): identical to NeuralProhet
        k (int): number of CV folds
        valid_pct (float): percentage of overall samples to be in validation
        test_pct (float): percentage of overall samples to be in test

    Returns:
        tuple of folds_val, folds_test, where each are same as crossvalidation_split_df returns
    """
    fold_pct_test = float(test_pct) / k
    folds_test = crossvalidation_split_df(df, n_lags, n_forecasts, k, fold_pct=fold_pct_test, fold_overlap_pct=0.0)
    df_train = folds_test[0][0]
    fold_pct_val = float(valid_pct) / k / (1.0 - test_pct)
    folds_val = crossvalidation_split_df(df_train, n_lags, n_forecasts, k, fold_pct=fold_pct_val, fold_overlap_pct=0.0)
    return folds_val, folds_test


def _split_df(df, n_lags, n_forecasts, valid_p, inputs_overbleed):
    """Splits timeseries df into train and validation sets.

    Prevents overbleed of targets. Overbleed of inputs can be configured. In case of global modeling the split could be either local or global.

    Args:
        df (pd.DataFrame): data
        n_lags (int): identical to NeuralProhet
        n_forecasts (int): identical to NeuralProhet
        valid_p (float, int): fraction (0,1) of data to use for holdout validation set,
            or number of validation samples >1
        inputs_overbleed (bool): Whether to allow last training targets to be first validation inputs (never targets)

    Returns:
        df_train (pd.DataFrame):  training data
        df_val (pd.DataFrame): validation data
    """
    n_samples = len(df) - n_lags + 2 - (2 * n_forecasts)
    n_samples = n_samples if inputs_overbleed else n_samples - n_lags
    if 0.0 < valid_p < 1.0:
        n_valid = max(1, int(n_samples * valid_p))
    else:
        assert valid_p >= 1
        assert type(valid_p) == int
        n_valid = valid_p
    n_train = n_samples - n_valid
    assert n_train >= 1

    split_idx_train = n_train + n_lags + n_forecasts - 1
    split_idx_val = split_idx_train - n_lags if inputs_overbleed else split_idx_train
    df_train = df.copy(deep=True).iloc[:split_idx_train].reset_index(drop=True)
    df_val = df.copy(deep=True).iloc[split_idx_val:].reset_index(drop=True)
    log.debug("{} n_train, {} n_eval".format(n_train, n_samples - n_train))
    return df_train, df_val


def find_time_threshold(df_list, n_lags, valid_p, inputs_overbleed):
    if not 0 < valid_p < 1:
        log.error("Please type a valid value for valid_p (for global modeling it should be between 0 and 1.0)")
    df_joint, _ = join_dataframes(df_list)
    df_joint = df_joint.sort_values("ds")
    df_joint = df_joint.reset_index(drop=True)
    n_samples = len(df_joint)
    n_samples = n_samples if inputs_overbleed else n_samples - n_lags
    n_valid = max(1, int(n_samples * valid_p))
    n_train = n_samples - n_valid
    threshold_time_stamp = df_joint.loc[n_train, "ds"]
    log.debug("Time threshold: ", threshold_time_stamp)
    return threshold_time_stamp


def split_considering_timestamp(df_list, threshold_time_stamp):
    df_train = list()
    df_val = list()
    for df in df_list:
        if df["ds"].max() < threshold_time_stamp:
            df_train.append(df)
        elif df["ds"].min() > threshold_time_stamp:
            df_val.append(df)
        else:
            df_train.append(df[df["ds"] < threshold_time_stamp])
            df_val.append(df[df["ds"] >= threshold_time_stamp])
    return df_train, df_val


def split_df(df, n_lags, n_forecasts, valid_p=0.2, inputs_overbleed=True, local_modeling=False):
    """Splits timeseries df into train and validation sets.

    Prevents overbleed of targets. Overbleed of inputs can be configured. In case of global modeling the split could be either local or global.

    Args:
        df (pd.DataFrame or list of pd.Dataframe): data
        n_lags (int): identical to NeuralProhet
        n_forecasts (int): identical to NeuralProhet
        valid_p (float, int): fraction (0,1) of data to use for holdout validation set,
            or number of validation samples >1
        inputs_overbleed (bool): Whether to allow last training targets to be first validation inputs (never targets)
        local_modeling (bool): when set to true each episode from list of dataframes will be considered
        locally (i.e. seasonality, data_params, normalization) - not fully implemented yet.
    Returns:
        df_train (pd.DataFrame or list of pd.Dataframe):  training data
        df_val (pd.DataFrame or list of pd.Dataframe): validation data
    """
    if isinstance(df, list):
        df_list = copy_list(df)
        df_train_list = list()
        df_val_list = list()
        if local_modeling:
            for df in df_list:
                df_train, df_val = _split_df(df, n_lags, n_forecasts, valid_p, inputs_overbleed)
                df_train_list.append(df_train)
                df_val_list.append(df_val)
            df_train, df_val = df_train_list, df_val_list
        else:
            threshold_time_stamp = find_time_threshold(df_list, n_lags, valid_p, inputs_overbleed)
            df_train, df_val = split_considering_timestamp(df_list, threshold_time_stamp)
    else:
        df_train, df_val = _split_df(df, n_lags, n_forecasts, valid_p, inputs_overbleed)
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


def add_missing_dates_nan(df, freq):
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


def fill_linear_then_rolling_avg(series, limit_linear, rolling):
    """Adds missing dates, fills missing values with linear imputation or trend.

    Args:
        series (pd.Series): series with nan to be filled in.
        limit_linear (int): maximum number of missing values to impute.
            Note: because imputation is done in both directions, this value is effectively doubled.
        rolling (int): maximal number of missing values to impute.
            Note: window width is rolling + 2*limit_linear

    Returns:
        filled df
    """
    # impute small gaps linearly:
    series = series.interpolate(method="linear", limit=limit_linear, limit_direction="both")
    # fill remaining gaps with rolling avg
    is_na = pd.isna(series)
    rolling_avg = series.rolling(rolling + 2 * limit_linear, min_periods=2 * limit_linear, center=True).mean()
    series.loc[is_na] = rolling_avg[is_na]
    remaining_na = sum(series.isnull())
    return series, remaining_na


def make_list_dataframes(df, episodes):
    if df is not None:
        df_list = [df.copy(deep=True)] * episodes
    else:
        df_list = [None] * episodes
    return df_list

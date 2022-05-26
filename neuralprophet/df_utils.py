from collections import OrderedDict
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging
import math


log = logging.getLogger("NP.df_utils")


@dataclass
class ShiftScale:
    shift: float = 0.0
    scale: float = 1.0


def prep_copy_df_dict(df):
    """Creates or copy a df_dict based on the df input.
    It either converts a pd.DataFrame to a dict or copies it in case of a dict input.

    Parameters
    ----------
        df : pd.DataFrame,dict
            containing df or dict with group of dfs

    Returns
    -------
        pd.DataFrames
            dict of dataframes or copy of dict of dataframes
        bool
            whether the input was unnamed
    """
    received_unnamed_df = False
    if isinstance(df, dict):
        df_dict = {key: df_aux.copy(deep=True) for (key, df_aux) in df.items()}
    elif isinstance(df, pd.DataFrame):
        received_unnamed_df = True
        df_dict = {"__df__": df.copy(deep=True)}
    elif df is None:
        return None, None
    else:
        raise ValueError("Please insert valid df type (i.e. pd.DataFrame, dict)")
    return df_dict, received_unnamed_df


def maybe_get_single_df_from_df_dict(df_dict, received_unnamed_df=True):
    """Extract dataframe from single length dict if placeholder-named.

    Parameters
    ----------
        df_dict : dict
            dict with potentially single pd.DataFrame
        received_unnamed_df : bool
            whether the input was unnamed

    Returns
    -------
        pd.Dataframe or dict
            original input format
    """
    if received_unnamed_df and isinstance(df_dict, dict) and len(df_dict) == 1:
        if list(df_dict.keys())[0] == "__df__":
            return df_dict["__df__"]
    else:
        return df_dict


def join_dataframes(df_dict):
    """Join dict of dataframes preserving the episodes so it can be recovered later.

    Parameters
    ----------
        df_dict : dict of pd.DataFrame
            containing column ``ds``, ``y`` with training data

    Returns
    -------
        pd.Dataframe
            Dataframe with concatenated episodes
        list
            keys of each timestamp
    """
    if not isinstance(df_dict, dict):
        raise ValueError("can not join other than dicts of DataFrames.")
    episodes = []
    for key in df_dict:
        episodes = episodes + [key] * len(df_dict[key])
    df_joined = pd.concat(df_dict, ignore_index=True)
    return df_joined, episodes


def get_max_num_lags(config_covar, n_lags):
    """Get the greatest number of lags between the autoregression lags and the covariates lags.

    Parameters
    ----------
        config_covar : OrderedDict
            configuration for covariates
        n_lags : int
            number of lagged values of series to include as model inputs

    Returns
    -------
        int
            Maximum number of lags between the autoregression lags and the covariates lags.
    """
    if config_covar is not None:
        log.debug("config_covar exists")
        max_n_lags = max([n_lags] + [val.n_lags for key, val in config_covar.items()])
    else:
        log.debug("config_covar does not exist")
        max_n_lags = n_lags
    return max_n_lags


def recover_dataframes(df_joined, episodes):
    """Recover dict of dataframes accordingly to Episodes.

    Parameters
    ----------
        df_joined : pd.DataFrame
            Dataframe concatenated containing column ``ds``, ``y`` with training data
        episodes : List
            containing the episodes from each timestamp

    Returns
    -------
        pd.Dataframe
            Original dict before concatenation
    """
    df_joined.insert(0, "eps", episodes)
    df_dict = {key: df for key, df in df_joined.groupby("eps")}
    df_dict = {key: df.drop(["eps"], axis=1) for (key, df) in df_dict.items()}
    return df_dict


def join_dataframes_for_split_df(df_dict):
    """Join dict of dataframes for procedures of splitting considering time stamp.

    Parameters
    ----------
        df_dict : dict of pd.DataFrame
            containing column ``ds``, ``y`` with training data

    Returns
    -------
        pd.Dataframe
            Dataframe with concatenated episodes (sorted 'ds', duplicates removed)
    """
    df_joint, _ = join_dataframes(df_dict)
    df_joint = df_joint.sort_values("ds")
    df_joint = df_joint.drop_duplicates(subset=["ds"])
    df_joint = df_joint.reset_index(drop=True)
    return df_joint


def data_params_definition(df, normalize, covariates_config=None, regressor_config=None, events_config=None):
    """
    Initialize data scaling values.

    Note
    ----
    We do a z normalization on the target series ``y``,
    unlike OG Prophet, which does shift by min and scale by max.

    Parameters
    ----------
    df : pd.DataFrame
        Time series to compute normalization parameters from.
    normalize : str
        Type of normalization to apply to the time series.

            options:

                ``soft`` (default), unless the time series is binary, in which case ``minmax`` is applied.

                ``off`` bypasses data normalization

                ``minmax`` scales the minimum value to 0.0 and the maximum value to 1.0

                ``standardize`` zero-centers and divides by the standard deviation

                ``soft`` scales the minimum value to 0.0 and the 95th quantile to 1.0

                ``soft1`` scales the minimum value to 0.1 and the 90th quantile to 0.9
    covariates_config : OrderedDict
        extra regressors with sub_parameters
    normalize : bool
        data normalization
    regressor_config : OrderedDict
        extra regressors (with known future values) with sub_parameters normalize (bool)
    events_config : OrderedDict
        user specified events configs

    Returns
    -------
    OrderedDict
        scaling values with ShiftScale entries containing ``shift`` and ``scale`` parameters.
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
    df_dict,
    normalize="auto",
    covariates_config=None,
    regressor_config=None,
    events_config=None,
    global_normalization=False,
    global_time_normalization=False,
):
    """Initialize data scaling values.

    Note
    ----
    We compute and store local and global normalization parameters independent of settings.

    Parameters
    ----------
        df : dict
            dict of DataFrames to compute normalization parameters from.
        normalize : str
            Type of normalization to apply to the time series.

                options:

                    ``soft`` (default), unless the time series is binary, in which case ``minmax`` is applied.

                    ``off`` bypasses data normalization

                    ``minmax`` scales the minimum value to 0.0 and the maximum value to 1.0

                    ``standardize`` zero-centers and divides by the standard deviation

                    ``soft`` scales the minimum value to 0.0 and the 95th quantile to 1.0

                    ``soft1`` scales the minimum value to 0.1 and the 90th quantile to 0.9
        covariates_config : OrderedDict
            extra regressors with sub_parameters
        regressor_config : OrderedDict
            extra regressors (with known future values)
        events_config : OrderedDict
            user specified events configs
        global_normalization : bool

            ``True``: sets global modeling training with global normalization

            ``False``: sets global modeling training with local normalization
        global_time_normalization : bool

            ``True``: normalize time globally across all time series

            ``False``: normalize time locally for each time series

            (only valid in case of global modeling - local normalization)

    Returns
    -------
        OrderedDict
            nested dict with data_params for each dataset where each contains
        OrderedDict
            ShiftScale entries containing ``shift`` and ``scale`` parameters for each column
    """
    # Compute Global data params
    df_merged, _ = join_dataframes(prep_copy_df_dict(df_dict)[0])
    global_data_params = data_params_definition(
        df_merged, normalize, covariates_config, regressor_config, events_config
    )
    if global_normalization:
        log.debug(
            "Global Normalization Data Parameters (shift, scale): {}".format(
                [(k, v) for k, v in global_data_params.items()]
            )
        )
    # Compute individual  data params
    local_data_params = OrderedDict()
    for key, df_i in df_dict.items():
        local_data_params[key] = data_params_definition(
            df_i, normalize, covariates_config, regressor_config, events_config
        )
        if global_time_normalization:
            # Overwrite local time normalization data_params with global values (pointer)
            local_data_params[key]["ds"] = global_data_params["ds"]
        if not global_normalization:
            log.debug(
                "Local Normalization Data Parameters (shift, scale): {}".format(
                    [(k, v) for k, v in local_data_params[key].items()]
                )
            )
    return local_data_params, global_data_params


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


def normalize(df, data_params):
    """
    Applies data scaling factors to df using data_params.

    Parameters
    ----------
        df : pd.DataFrame
            with columns ``ds``, ``y``, (and potentially more regressors)
        data_params : OrderedDict
            scaling values, as returned by init_data_params with ShiftScale entries containing ``shift`` and ``scale`` parameters

    Returns
    -------
        pd.DataFrame
            normalized dataframes
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


def check_single_dataframe(df, check_y, covariates, regressors, events):
    """Performs basic data sanity checks and ordering
    as well as prepare dataframe for fitting or predicting.

    Parameters
    ----------
        df : pd.DataFrame
            with columns ds
        check_y : bool
            if df must have series values (``True`` if training or predicting with autoregression)
        covariates : list or dict
            covariate column names
        regressors : list or dict
            regressor column names
        events : list or dict
            event column names

    Returns
    -------
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

    if len(df.ds.unique()) != len(df.ds):
        raise ValueError("Column ds has duplicate values. Please remove duplicates.")

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
    """Performs basic data sanity checks and ordering,
    as well as prepare dataframe for fitting or predicting.

    Parameters
    ----------
        df : pd.DataFrame or dict
            containing column ``ds``
        check_y : bool
            if df must have series values
            set to True if training or predicting with autoregression
        covariates : list or dict
            covariate column names
        regressors : list or dict
            regressor column names
        events : list or dict
            event column names

    Returns
    -------
        pd.DataFrame or dict
            checked dataframe
    """
    if isinstance(df, pd.DataFrame):
        checked_df = check_single_dataframe(df, check_y, covariates, regressors, events)
    elif isinstance(df, dict):
        checked_df = {}
        for key, df_i in df.items():
            checked_df[key] = check_single_dataframe(df_i, check_y, covariates, regressors, events)
    else:
        raise ValueError("Please insert valid df type (i.e. pd.DataFrame, dict)")
    return checked_df


def _crossvalidation_split_df(df, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct=0.0):
    """Splits data in k folds for crossvalidation.

    Parameters
    ----------
        df : pd.DataFrame
            data
        n_lags : int
            identical to NeuralProphet
        n_forecasts : int
            identical to NeuralProphet
        k : int
            number of CV folds
        fold_pct : float
            percentage of overall samples to be in each fold
        fold_overlap_pct : float
            percentage of overlap between the validation folds (default: 0.0)

    Returns
    -------
        list of k tuples [(df_train, df_val), ...]

            training data

            validation data
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


def find_valid_time_interval_for_cv(df_dict):
    """Find time interval of interception among all the time series from dict.

    Parameters
    ----------
        df_dict : dict
            dict of data

    Returns
    -------
        str
            time interval start
        str
            time interval end
    """
    # Creates first time interval based on data from first key
    time_interval_intersection = df_dict[list(df_dict.keys())[0]][["ds"]]
    for key in df_dict:
        time_interval_intersection = pd.merge(time_interval_intersection, df_dict[key], how="inner", on=["ds"])
        time_interval_intersection = time_interval_intersection[["ds"]]
    start_date = time_interval_intersection["ds"].iloc[0]
    end_date = time_interval_intersection["ds"].iloc[-1]
    return start_date, end_date


def unfold_dict_of_folds(folds_dict, k):
    """Convert dict of folds for typical format of folding of train and test data.

    Parameters
    ----------
        folds_dict : dict
            dict of folds
        k : int
            number of folds initially set

    Returns
    -------
        list of k tuples [(df_train, df_val), ...]

            training data

            validation data
    """
    folds = []
    df_train_dict = {}
    df_test_dict = {}
    for j in range(0, k):
        for key in folds_dict:
            assert k == len(folds_dict[key])
            df_train_dict[key] = folds_dict[key][j][0]
            df_test_dict[key] = folds_dict[key][j][1]
        folds.append((df_train_dict, df_test_dict))
        df_train_dict = {}
        df_test_dict = {}
    return folds


def _crossvalidation_with_time_threshold(df_dict, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct=0.0):
    """Splits data in k folds for crossvalidation accordingly to time threshold.

    Parameters
    ----------
        df_dict : dict
            data
        n_lags : int
            identical to NeuralProphet
        n_forecasts : int
            identical to NeuralProphet
        k : int
            number of CV folds
        fold_pct : float
            percentage of overall samples to be in each fold
        fold_overlap_pct : float
            percentage of overlap between the validation folds (default: 0.0)

    Returns
    -------
        list of k tuples [(df_train, df_val), ...]

            training data

            validation data
    """
    df_joint = join_dataframes_for_split_df(df_dict)
    total_samples = len(df_joint) - n_lags + 2 - (2 * n_forecasts)
    samples_fold = max(1, int(fold_pct * total_samples))
    samples_overlap = int(fold_overlap_pct * samples_fold)
    assert samples_overlap < samples_fold
    min_train = total_samples - samples_fold - (k - 1) * (samples_fold - samples_overlap)
    assert min_train >= samples_fold
    folds = []
    df_fold, _ = prep_copy_df_dict(df_dict)
    for i in range(k, 0, -1):
        threshold_time_stamp = find_time_threshold(df_fold, n_lags, n_forecasts, samples_fold, inputs_overbleed=True)
        df_dict_train, df_dict_val = split_considering_timestamp(
            df_fold, n_lags, n_forecasts, inputs_overbleed=True, threshold_time_stamp=threshold_time_stamp
        )
        folds.append((df_dict_train, df_dict_val))
        split_idx = len(df_joint) - samples_fold + samples_overlap
        df_joint = df_joint[:split_idx].reset_index(drop=True)
        threshold_time_stamp = df_joint["ds"].iloc[-1]
        for key in df_fold:
            df = df_fold[key].copy(deep=True)
            df_fold[key] = (
                df.copy(deep=True).iloc[: len(df[df["ds"] < threshold_time_stamp]) + 1].reset_index(drop=True)
            )
    folds = folds[::-1]
    return folds


def crossvalidation_split_df(
    df, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct=0.0, global_model_cv_type="global-time"
):
    """Splits data in k folds for crossvalidation.

    Parameters
    ----------
        df : pd.DataFrame or dict
            data
        n_lags : int
            identical to NeuralProphet
        n_forecasts : int
            identical to NeuralProphet
        k : int
            number of CV folds
        fold_pct : float
            percentage of overall samples to be in each fold
        fold_overlap_pct : float
            percentage of overlap between the validation folds (default: 0.0)
        global_model_cv_type : str
            Type of crossvalidation to apply to the dict of time series.

                options:

                    ``global-time`` (default) crossvalidation is performed according to a time stamp threshold.

                    ``local`` each episode will be crosvalidated locally (may cause time leakage among different episodes)

                    ``intersect`` only the time intersection of all the episodes will be considered. A considerable amount of data may not be used. However, this approach guarantees an equal number of train/test samples for each episode.

    Returns
    -------
        list of k tuples [(df_train, df_val), ...]

            training data

            validation data
    """

    if isinstance(df, pd.DataFrame):
        df_is_dict = False
        df_dict = {"__df__": df}
    elif isinstance(df, dict):
        df_is_dict = True
        df_dict = df
    else:
        raise ValueError("Please insert valid df type (i.e. pd.DataFrame, dict)")
    if len(df_dict) == 1:
        for df_name, df_i in df_dict.items():
            folds = _crossvalidation_split_df(df_i, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct)
    else:
        if global_model_cv_type == "global-time" or global_model_cv_type is None:
            # Use time threshold to perform crossvalidation (the distribution of data of different episodes may not be equivalent)
            folds = _crossvalidation_with_time_threshold(df_dict, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct)
        elif global_model_cv_type == "local":
            # Crossvalidate time series locally (time leakage may be a problem)
            folds_dict = {}
            for df_name, df_i in df_dict.items():
                folds_dict[df_name] = _crossvalidation_split_df(
                    df_i, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct
                )
            folds = unfold_dict_of_folds(folds_dict, k)
        elif global_model_cv_type == "intersect":
            # Use data only from the time period of intersection among time series
            folds_dict = {}
            # Check for intersection of time so time leakage does not occur among different time series
            start_date, end_date = find_valid_time_interval_for_cv(df_dict)
            for df_name, df_i in df_dict.items():
                mask = (df_i["ds"] >= start_date) & (df_i["ds"] <= end_date)
                df_i = df_i[mask].copy(deep=True)
                folds_dict[df_name] = _crossvalidation_split_df(
                    df_i, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct
                )
            folds = unfold_dict_of_folds(folds_dict, k)
        else:
            raise ValueError(
                "Please choose a valid type of global model crossvalidation (i.e. global-time, local, or intersect)"
            )
    return folds


def double_crossvalidation_split_df(df, n_lags, n_forecasts, k, valid_pct, test_pct):
    """Splits data in two sets of k folds for crossvalidation on validation and test data.

    Parameters
    ----------
        df (pd.DataFrame): data
        n_lags (int): identical to NeuralProphet
        n_forecasts (int): identical to NeuralProphet
        k (int): number of CV folds
        valid_pct (float): percentage of overall samples to be in validation
        test_pct (float): percentage of overall samples to be in test

    Returns
    -------
        tuple of k tuples [(folds_val, folds_test), …]
            elements same as :meth:`crossvalidation_split_df` returns
    """
    fold_pct_test = float(test_pct) / k
    folds_test = crossvalidation_split_df(df, n_lags, n_forecasts, k, fold_pct=fold_pct_test, fold_overlap_pct=0.0)
    df_train = folds_test[0][0]
    fold_pct_val = float(valid_pct) / k / (1.0 - test_pct)
    folds_val = crossvalidation_split_df(df_train, n_lags, n_forecasts, k, fold_pct=fold_pct_val, fold_overlap_pct=0.0)
    return folds_val, folds_test


def _split_df(df, n_lags, n_forecasts, valid_p, inputs_overbleed):
    """Splits timeseries df into train and validation sets.
    Additionally, prevents overbleed of targets. Overbleed of inputs can be configured.
    In case of global modeling the split could be either local or global.

    Parameters
    ----------
        df : pd.DataFrame
            data to be splitted
        n_lags : int
            identical to NeuralProphet
        n_forecasts : int
            identical to NeuralProphet
        valid_p : float, int
            fraction (0,1) of data to use for holdout validation set, or number of validation samples >1
        inputs_overbleed : bool
            Whether to allow last training targets to be first validation inputs (never targets)

    Returns
    -------
        pd.DataFrame
            training data
        pd.DataFrame
            validation data
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


def find_time_threshold(df_dict, n_lags, n_forecasts, valid_p, inputs_overbleed):
    """Find time threshold for dividing timeseries into train and validation sets.
    Prevents overbleed of targets. Overbleed of inputs can be configured.

    Parameters
    ----------
        df_dict : dict
            dict of data
        n_lags : int
            identical to NeuralProphet
        valid_p : float
            fraction (0,1) of data to use for holdout validation set
        inputs_overbleed : bool
            Whether to allow last training targets to be first validation inputs (never targets)

    Returns
    -------
        str
            time stamp threshold defines the boundary for the train and validation sets split.
    """
    df_joint = join_dataframes_for_split_df(df_dict)
    n_samples = len(df_joint) - n_lags + 2 - (2 * n_forecasts)
    n_samples = n_samples if inputs_overbleed else n_samples - n_lags
    if 0.0 < valid_p < 1.0:
        n_valid = max(1, int(n_samples * valid_p))
    else:
        assert valid_p >= 1
        assert type(valid_p) == int
        n_valid = valid_p
    n_train = n_samples - n_valid
    threshold_time_stamp = df_joint.loc[n_train, "ds"]
    log.debug("Time threshold: ", threshold_time_stamp)
    return threshold_time_stamp


def split_considering_timestamp(df_dict, n_lags, n_forecasts, inputs_overbleed, threshold_time_stamp):
    """Splits timeseries into train and validation sets according to given threshold_time_stamp.

    Parameters
    ----------
        df_dict : dict
            dataframe or dict of dataframes containing column ``ds``, ``y`` with all data
        n_lags : int
            identical to NeuralProphet
        n_forecasts : int
            identical to NeuralProphet
        inputs_overbleed : bool
            Whether to allow last training targets to be first validation inputs (never targets)
        threshold_time_stamp : str
            time stamp boundary that defines splitting of data

    Returns
    -------
        pd.DataFrame, dict
            training data
        pd.DataFrame, dict
            validation data
    """
    df_train = {}
    df_val = {}
    for key in df_dict:
        if df_dict[key]["ds"].max() < threshold_time_stamp:
            df_train[key] = df_dict[key].copy(deep=True).reset_index(drop=True)
        elif df_dict[key]["ds"].min() > threshold_time_stamp:
            df_val[key] = df_dict[key].copy(deep=True).reset_index(drop=True)
        else:
            df = df_dict[key].copy(deep=True)
            n_train = len(df[df["ds"] < threshold_time_stamp])
            split_idx_train = n_train + n_lags + n_forecasts - 1
            split_idx_val = split_idx_train - n_lags if inputs_overbleed else split_idx_train
            df_train[key] = df.copy(deep=True).iloc[:split_idx_train].reset_index(drop=True)
            df_val[key] = df.copy(deep=True).iloc[split_idx_val:].reset_index(drop=True)
    return df_train, df_val


def split_df(df, n_lags, n_forecasts, valid_p=0.2, inputs_overbleed=True, local_split=False):
    """Splits timeseries df into train and validation sets.

    Prevents overbleed of targets. Overbleed of inputs can be configured.
    In case of global modeling the split could be either local or global.

    Parameters
    ----------
        df_dict : dict
            dataframe or dict of dataframes containing column ``ds``, ``y`` with all data
        n_lags : int
            identical to NeuralProphet
        n_forecasts : int
            identical to NeuralProphet
        valid_p : float, int
            fraction (0,1) of data to use for holdout validation set, or number of validation samples >1
        inputs_overbleed : bool
            Whether to allow last training targets to be first validation inputs (never targets)
        local_split : bool
            when set to true, each episode from a dict of dataframes will be split locally

    Returns
    -------
        pd.DataFrame, dict
            training data
        pd.DataFrame, dict
            validation data
    """
    if isinstance(df, pd.DataFrame):
        df_is_dict = False
        df_dict = {"__df__": df}
    elif isinstance(df, dict):
        df_is_dict = True
        df_dict = df
    else:
        raise ValueError("Please insert valid df type (i.e. pd.DataFrame, dict)")
    df_train = {}
    df_val = {}
    if local_split:
        for key in df_dict:
            df_train[key], df_val[key] = _split_df(df_dict[key], n_lags, n_forecasts, valid_p, inputs_overbleed)
    else:
        if len(df_dict) == 1:
            for df_name, df_i in df_dict.items():
                df_train[df_name], df_val[df_name] = _split_df(df_i, n_lags, n_forecasts, valid_p, inputs_overbleed)
        else:
            # Split data according to time threshold defined by the valid_p
            threshold_time_stamp = find_time_threshold(df_dict, n_lags, n_forecasts, valid_p, inputs_overbleed)
            df_train, df_val = split_considering_timestamp(
                df_dict, n_lags, n_forecasts, inputs_overbleed, threshold_time_stamp
            )
    if not df_is_dict:
        df_train, df_val = df_train["__df__"], df_val["__df__"]
    return df_train, df_val


def make_future_df(
    df_columns, last_date, periods, freq, events_config=None, events_df=None, regressor_config=None, regressors_df=None
):
    """Extends df periods number steps into future.

    Parameters
    ----------
        df_columns : pd.DataFrame
            Dataframe columns
        last_date : pd.Datetime
            last history date
        periods : int
            number of future steps to predict
        freq : str
            Data step sizes. Frequency of data recording, any valid frequency
            for pd.date_range, such as ``D`` or ``M``
        events_config : OrderedDict
            User specified events configs
        events_df : pd.DataFrame
            containing column ``ds`` and ``event``
        regressor_config : OrderedDict
            configuration for user specified regressors,
        regressors_df : pd.DataFrame
            containing column ``ds`` and one column for each of the external regressors

    Returns
    -------
        pd.DataFrame
            input df with ``ds`` extended into future, and ``y`` set to None
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

    Parameters
    ----------
        df : pd.DataFrame
            Dataframe with columns ``ds`` datestamps and ``y`` time series values
        events_config : OrderedDict
            User specified events configs
        events_df : pd.DataFrame
            containing column ``ds`` and ``event``

    Returns
    -------
        pd.DataFrame
            input df with columns for user_specified features
    """

    for event in events_config.keys():
        event_feature = pd.Series([0.0] * df.shape[0])
        dates = events_df[events_df.event == event].ds
        event_feature[df.ds.isin(dates)] = 1.0
        df[event] = event_feature
    return df


def add_missing_dates_nan(df, freq):
    """Fills missing datetimes in ``ds``, with NaN for all other columns

    Parameters
    ----------
        df : pd.Dataframe
            with column ``ds``  datetimes
        freq : str
            Frequency of data recording, any valid frequency for pd.date_range,
            such as ``D`` or ``M``

    Returns
    -------
        pd.DataFrame
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

    Parameters
    ----------
        series : pd.Series
            series with nan to be filled in.
        limit_linear : int
            maximum number of missing values to impute.

            Note
            ----
            because imputation is done in both directions, this value is effectively doubled.

        rolling : int
            maximal number of missing values to impute.

            Note
            ----
            window width is rolling + 2*limit_linear

    Returns
    -------
        pd.DataFrame
            manipulated dataframe containing filled values
    """
    # impute small gaps linearly:
    series = series.interpolate(method="linear", limit=limit_linear, limit_direction="both")
    # fill remaining gaps with rolling avg
    is_na = pd.isna(series)
    rolling_avg = series.rolling(rolling + 2 * limit_linear, min_periods=2 * limit_linear, center=True).mean()
    series.loc[is_na] = rolling_avg[is_na]
    remaining_na = sum(series.isnull())
    return series, remaining_na


def get_freq_dist(ds_col):
    """Get frequency distribution of ``ds`` column.

    Parameters
    ----------
        ds_col : pd.DataFrame
            ``ds`` column of dataframe

    Returns
    -------
        tuple
            numeric delta values (``ms``) and distribution of frequency counts
    """
    converted_ds = pd.to_datetime(ds_col).view(dtype=np.int64)
    diff_ds = np.unique(converted_ds.diff(), return_counts=True)
    return diff_ds


def convert_str_to_num_freq(freq_str):
    """Convert frequency tags into numeric delta in ms

    Parameters
    ----------
        freq_str str
            frequency tag

    Returns
    -------
        numeric
            frequency numeric delta in ms
    """
    if freq_str is None:
        freq_num = 0
    else:
        aux_ts = pd.DataFrame(pd.date_range("1994-01-01", periods=100, freq=freq_str))
        frequencies, distribution = get_freq_dist(aux_ts[0])
        freq_num = frequencies[np.argmax(distribution)]
        # if freq_str == "B" or freq_str == "BH":  # exception - Business day and Business hour
        #     freq_num = freq_num + 0.777
    return freq_num


def convert_num_to_str_freq(freq_num, initial_time_stamp):
    """Convert numeric frequencies into frequency tags

    Parameters
    ----------
        freq_num : int
            numeric values of delta in ms
        initial_time_stamp : str
            initial time stamp of data

    Returns
    -------
        str
            frequency tag
    """
    aux_ts = pd.date_range(initial_time_stamp, periods=100, freq=pd.to_timedelta(freq_num))
    freq_str = pd.infer_freq(aux_ts)
    return freq_str


def get_dist_considering_two_freqs(dist):
    """Add occasions of the two most common frequencies

    Note
    ----
    Useful for the frequency exceptions (i.e. ``M``, ``Y``, ``Q``, ``B``, and ``BH``).

    Parameters
    ----------
        dist : list
            list of occasions of frequencies

    Returns
    -------
        numeric
            sum of the two most common frequencies occasions
    """
    # get distribution considering the two most common frequencies - useful for monthly and business day
    f1 = dist.max()
    dist = np.delete(dist, np.argmax(dist))
    f2 = dist.max()
    return f1 + f2


def _infer_frequency(df, freq, min_freq_percentage=0.7):
    """Automatically infers frequency of dataframe or dict of dataframes.

    Parameters
    ----------
        df : pd.DataFrame
            Dataframe with columns ``ds`` datestamps and ``y`` time series values
        freq : str
            Data step sizes, i.e. frequency of data recording,

            Note
            ----
            Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto``
            (default) to automatically set frequency.

        min_freq_percentage : float
            threshold for defining major frequency of data (default: ``0.7``

    Returns
    -------
        str
            Valid frequency tag according to major frequency.

    """
    frequencies, distribution = get_freq_dist(df["ds"])
    # exception - monthly df (31 days freq or 30 days freq)
    if frequencies[np.argmax(distribution)] == 2.6784e15 or frequencies[np.argmax(distribution)] == 2.592e15:
        dominant_freq_percentage = get_dist_considering_two_freqs(distribution) / len(df["ds"])
        num_freq = 2.6784e15
        inferred_freq = "MS" if pd.to_datetime(df["ds"][0]).day < 15 else "M"
    # exception - yearly df (365 days freq or 366 days freq)
    elif frequencies[np.argmax(distribution)] == 3.1536e16 or frequencies[np.argmax(distribution)] == 3.16224e16:
        dominant_freq_percentage = get_dist_considering_two_freqs(distribution) / len(df["ds"])
        num_freq = 3.1536e16
        inferred_freq = "YS" if pd.to_datetime(df["ds"][0]).day < 15 else "Y"
    # exception - quaterly df (most common == 92 days - 3rd,4th quarters and second most common == 91 days 2nd quarter and 1st quarter in leap year)
    elif (
        frequencies[np.argmax(distribution)] == 7.9488e15
        and frequencies[np.argsort(distribution, axis=0)[-2]] == 7.8624e15
    ):
        dominant_freq_percentage = get_dist_considering_two_freqs(distribution) / len(df["ds"])
        num_freq = 7.9488e15
        inferred_freq = "QS" if pd.to_datetime(df["ds"][0]).day < 15 else "Q"
    # exception - Business day (most common == day delta and second most common == 3 days delta and second most common is at least 12% of the deltas)
    elif (
        frequencies[np.argmax(distribution)] == 8.64e13
        and frequencies[np.argsort(distribution, axis=0)[-2]] == 2.592e14
        and distribution[np.argsort(distribution, axis=0)[-2]] / len(df["ds"]) >= 0.12
    ):
        dominant_freq_percentage = get_dist_considering_two_freqs(distribution) / len(df["ds"])
        num_freq = 8.64e13
        inferred_freq = "B"
    # exception - Business hour (most common == hour delta and second most common == 17 hours delta and second most common is at least 8% of the deltas)
    elif (
        frequencies[np.argmax(distribution)] == 3.6e12
        and frequencies[np.argsort(distribution, axis=0)[-2]] == 6.12e13
        and distribution[np.argsort(distribution, axis=0)[-2]] / len(df["ds"]) >= 0.08
    ):
        dominant_freq_percentage = get_dist_considering_two_freqs(distribution) / len(df["ds"])
        num_freq = 3.6e12
        inferred_freq = "BH"
    else:
        dominant_freq_percentage = distribution.max() / len(df["ds"])
        num_freq = frequencies[np.argmax(distribution)]  # get value of most common diff
        inferred_freq = convert_num_to_str_freq(num_freq, df["ds"].iloc[0])

    log.info(
        "Major frequency {} corresponds to {}% of the data.".format(
            inferred_freq, np.round(dominant_freq_percentage * 100, 3)
        )
    )
    ideal_freq_exists = True if dominant_freq_percentage >= min_freq_percentage else False
    if ideal_freq_exists:
        # if major freq exists
        if freq == "auto" or freq is None:  # automatically set df freq to inferred freq
            freq_str = inferred_freq
            log.info("Dataframe freq automatically defined as {}".format(freq_str))
        else:
            freq_str = freq
            if convert_str_to_num_freq(freq) != convert_str_to_num_freq(
                inferred_freq
            ):  # check if given freq is the major
                log.warning("Defined frequency {} is different than major frequency {}".format(freq_str, inferred_freq))
            else:
                if freq_str in [
                    "M",
                    "MS",
                    "Q",
                    "QS",
                    "Y",
                    "YS",
                ]:  # temporary solution for avoiding setting wrong start date
                    freq_str = inferred_freq
                log.info("Defined frequency is equal to major frequency - {}".format(freq_str))
    else:
        # if ideal freq does not exist
        if freq == "auto" or freq is None:
            log.warning(
                "The auto-frequency feature is not able to detect the following frequencies: SM, BM, CBM, SMS, BMS, CBMS, BQ, BQS, BA, or, BAS. If the frequency of the dataframe is any of the mentioned please define it manually."
            )
            raise ValueError("Detected multiple frequencies in the timeseries please pre-process data.")
        else:
            freq_str = freq
            log.warning(
                "Dataframe has multiple frequencies. It will be resampled according to given freq {}. Ignore message if actual frequency is any of the following:  SM, BM, CBM, SMS, BMS, CBMS, BQ, BQS, BA, or, BAS.".format(
                    freq
                )
            )
    return freq_str


def infer_frequency(df, freq, n_lags, min_freq_percentage=0.7):
    """Automatically infers frequency of dataframe or dict of dataframes.

    Parameters
    ----------
        df : pd.DataFrame
            Dataframe with columns ``ds`` datestamps and ``y`` time series values
        freq : str
            Data step sizes, i.e. frequency of data recording,

            Note
            ----
            Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto`` (default) to automatically set frequency.
        n_lags : int
            identical to NeuralProphet
        min_freq_percentage : float
            threshold for defining major frequency of data (default: ``0.7``



    Returns
    -------
        str
            Valid frequency tag according to major frequency.

    """

    df_dict, received_unnamed_df = prep_copy_df_dict(df)
    freq_df = list()
    for key in df_dict:
        freq_df.append(_infer_frequency(df_dict[key], freq, min_freq_percentage))
    if len(set(freq_df)) != 1 and n_lags > 0:
        raise ValueError(
            "One or more dataframes present different major frequencies, please make sure all dataframes present the same major frequency for auto-regression"
        )
    elif len(set(freq_df)) != 1 and n_lags == 0:
        # The most common freq is set as the main one (but it does not really matter for Prophet approach)
        freq_str = max(set(freq_df), key=freq_df.count)
        log.warning("One or more major frequencies are different - setting main frequency as {}".format(freq_str))
    else:
        freq_str = freq_df[0]
    return freq_str


def compare_dict_keys(dict_1, dict_2, name_dict_1, name_dict_2):
    """Compare keys of two different dicts (i.e., events and dataframes).

    Parameters
    ----------
        dict_1 : dict
            first dict
        dict_2 : dict
            second dict
        name_dict_1 : str
            name of first dict
        name_dict_2 : str
            name of second dict

    """
    df_names_1, df_names_2 = list(dict_1.keys()), list(dict_2.keys())
    if len(df_names_1) != len(df_names_2):
        raise ValueError(
            "Please, make sure {} and {} dicts have the same number of terms".format(name_dict_1, name_dict_2)
        )
    missing_names = [name for name in df_names_2 if name not in df_names_1]
    if len(missing_names) > 0:
        raise ValueError(" Key(s) {} not valid - missing from {} dict keys".format(missing_names, name_dict_1))
    log.debug("{} and {} dicts are compatible".format(name_dict_1, name_dict_2))

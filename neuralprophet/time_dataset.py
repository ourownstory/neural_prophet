from collections import OrderedDict
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from neuralprophet import hdays as hdays_part2
import holidays as hdays_part1
from collections import defaultdict
from neuralprophet import utils
from neuralprophet.df_utils import get_max_num_lags
import logging

log = logging.getLogger("NP.time_dataset")


class GlobalTimeDataset(Dataset):
    def __init__(self, df, **kwargs):
        """Initialize Timedataset from time-series df.

        Parameters
        ----------
            df : pd.DataFrame
                dataframe containing column ``ds``, ``y``, and optionally``ID`` and
                normalized columns normalized columns ``ds``, ``y``, ``t``, ``y_scaled``
            **kwargs : dict
                Identical to :meth:`tabularize_univariate_datetime`
        """
        self.combined_timedataset = []
        # TODO (future): vectorize
        self.length = 0
        for df_name, df_i in df.groupby("ID"):
            timedataset = TimeDataset(df_i, df_name, **kwargs)
            self.length += timedataset.length
            for i in range(0, len(timedataset)):
                self.combined_timedataset.append(timedataset[i])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.combined_timedataset[idx]


class TimeDataset(Dataset):
    """Create a PyTorch dataset of a tabularized time-series"""

    def __init__(self, df, name, **kwargs):
        """Initialize Timedataset from time-series df.

        Parameters
        ----------
            df : pd.DataFrame
                Time series data
            name : str
                Name of time-series
            **kwargs : dict
                Identical to :meth:`tabularize_univariate_datetime`
        """
        self.name = name
        self.length = None
        self.inputs = OrderedDict({})
        self.targets = None
        self.meta = OrderedDict({})
        self.two_level_inputs = ["seasonalities", "covariates"]
        inputs, targets, drop_missing = tabularize_univariate_datetime(df, **kwargs)
        self.init_after_tabularized(inputs, targets)
        self.drop_nan_after_init(df, kwargs["predict_steps"], drop_missing)

    def drop_nan_after_init(self, df, predict_steps, drop_missing):
        """Checks if inputs/targets contain any NaN values and drops them, if user opts to.

        Parameters
        ----------
            drop_missing : bool
                whether to automatically drop missing samples from the data
        """
        nan_idx = []
        for i, (inputs, targets, meta) in enumerate(self):
            for key, data in inputs.items():  # key: lags/seasonality, data: torch tensor (oder OrderedDict)
                if key in self.two_level_inputs or key == "events" or key == "regressors":
                    # Extract tensor out of OrderedDict to see if it contains NaNs
                    tuple_list = list(data.items())
                    tensor = tuple_list[0][1]
                    if np.isnan(np.array(tensor)).any() and (i not in nan_idx):
                        nan_idx.append(i)
                else:
                    # save index of the NaN-containing sample
                    if np.isnan(np.array(data)).any() and (i not in nan_idx):
                        nan_idx.append(i)
            if np.isnan(np.array(targets)).any() and (i not in nan_idx):
                if (
                    i < len(self) - predict_steps
                ):  # do not remove the targets that were inserted for prediction at the end
                    nan_idx.append(i)  # nan_idx contains all indices of inputs/targets containing 1 or more NaN values
        if drop_missing == True and len(nan_idx) > 0:
            log.warning(f"{len(nan_idx)} samples with missing values were dropped from the data. ")
            for key, data in self.inputs.items():
                if key not in ["time", "lags"]:
                    for name, features in data.items():
                        self.inputs[key][name] = np.delete(self.inputs[key][name], nan_idx, 0)
                else:
                    self.inputs[key] = np.delete(self.inputs[key], nan_idx, 0)
            self.targets = np.delete(self.targets, nan_idx, 0)
            self.length = self.inputs["time"].shape[0]
        if drop_missing == False and len(nan_idx) > 0:
            raise ValueError(
                "Inputs/targets with missing values detected. "
                "Please either adjust imputation parameters, or set 'drop_missing' to True to drop those samples."
            )

    def init_after_tabularized(self, inputs, targets=None):
        """Create Timedataset with data.

        Parameters
        ----------
            inputs : ordered dict
                Identical to returns from :meth:`tabularize_univariate_datetime`
            targets : np.array, float
                Identical to returns from :meth:`tabularize_univariate_datetime`
        """
        inputs_dtype = {
            "time": torch.float,
            # "changepoints": torch.bool,
            "seasonalities": torch.float,
            "events": torch.float,
            "lags": torch.float,
            "covariates": torch.float,
            "regressors": torch.float,
        }
        targets_dtype = torch.float
        self.length = inputs["time"].shape[0]

        for key, data in inputs.items():
            if key in self.two_level_inputs or key == "events" or key == "regressors":
                self.inputs[key] = OrderedDict({})
                for name, features in data.items():
                    self.inputs[key][name] = torch.from_numpy(features).type(inputs_dtype[key])
            else:
                self.inputs[key] = torch.from_numpy(data).type(inputs_dtype[key])
        self.targets = torch.from_numpy(targets).type(targets_dtype).unsqueeze(dim=2)
        self.meta["df_name"] = self.name

    def __getitem__(self, index):
        """Overrides parent class method to get an item at index.

        Parameters
        ----------
            index : int
                Sample location in dataset

        Returns
        -------
        OrderedDict
            Model inputs, each of len(df) but with varying dimensions

            Note
            ----
            Contains the following data:

            Model Inputs
                * ``time`` (np.array, float), dims: (num_samples, 1)
                * ``seasonalities`` (OrderedDict), named seasonalities
                each with features (np.array, float) - dims: (num_samples, n_features[name])
                * ``lags`` (np.array, float), dims: (num_samples, n_lags)
                * ``covariates`` (OrderedDict), named covariates,
                each with features (np.array, float) of dims: (num_samples, n_lags)
                * ``events`` (OrderedDict), events,
                each with features (np.array, float) of dims: (num_samples, n_lags)
                * ``regressors`` (OrderedDict), regressors,
                each with features (np.array, float) of dims: (num_samples, n_lags)
        np.array, float
            Targets to be predicted of same length as each of the model inputs, dims: (num_samples, n_forecasts)
        """
        # Future TODO: vectorize
        sample = OrderedDict({})
        for key, data in self.inputs.items():
            if key in self.two_level_inputs:
                sample[key] = OrderedDict({})
                for name, period_features in self.inputs[key].items():
                    sample[key][name] = period_features[index]
            elif key == "events" or key == "regressors":
                sample[key] = OrderedDict({})
                for mode, features in self.inputs[key].items():
                    sample[key][mode] = features[index, :, :]
            else:
                sample[key] = data[index]
        targets = self.targets[index]
        meta = self.meta
        return sample, targets, meta

    def __len__(self):
        """Overrides Parent class method to get data length."""
        return self.length


def tabularize_univariate_datetime(
    df,
    predict_mode=False,
    n_lags=0,
    n_forecasts=1,
    predict_steps=1,
    config_season=None,
    config_events=None,
    config_country_holidays=None,
    config_covar=None,
    config_regressors=None,
    config_missing=None,
):
    """Create a tabular dataset from univariate timeseries for supervised forecasting.

    Note
    ----
    Data must have no gaps.
    If data contains missing values, they are ignored for the creation of the dataset.

    Parameters
    ----------
        df : pd.DataFrame
            Sequence of observations with original ``ds``, ``y`` and normalized ``t``, ``y_scaled`` columns
        config_season : configure.Season
            Configuration for seasonalities
        n_lags : int
            Number of lagged values of series to include as model inputs (aka AR-order)
        n_forecasts : int
            Number of steps to forecast into future
        config_events : OrderedDict)
            User specified events, each with their upper, lower windows (int) and regularization
        config_country_holidays : OrderedDict)
            Configurations (holiday_names, upper, lower windows, regularization) for country specific holidays
        config_covar : configure.Covar
            Configuration for covariates
        config_regressors : OrderedDict
            Configuration for regressors
        predict_mode : bool
            Chooses the prediction mode

            Options
                * (default) ``False``: Includes target values
                * ``True``: Does not include targets but includes entire dataset as input

    Returns
    -------
        OrderedDict
            Model inputs, each of len(df) but with varying dimensions

            Note
            ----
            Contains the following data:

            Model Inputs
                * ``time`` (np.array, float), dims: (num_samples, 1)
                * ``seasonalities`` (OrderedDict), named seasonalities
                each with features (np.array, float) - dims: (num_samples, n_features[name])
                * ``lags`` (np.array, float), dims: (num_samples, n_lags)
                * ``covariates`` (OrderedDict), named covariates,
                each with features (np.array, float) of dims: (num_samples, n_lags)
                * ``events`` (OrderedDict), events,
                each with features (np.array, float) of dims: (num_samples, n_lags)
                * ``regressors`` (OrderedDict), regressors,
                each with features (np.array, float) of dims: (num_samples, n_lags)
        np.array, float
            Targets to be predicted of same length as each of the model inputs, dims: (num_samples, n_forecasts)
    """
    max_lags = get_max_num_lags(config_covar, n_lags)
    n_samples = len(df) - max_lags + 1 - n_forecasts
    # data is stored in OrderedDict
    inputs = OrderedDict({})

    def _stride_time_features_for_forecasts(x):
        # only for case where n_lags > 0
        return np.array([x[max_lags + i : max_lags + i + n_forecasts] for i in range(n_samples)], dtype=np.float64)

    # time is the time at each forecast step
    t = df.loc[:, "t"].values
    if max_lags == 0:
        assert n_forecasts == 1
        time = np.expand_dims(t, 1)
    else:
        time = _stride_time_features_for_forecasts(t)
    inputs["time"] = time

    if config_season is not None:
        seasonalities = seasonal_features_from_dates(df["ds"], config_season)
        for name, features in seasonalities.items():
            if max_lags == 0:
                seasonalities[name] = np.expand_dims(features, axis=1)
            else:
                # stride into num_forecast at dim=1 for each sample, just like we did with time
                seasonalities[name] = _stride_time_features_for_forecasts(features)
        inputs["seasonalities"] = seasonalities

    def _stride_lagged_features(df_col_name, feature_dims):
        # only for case where max_lags > 0
        assert feature_dims >= 1
        series = df.loc[:, df_col_name].values
        ## Added dtype=np.float64 to solve the problem with np.isnan for ubuntu test
        return np.array(
            [series[i + max_lags - feature_dims : i + max_lags] for i in range(n_samples)], dtype=np.float64
        )

    if n_lags > 0 and "y" in df.columns:
        inputs["lags"] = _stride_lagged_features(df_col_name="y_scaled", feature_dims=n_lags)

    if config_covar is not None and max_lags > 0:
        covariates = OrderedDict({})
        for covar in df.columns:
            if covar in config_covar:
                assert config_covar[covar].n_lags > 0
                window = config_covar[covar].n_lags
                covariates[covar] = _stride_lagged_features(df_col_name=covar, feature_dims=window)
        inputs["covariates"] = covariates

    # get the regressors features
    if config_regressors is not None:
        additive_regressors, multiplicative_regressors = make_regressors_features(df, config_regressors)

        regressors = OrderedDict({})
        if max_lags == 0:
            if additive_regressors is not None:
                regressors["additive"] = np.expand_dims(additive_regressors, axis=1)
            if multiplicative_regressors is not None:
                regressors["multiplicative"] = np.expand_dims(multiplicative_regressors, axis=1)
        else:
            if additive_regressors is not None:
                additive_regressor_feature_windows = []
                for i in range(0, additive_regressors.shape[1]):
                    # stride into num_forecast at dim=1 for each sample, just like we did with time
                    stride = _stride_time_features_for_forecasts(additive_regressors[:, i])
                    additive_regressor_feature_windows.append(stride)
                additive_regressors = np.dstack(additive_regressor_feature_windows)
                regressors["additive"] = additive_regressors

            if multiplicative_regressors is not None:
                multiplicative_regressor_feature_windows = []
                for i in range(0, multiplicative_regressors.shape[1]):
                    # stride into num_forecast at dim=1 for each sample, just like we did with time
                    stride = _stride_time_features_for_forecasts(multiplicative_regressors[:, i])
                    multiplicative_regressor_feature_windows.append(stride)
                multiplicative_regressors = np.dstack(multiplicative_regressor_feature_windows)
                regressors["multiplicative"] = multiplicative_regressors

        inputs["regressors"] = regressors

    # get the events features
    if config_events is not None or config_country_holidays is not None:
        additive_events, multiplicative_events = make_events_features(df, config_events, config_country_holidays)

        events = OrderedDict({})
        if max_lags == 0:
            if additive_events is not None:
                events["additive"] = np.expand_dims(additive_events, axis=1)
            if multiplicative_events is not None:
                events["multiplicative"] = np.expand_dims(multiplicative_events, axis=1)
        else:
            if additive_events is not None:
                additive_event_feature_windows = []
                for i in range(0, additive_events.shape[1]):
                    # stride into num_forecast at dim=1 for each sample, just like we did with time
                    additive_event_feature_windows.append(_stride_time_features_for_forecasts(additive_events[:, i]))
                additive_events = np.dstack(additive_event_feature_windows)
                events["additive"] = additive_events

            if multiplicative_events is not None:
                multiplicative_event_feature_windows = []
                for i in range(0, multiplicative_events.shape[1]):
                    # stride into num_forecast at dim=1 for each sample, just like we did with time
                    multiplicative_event_feature_windows.append(
                        _stride_time_features_for_forecasts(multiplicative_events[:, i])
                    )
                multiplicative_events = np.dstack(multiplicative_event_feature_windows)
                events["multiplicative"] = multiplicative_events

        inputs["events"] = events

    if predict_mode:
        targets = np.empty_like(time)
        targets = np.nan_to_num(targets)
    else:
        targets = _stride_time_features_for_forecasts(df["y_scaled"].values)

    tabularized_input_shapes_str = ""
    for key, value in inputs.items():
        if key in ["seasonalities", "covariates", "events", "regressors"]:
            for name, period_features in value.items():
                tabularized_input_shapes_str += f"    {name} {key} {period_features}\n"
        else:
            tabularized_input_shapes_str += f"    {key} {value.shape} \n"
    log.debug(f"Tabularized inputs shapes: \n{tabularized_input_shapes_str}")

    return inputs, targets, config_missing.drop_missing


def fourier_series(dates, period, series_order):
    """Provides Fourier series components with the specified frequency and order.

    Note
    ----
    Identical to OG Prophet.

    Parameters
    ----------
        dates : pd.Series
            Containing timestamps
        period : float
            Number of days of the period
        series_order : int
            Number of fourier components

    Returns
    -------
        np.array
            Matrix with seasonality features
    """
    # convert to days since epoch
    t = np.array((dates - datetime(1970, 1, 1)).dt.total_seconds().astype(float)) / (3600 * 24.0)
    return fourier_series_t(t, period, series_order)


def fourier_series_t(t, period, series_order):
    """Provides Fourier series components with the specified frequency and order.

    Note
    ----
    This function is identical to Meta AI's Prophet Library

    Parameters
    ----------
        t : pd.Series, float
            Containing time as floating point number of days
        period : float
            Number of days of the period
        series_order : int
            Number of fourier components

    Returns
    -------
        np.array
            Matrix with seasonality features
    """
    features = np.column_stack(
        [fun((2.0 * (i + 1) * np.pi * t / period)) for i in range(series_order) for fun in (np.sin, np.cos)]
    )
    return features


def make_country_specific_holidays_df(year_list, country):
    """
    Make dataframe of country specific holidays for given years and countries

    Parameters
    ----------
        year_list : list
            List of years
        country : string
            Country name

    Returns
    -------
        pd.DataFrame
            Containing country specific holidays df with columns 'ds' and 'holiday'
    """

    try:
        country_specific_holidays = getattr(hdays_part2, country)(years=year_list)
    except AttributeError:
        try:
            country_specific_holidays = getattr(hdays_part1, country)(years=year_list)
        except AttributeError:
            raise AttributeError(f"Holidays in {country} are not currently supported!")
    country_specific_holidays_dict = defaultdict(list)
    for date, holiday in country_specific_holidays.items():
        country_specific_holidays_dict[holiday].append(pd.to_datetime(date))
    return country_specific_holidays_dict


def make_events_features(df, config_events=None, config_country_holidays=None):
    """
    Construct arrays of all event features

    Parameters
    ----------
        df : pd.DataFrame
            Dataframe with all values including the user specified events (provided by user)
        config_events : OrderedDict
            User specified events, each with their upper, lower windows (int), regularization
        config_country_holidays : configure.Holidays
            Configurations (holiday_names, upper, lower windows, regularization) for country specific holidays

    Returns
    -------
        np.array
            All additive event features (both user specified and country specific)
        np.array
            All multiplicative event features (both user specified and country specific)
    """

    additive_events = pd.DataFrame()
    multiplicative_events = pd.DataFrame()

    # create all user specified events
    if config_events is not None:
        for event, configs in config_events.items():
            if event not in df.columns:
                df[event] = np.zeros_like(df["ds"], dtype=np.float64)
            feature = df[event]
            lw = configs.lower_window
            uw = configs.upper_window
            mode = configs.mode
            # create lower and upper window features
            for offset in range(lw, uw + 1):
                key = utils.create_event_names_for_offsets(event, offset)
                offset_feature = feature.shift(periods=offset, fill_value=0.0)
                if mode == "additive":
                    additive_events[key] = offset_feature
                else:
                    multiplicative_events[key] = offset_feature

    # create all country specific holidays
    if config_country_holidays is not None:
        lw = config_country_holidays.lower_window
        uw = config_country_holidays.upper_window
        mode = config_country_holidays.mode
        year_list = list({x.year for x in df.ds})
        country_holidays_dict = make_country_specific_holidays_df(year_list, config_country_holidays.country)
        for holiday in config_country_holidays.holiday_names:
            feature = pd.Series([0.0] * df.shape[0])
            if holiday in country_holidays_dict.keys():
                dates = country_holidays_dict[holiday]
                feature[df.ds.isin(dates)] = 1.0
            for offset in range(lw, uw + 1):
                key = utils.create_event_names_for_offsets(holiday, offset)
                offset_feature = feature.shift(periods=offset, fill_value=0)
                if mode == "additive":
                    additive_events[key] = offset_feature
                else:
                    multiplicative_events[key] = offset_feature

    # Make sure column order is consistent
    if not additive_events.empty:
        additive_events = additive_events[sorted(additive_events.columns.tolist())]
        additive_events = additive_events.values
    else:
        additive_events = None
    if not multiplicative_events.empty:
        multiplicative_events = multiplicative_events[sorted(multiplicative_events.columns.tolist())]
        multiplicative_events = multiplicative_events.values
    else:
        multiplicative_events = None

    return additive_events, multiplicative_events


def make_regressors_features(df, config_regressors):
    """Construct arrays of all scalar regressor features

    Parameters
    ----------
        df : pd.DataFrame
            Dataframe with all values including the user specified regressors
        config_regressors : OrderedDict
            User specified regressors config

    Returns
    -------
        np.array
            All additive regressor features
        np.array
            All multiplicative regressor features

    """
    additive_regressors = pd.DataFrame()
    multiplicative_regressors = pd.DataFrame()

    for reg in df.columns:
        if reg in config_regressors:
            mode = config_regressors[reg].mode
            if mode == "additive":
                additive_regressors[reg] = df[reg]
            else:
                multiplicative_regressors[reg] = df[reg]

    if not additive_regressors.empty:
        additive_regressors = additive_regressors[sorted(additive_regressors.columns.tolist())]
        additive_regressors = additive_regressors.values
    else:
        additive_regressors = None
    if not multiplicative_regressors.empty:
        multiplicative_regressors = multiplicative_regressors[sorted(multiplicative_regressors.columns.tolist())]
        multiplicative_regressors = multiplicative_regressors.values
    else:
        multiplicative_regressors = None

    return additive_regressors, multiplicative_regressors


def seasonal_features_from_dates(dates, config_season):
    """Dataframe with seasonality features.

    Includes seasonality features, holiday features, and added regressors.

    Parameters
    ----------
        dates : pd.Series
            With dates for computing seasonality features
        config_season : configure.Season
            Configuration for seasonalities

    Returns
    -------
        OrderedDict
            Dictionary with keys for each period name containing an np.array
            with the respective regression features. each with dims: (len(dates), 2*fourier_order)
    """
    assert len(dates.shape) == 1
    seasonalities = OrderedDict({})
    # Seasonality features
    for name, period in config_season.periods.items():
        if period.resolution > 0:
            if config_season.computation == "fourier":
                features = fourier_series(
                    dates=dates,
                    period=period.period,
                    series_order=period.resolution,
                )
            else:
                raise NotImplementedError
            seasonalities[name] = features
    return seasonalities

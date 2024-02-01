import logging
from collections import OrderedDict, defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

from neuralprophet import configure, utils
from neuralprophet.df_utils import get_max_num_lags
from neuralprophet.hdays_utils import get_country_holidays

log = logging.getLogger("NP.time_dataset")


class TimeDataset(Dataset):
    """Create a PyTorch dataset of a tabularized time-series"""

    def __init__(
        self,
        df,
        name,
        predict_mode,
        n_lags,
        n_forecasts,
        prediction_frequency,
        predict_steps,
        config_seasonality,
        config_events,
        config_country_holidays,
        config_regressors,
        config_lagged_regressors,
        config_missing,
    ):
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
        ## Outcome after a call to init (summary):
        # - add events and holidays columns to df
        # - calculated the number of usable samples (accounting for nan and filters)
        # - creates mapping of sample index to df index

        ## Context Notes
        # Currently done to df before it arrives here:
        # -> fit calls prep_or_copy_df, _check_dataframe, and _handle_missing_data, passes to _train
        # -> _train calls prep_or_copy_df, then passes to init_train_loader, which returns the train_loader
        # -> init_train_loader calls prep_or_copy_df, _normalize, _create_dataset (returns TimeDataset), returns dataset wrapped in DataLoader
        # ->_create_dataset calls prep_or_copy_df, then returns GlobalTimeDataset
        # Future TODO: integrate some of these preprocessing steps happening outside?

        self.df = df.reset_index(drop=True)  # Needed for index based operations in __getitem__
        if "index" in list(self.df.columns):  # should not be the case
            self.df = self.df.drop("index", axis=1)
        self.meta = OrderedDict({})
        self.meta["df_name"] = name

        self.predict_mode = predict_mode
        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self.prediction_frequency = prediction_frequency
        self.predict_steps = predict_steps
        self.config_seasonality = config_seasonality
        self.config_events = config_events
        self.config_country_holidays = config_country_holidays
        self.config_regressors = config_regressors
        self.config_lagged_regressors = config_lagged_regressors
        self.config_missing = config_missing

        # self.config_args = kwargs
        # self.predict_mode = kwargs["predict_mode"]
        # self.n_lags = kwargs["n_lags"]
        # self.n_forecasts = kwargs["n_forecasts"]
        # self.config_events = kwargs["config_events"]
        # self.config_country_holidays = kwargs["config_country_holidays"]
        # self.config_lagged_regressors = kwargs["config_lagged_regressors"]

        self.max_lags = get_max_num_lags(n_lags=self.n_lags, config_lagged_regressors=self.config_lagged_regressors)

        self.two_level_inputs = [
            "seasonalities",
            "covariates",
            "events",
            "regressors",
        ]

        # Preprocessing of events and holidays features (added to self.df)
        (
            self.df,
            self.additive_event_and_holiday_names,
            self.multiplicative_event_and_holiday_names,
        ) = add_event_features_to_df(
            self.df,
            self.config_events,
            self.config_country_holidays,
        )
        # pre-sort additive/multiplicative regressors
        self.additive_regressors_names, self.multiplicative_regressors_names = sort_regressor_names(
            self.config_regressors
        )

        # Construct index map
        self.sample2index_map, self.length = self.create_sample2index_map(df)

    def __getitem__(self, index):
        """Overrides parent class method to get an item at index.
        Parameters
        ----------
            index : int
                Sample location in dataset, starting at 0, maximum at length-1
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
        OrderedDict
            Meta information: static information about the local dataset
        """
        # Convert dataset sample index to valid dataframe positional index
        # - sample index is any index up to len(dataset)
        # - dataframe positional index is given by position of first target in dataframe for given sample index
        df_index = self.sample_index_to_df_index(index)

        # Tabularize - extract features from dataframe at given target index position
        inputs, target = tabularize_univariate_datetime_single_index(
            df=self.df,
            origin_index=df_index,
            predict_mode=self.predict_mode,
            n_lags=self.n_lags,
            max_lags=self.max_lags,
            n_forecasts=self.n_forecasts,
            config_seasonality=self.config_seasonality,
            config_lagged_regressors=self.config_lagged_regressors,
            additive_event_and_holiday_names=self.additive_event_and_holiday_names,
            multiplicative_event_and_holiday_names=self.multiplicative_event_and_holiday_names,
            additive_regressors_names=self.additive_regressors_names,
            multiplicative_regressors_names=self.multiplicative_regressors_names,
        )
        return inputs, target, self.meta

    def __len__(self):
        """Overrides Parent class method to get data length."""
        return self.length

    def sample_index_to_df_index(self, sample_index):
        """Translates a single outer sample to dataframe index"""
        # Will need more sophisticated mapping for GlobalTimeDataset
        return self.sample2index_map[sample_index]

    def create_sample2index_map(self, df):
        """creates mapping of sample index to corresponding df index at prediction origin.
        (prediction origin: last observation before forecast / future period starts).
        return created mapping to sample2index_map and number of samples.
        """

        # Limit target range due to input lags and number of forecasts
        df_length = len(df)
        origin_start_end_mask = create_origin_start_end_mask(
            df_length=df_length, max_lags=self.max_lags, n_forecasts=self.n_forecasts
        )

        # Prediction Frequency
        # Filter missing samples and prediction frequency (does not actually drop, but creates indexmapping)
        # analogous to `self.filter_samples_after_init(self.kwargs["prediction_frequency"])`
        prediction_frequency_mask = create_prediction_frequency_filter_mask(df, self.prediction_frequency)

        # TODO Create NAN-free index mapping of sample index to df index
        # analogous to `self.drop_nan_after_init(
        # self.df, self.kwargs["predict_steps"], self.kwargs["config_missing"].drop_missing)
        nan_mask = create_nan_mask(
            df, self.predict_steps, self.config_missing.drop_missing
        )  # boolean array where NAN are False

        # Combine masks
        mask = np.logical_and(prediction_frequency_mask, origin_start_end_mask)
        valid_sample_mask = np.logical_and(mask, nan_mask)
        # Convert boolean valid_sample to list of the positinal index of all true/one entries
        #   e.g. [0,0,1,1,0,1,0] -> [2,3,5]
        index_range = np.arange(0, df_length)
        sample_index_2_df_origin_index = index_range[valid_sample_mask]

        num_samples = np.sum(valid_sample_mask)
        assert len(sample_index_2_df_origin_index) == num_samples

        return sample_index_2_df_origin_index, num_samples


class GlobalTimeDataset(TimeDataset):
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
        self.df_names = sorted(list(np.unique(df.loc[:, "ID"].values)))
        # if len(self.df_names) == 1:
        #     super().__init__(df, self.df_names[0], **kwargs)
        # else:
        # raise NotImplementedError
        # timedatasets = [TimeDataset(df_i, df_name, **kwargs) for df_name, df_i in df.groupby("ID")]
        # self.combined_timedataset = [item for timedataset in timedatasets for item in timedataset]
        # self.length = sum(timedataset.length for timedataset in timedatasets)
        self.datasets = OrderedDict({})
        for df_name in self.df_names:
            self.datasets[df_name] = TimeDataset(df[df["ID"] == df_name], df_name, **kwargs)
        self.length = sum(dataset.length for (name, dataset) in self.datasets.items())
        global_sample_to_local_ID = []
        global_sample_to_local_sample = []
        for name, dataset in self.datasets.items():
            global_sample_to_local_ID.append(np.full(shape=dataset.length, fill_value=name))
            global_sample_to_local_sample.append(np.arange(dataset.length))
        self.global_sample_to_local_ID = np.concatenate(global_sample_to_local_ID)
        self.global_sample_to_local_sample = np.concatenate(global_sample_to_local_sample)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Overrides parent class method to get an item at index.
        Parameters
        ----------
            index : int
                Sample location in dataset, starting at 0
        """
        df_name = self.global_sample_to_local_ID[idx]
        local_pos = self.global_sample_to_local_sample[idx]
        return self.datasets[df_name].__getitem__(local_pos)


def tabularize_univariate_datetime_single_index(
    df: pd.DataFrame,
    origin_index: int,
    predict_mode: bool = False,
    n_lags: int = 0,
    max_lags: int = 0,
    n_forecasts: int = 1,
    config_seasonality: Optional[configure.ConfigSeasonality] = None,
    config_lagged_regressors: Optional[configure.ConfigLaggedRegressors] = None,
    # config_events: Optional[configure.ConfigEvents] = None,
    # config_country_holidays=None,
    additive_event_and_holiday_names: list[str] = [],
    multiplicative_event_and_holiday_names: list[str] = [],
    # config_regressors: Optional[configure.ConfigFutureRegressors] = None,
    additive_regressors_names: list[str] = [],
    multiplicative_regressors_names: list[str] = [],
):
    """Create a tabular data sample from timeseries dataframe, used for mini-batch creation.
    Note
    ----
    Data must have no gaps for sample extracted at given index position.
    ----------
        df : pd.DataFrame
            Sequence of observations with original ``ds``, ``y`` and normalized ``t``, ``y_scaled`` columns
        origin_index: int:
            dataframe index position of last observed lag before forecast starts.
        n_forecasts : int
            Number of steps to forecast into future
        n_lags : int
            Number of lagged values of series to include as model inputs (aka AR-order)
        config_seasonality : configure.ConfigSeasonality
            Configuration for seasonalities
        config_lagged_regressors : configure.ConfigLaggedRegressors
            Configurations for lagged regressors
        config_events : configure.ConfigEvents
            User specified events, each with their upper, lower windows (int) and regularization
        config_country_holidays : configure.ConfigCountryHolidays
            Configurations (holiday_names, upper, lower windows, regularization) for country specific holidays
        config_regressors : configure.ConfigFutureRegressors
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
            Targets to be predicted of same length as each of the model inputs, dims: (n_forecasts, 1)
    """
    # TODO: pre-process al type conversions (e.g. torch.float32) in __init__

    # sample features are stored and returned in OrderedDict
    inputs = OrderedDict({})

    if max_lags == 0:
        assert n_forecasts == 1

    if predict_mode:
        targets = torch.zeros((n_forecasts, 1), dtype=torch.float32)
    else:
        if n_forecasts == 1:
            if max_lags == 0:
                targets = df.at[origin_index, "y_scaled"]
            if max_lags > 0:
                targets = df.at[origin_index + 1, "y_scaled"]
            targets = np.expand_dims(targets, 0)
            targets = np.expand_dims(targets, 1)  # extra dimension at end for quantiles:median
        else:
            # Note: df.loc is inclusive of slice end, while df.iloc is not.
            targets = df.loc[origin_index + 1 : origin_index + n_forecasts, "y_scaled"].values
            targets = np.expand_dims(targets, 1)  # extra dimension at end for quantiles:median
        targets = torch.as_tensor(targets, dtype=torch.float32)

    # TIME: the time at each sample's lags and forecasts
    if max_lags == 0:
        inputs["time"] = df.at[origin_index, "t"]
        inputs["time"] = np.expand_dims(inputs["time"], 0)
        inputs["time"] = torch.tensor(inputs["time"], dtype=torch.float32)

    else:
        # extract time value of n_lags steps before  and icluding origin_index and n_forecasts steps after origin_index
        # Note: df.loc is inclusive of slice end, while df.iloc is not.
        inputs["time"] = df.loc[origin_index - n_lags + 1 : origin_index + n_forecasts, "t"].values
        inputs["time"] = torch.as_tensor(inputs["time"], dtype=torch.float32)

    # LAGS: From y-series, extract preceeding n_lags steps up to and including origin_index
    if n_lags >= 1 and "y_scaled" in df.columns:
        # Note: df.loc is inclusive of slice end, while df.iloc is not.
        inputs["lags"] = df.loc[origin_index - n_lags + 1 : origin_index, "y_scaled"].values
        inputs["lags"] = torch.as_tensor(inputs["lags"], dtype=torch.float32)

    # COVARIATES / LAGGED REGRESSORS: Lagged regressor inputs: analogous to LAGS
    if config_lagged_regressors is not None and max_lags > 0:
        lagged_regressors = OrderedDict({})
        # Future TODO: optimize this computation for many lagged_regressors
        for lagged_reg in df.columns:
            if lagged_reg in config_lagged_regressors:
                covar_lags = config_lagged_regressors[lagged_reg].n_lags
                assert covar_lags > 0
                # Note: df.loc is inclusive of slice end, while df.iloc is not.
                lagged_regressors[lagged_reg] = df.loc[origin_index - covar_lags + 1 : origin_index, lagged_reg].values
                lagged_regressors[lagged_reg] = torch.as_tensor(lagged_regressors[lagged_reg], dtype=torch.float32)
        inputs["covariates"] = lagged_regressors

    # SEASONALITIES
    # TODO: precompute and save fourier features and only tabularize / slide windows when calling __getitem__
    if config_seasonality is not None:
        seasonalities = OrderedDict({})
        if max_lags == 0:
            dates = pd.Series(df.at[origin_index, "ds"])
        else:
            # Note: df.loc is inclusive of slice end, while df.iloc is not.
            dates = pd.Series(df.loc[origin_index - n_lags + 1 : origin_index + n_forecasts, "ds"].values)
        # Seasonality features
        for name, period in config_seasonality.periods.items():
            if period.resolution > 0:
                if config_seasonality.computation == "fourier":
                    # Compute Fourier series components with the specified frequency and order.
                    # convert to days since epoch
                    t = np.array((dates - datetime(1900, 1, 1)).dt.total_seconds().astype(np.float32)) / (3600 * 24.0)
                    # features: Matrix with dims (length len(dates), 2*resolution)
                    features = np.column_stack(
                        [np.sin(2.0 * (i + 1) * np.pi * t / period.period) for i in range(period.resolution)]
                        + [np.cos(2.0 * (i + 1) * np.pi * t / period.period) for i in range(period.resolution)]
                    )
                else:
                    raise NotImplementedError
                if period.condition_name is not None:
                    # multiply seasonality features with condition mask/values
                    if max_lags == 0:
                        condition_values = pd.Series(df.at[origin_index, period.condition_name]).values[:, np.newaxis]
                    else:
                        condition_values = df.loc[
                            origin_index - n_lags + 1 : origin_index + n_forecasts, period.condition_name
                        ].values[:, np.newaxis]
                    features = features * condition_values
                seasonalities[name] = torch.as_tensor(features, dtype=torch.float32)
        inputs["seasonalities"] = seasonalities

    # FUTURE REGRESSORS: get the future regressors features
    # create numpy array of values of additive and multiplicative regressors, at correct indexes
    # features dims: (n_forecasts, n_features)
    any_future_regressors = 0 < len(additive_regressors_names + multiplicative_regressors_names)
    if any_future_regressors:  # if config_regressors is not None:
        regressors = OrderedDict({})
        if max_lags == 0:
            if len(additive_regressors_names) > 0:
                features = df.loc[origin_index, additive_regressors_names].values
                regressors["additive"] = torch.as_tensor(
                    np.expand_dims(np.array(features, dtype=np.float32), axis=0), dtype=torch.float32
                )
            if len(multiplicative_regressors_names) > 0:
                features = df.loc[origin_index, multiplicative_regressors_names].values
                regressors["multiplicative"] = torch.as_tensor(
                    np.expand_dims(np.array(features, dtype=np.float32), axis=0), dtype=torch.float32
                )
        else:
            if len(additive_regressors_names) > 0:
                features = df.loc[
                    origin_index + 1 - n_lags : origin_index + n_forecasts, additive_regressors_names
                ].values
                # regressors["additive"] = torch.as_tensor(features, dtype=torch.float32)
                regressors["additive"] = torch.as_tensor(np.array(features, dtype=np.float32), dtype=torch.float32)
            if len(multiplicative_regressors_names) > 0:
                features = df.loc[
                    origin_index + 1 - n_lags : origin_index + n_forecasts, multiplicative_regressors_names
                ].values
                # regressors["multiplicative"] = torch.as_tensor(features, dtype=torch.float32)
                regressors["multiplicative"] = torch.as_tensor(
                    np.array(features, dtype=np.float32), dtype=torch.float32
                )
        inputs["regressors"] = regressors

    # FUTURE EVENTS: get the events features
    # create numpy array of values of additive and multiplicative events, at correct indexes
    # features dims: (n_forecasts, n_features)
    any_events = 0 < len(additive_event_and_holiday_names + multiplicative_event_and_holiday_names)
    if any_events:
        events = OrderedDict({})
        if max_lags == 0:
            # forecasts are at origin_index
            if len(additive_event_and_holiday_names) > 0:
                features = df.loc[origin_index, additive_event_and_holiday_names].values
                events["additive"] = torch.as_tensor(
                    np.expand_dims(np.array(features, dtype=np.float32), axis=0), dtype=torch.float32
                )
            if len(multiplicative_event_and_holiday_names) > 0:
                features = df.loc[origin_index, multiplicative_event_and_holiday_names].values
                events["multiplicative"] = torch.as_tensor(
                    np.expand_dims(np.array(features, dtype=np.float32), axis=0), dtype=torch.float32
                )
        else:
            # forecasts are at origin_index + 1 up to origin_index + n_forecasts
            if len(additive_event_and_holiday_names) > 0:
                features = df.loc[
                    origin_index + 1 - n_lags : origin_index + n_forecasts, additive_event_and_holiday_names
                ].values
                events["additive"] = torch.as_tensor(np.array(features, dtype=np.float32), dtype=torch.float32)

            if len(multiplicative_event_and_holiday_names) > 0:
                features = df.loc[
                    origin_index + 1 - n_lags : origin_index + n_forecasts, multiplicative_event_and_holiday_names
                ].values
                events["multiplicative"] = torch.as_tensor(np.array(features, dtype=np.float32), dtype=torch.float32)
        inputs["events"] = events

    # ONLY FOR DEBUGGING
    # tabularized_input_shapes_str = ""
    # for key, value in inputs.items():
    #     if key in [
    #         "seasonalities",
    #         "covariates",
    #         "events",
    #         "regressors",
    #     ]:
    #         for name, period_features in value.items():
    #             tabularized_input_shapes_str += f"    {name} {key} {period_features}\n"
    #     else:
    #         tabularized_input_shapes_str += f"    {key} {value.shape} \n"
    # log.debug(f"Tabularized inputs shapes: \n{tabularized_input_shapes_str}")
    return inputs, targets


def fourier_series(dates, period, series_order):
    """Provides Fourier series components with the specified frequency and order.
    Note
    ----
    Identical to OG Prophet.
    Parameters
    ----------
        dates : pd.Series
            Containing time stamps
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
    t = np.array((dates - datetime(1970, 1, 1)).dt.total_seconds().astype(np.float32)) / (3600 * 24.0)
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


def make_country_specific_holidays_dict(year_list, country):
    """
    Make dataframe of country specific holidays for given years and countries
    Parameters
    ----------
        year_list : list
            List of years
        country : str, list
            List of country names
    Returns
    -------
        pd.DataFrame
            Containing country specific holidays df with columns 'ds' and 'holiday'
    """
    # iterate over countries and get holidays for each country
    # convert to list if not already
    if isinstance(country, str):
        country = [country]
    country_specific_holidays = {}
    for single_country in country:
        single_country_specific_holidays = get_country_holidays(single_country, year_list)
        # only add holiday if it is not already in the dict
        country_specific_holidays.update(single_country_specific_holidays)
    country_specific_holidays_dict = defaultdict(list)
    for date, holiday in country_specific_holidays.items():
        country_specific_holidays_dict[holiday].append(pd.to_datetime(date))
    return country_specific_holidays_dict


def get_event_offset_features(event, config, feature):
    """
    Create event offset features for the given event, config and feature
    Parameters
    ----------
        event : str
            Name of the event
        config : configure.ConfigEvents
            User specified events, holidays, and country specific holidays
        feature : pd.Series
            Feature for the event
    Returns
    -------
        tuple
            Tuple of additive_events and multiplicative_events
    """
    events = pd.DataFrame({})
    lw = config.lower_window
    uw = config.upper_window
    for offset in range(lw, uw + 1):
        key = utils.create_event_names_for_offsets(event, offset)
        offset_feature = feature.shift(periods=offset, fill_value=0.0)
        events[key] = offset_feature
    return events


def _create_event_offset_features(event, config, feature, additive_events, multiplicative_events):
    """
    Create event offset features for the given event, config and feature
    Parameters
    ----------
        event : str
            Name of the event
        config : configure.ConfigEvents
            User specified events, holidays, and country specific holidays
        feature : pd.Series
            Feature for the event
        additive_events : pd.DataFrame
            Dataframe of additive events
        multiplicative_events : pd.DataFrame
            Dataframe of multiplicative events
    Returns
    -------
        tuple
            Tuple of additive_events and multiplicative_events
    """
    lw = config.lower_window
    uw = config.upper_window
    mode = config.mode
    for offset in range(lw, uw + 1):
        key = utils.create_event_names_for_offsets(event, offset)
        offset_feature = feature.shift(periods=offset, fill_value=0.0)
        if mode == "additive":
            additive_events[key] = offset_feature
        else:
            multiplicative_events[key] = offset_feature


def add_event_features_to_df(
    df,
    config_events: Optional[configure.ConfigEvents] = None,
    config_country_holidays: Optional[configure.ConfigCountryHolidays] = None,
):
    """
    Construct columns containing the features of each event, added to df.
    Parameters
    ----------
        df : pd.DataFrame
            Dataframe with all values including the user specified events (provided by user)
        config_events : configure.ConfigEvents
            User specified events, each with their upper, lower windows (int), regularization
        config_country_holidays : configure.ConfigCountryHolidays
            Configurations (holiday_names, upper, lower windows, regularization) for country specific holidays
    Returns
    -------
        np.array
            All additive event features (both user specified and country specific)
        np.array
            All multiplicative event features (both user specified and country specific)
    """
    # create all additional user specified offest events
    additive_events_names = []
    multiplicative_events_names = []
    if config_events is not None:
        for event in sorted(list(config_events.keys())):
            feature = df[event]
            config = config_events[event]
            mode = config.mode
            for offset in range(config.lower_window, config.upper_window + 1):
                event_offset_name = utils.create_event_names_for_offsets(event, offset)
                df[event_offset_name] = feature.shift(periods=offset, fill_value=0.0)
                if mode == "additive":
                    additive_events_names.append(event_offset_name)
                else:
                    multiplicative_events_names.append(event_offset_name)

    # create all country specific holidays and their offsets.
    additive_holiday_names = []
    multiplicative_holiday_names = []
    if config_country_holidays is not None:
        year_list = list({x.year for x in df.ds})
        country_holidays_dict = make_country_specific_holidays_dict(year_list, config_country_holidays.country)
        config = config_country_holidays
        mode = config.mode
        for holiday in config_country_holidays.holiday_names:
            # feature = pd.Series([0.0] * df.shape[0])
            feature = pd.Series(np.zeros(df.shape[0], dtype=np.float32))
            if holiday in country_holidays_dict.keys():
                dates = country_holidays_dict[holiday]
                feature[df.ds.isin(dates)] = 1.0
            else:
                raise ValueError(f"Holiday {holiday} not found in country holidays")
            for offset in range(config.lower_window, config.upper_window + 1):
                holiday_offset_name = utils.create_event_names_for_offsets(holiday, offset)
                df[holiday_offset_name] = feature.shift(periods=offset, fill_value=0.0)
                if mode == "additive":
                    additive_holiday_names.append(holiday_offset_name)
                else:
                    multiplicative_holiday_names.append(holiday_offset_name)
    # Future TODO: possibly undo merge of events and holidays.
    additive_event_and_holiday_names = sorted(additive_events_names + additive_holiday_names)
    multiplicative_event_and_holiday_names = sorted(multiplicative_events_names + multiplicative_holiday_names)
    return df, additive_event_and_holiday_names, multiplicative_event_and_holiday_names


def make_events_features(df, config_events: Optional[configure.ConfigEvents] = None, config_country_holidays=None):
    """
    Construct arrays of all event features
    Parameters
    ----------
        df : pd.DataFrame
            Dataframe with all values including the user specified events (provided by user)
        config_events : configure.ConfigEvents
            User specified events, each with their upper, lower windows (int), regularization
        config_country_holidays : configure.ConfigCountryHolidays
            Configurations (holiday_names, upper, lower windows, regularization) for country specific holidays
    Returns
    -------
        np.array
            All additive event features (both user specified and country specific)
        np.array
            All multiplicative event features (both user specified and country specific)
    """
    df = df.reset_index(drop=True)
    additive_events = pd.DataFrame()
    multiplicative_events = pd.DataFrame()

    # create all user specified events
    if config_events is not None:
        for event, configs in config_events.items():
            feature = df[event]
            _create_event_offset_features(event, configs, feature, additive_events, multiplicative_events)

    # create all country specific holidays
    if config_country_holidays is not None:
        year_list = list({x.year for x in df.ds})
        country_holidays_dict = make_country_specific_holidays_dict(year_list, config_country_holidays.country)
        for holiday in config_country_holidays.holiday_names:
            feature = pd.Series([0.0] * df.shape[0])
            if holiday in country_holidays_dict.keys():
                dates = country_holidays_dict[holiday]
                feature[df.ds.isin(dates)] = 1.0
            _create_event_offset_features(
                holiday, config_country_holidays, feature, additive_events, multiplicative_events
            )

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


# def make_regressors_features(df, config_regressors):
#     """Construct arrays of all scalar regressor features
#     Parameters
#     ----------
#         df : pd.DataFrame
#             Dataframe with all values including the user specified regressors
#         config_regressors : configure.ConfigFutureRegressors
#             User specified regressors config
#     Returns
#     -------
#         np.array
#             All additive regressor features
#         np.array
#             All multiplicative regressor features
#     """
#     additive_regressors = pd.DataFrame()
#     multiplicative_regressors = pd.DataFrame()

#     for reg in df.columns:
#         if reg in config_regressors:
#             mode = config_regressors[reg].mode
#             if mode == "additive":
#                 additive_regressors[reg] = df[reg]
#             else:
#                 multiplicative_regressors[reg] = df[reg]

#     if not additive_regressors.empty:
#         additive_regressors = additive_regressors[sorted(additive_regressors.columns.tolist())]
#         additive_regressors = additive_regressors.values
#     else:
#         additive_regressors = None
#     if not multiplicative_regressors.empty:
#         multiplicative_regressors = multiplicative_regressors[sorted(multiplicative_regressors.columns.tolist())]
#         multiplicative_regressors = multiplicative_regressors.values
#     else:
#         multiplicative_regressors = None

#     return additive_regressors, multiplicative_regressors


# def seasonal_features_from_dates(df, config_seasonality: configure.ConfigSeasonality):
#     """Dataframe with seasonality features.
#     Includes seasonality features
#     Parameters
#     ----------
#         df : pd.DataFrame
#             Dataframe with all values
#         config_seasonality : configure.ConfigSeasonality
#             Configuration for seasonalities
#     Returns
#     -------
#         OrderedDict
#             Dictionary with keys for each period name containing an np.array
#             with the respective regression features. each with dims: (len(dates), 2*fourier_order)
#     """
#     dates = df["ds"]
#     assert len(dates.shape) == 1
#     seasonalities = OrderedDict({})
#     # Seasonality features
#     for name, period in config_seasonality.periods.items():
#         if period.resolution > 0:
#             if config_seasonality.computation == "fourier":
#                 features = fourier_series(
#                     dates=dates,
#                     period=period.period,
#                     series_order=period.resolution,
#                 )
#             else:
#                 raise NotImplementedError
#             if period.condition_name is not None:
#                 features = features * df[period.condition_name].values[:, np.newaxis]
#             seasonalities[name] = features
#     return seasonalities


def create_origin_start_end_mask(df_length, max_lags, n_forecasts):
    """Creates a boolean mask for valid prediction origin positions.
    (based on limiting input lags and forecast targets at start and end of df)"""
    if max_lags >= 1:
        start_pad = np.zeros(max_lags - 1, dtype=bool)
        valid_targets = np.ones(df_length - max_lags - n_forecasts + 1, dtype=bool)
        end_pad = np.zeros(n_forecasts, dtype=bool)
        target_start_end_mask = np.concatenate((start_pad, valid_targets, end_pad), axis=None)
    elif max_lags == 0 and n_forecasts == 1:
        # without lags, forecast targets and origins are identical
        target_start_end_mask = np.ones(df_length, dtype=bool)
    else:
        raise ValueError(f"max_lags value of {max_lags} not supported for n_forecasts {n_forecasts}.")
    return target_start_end_mask


def create_prediction_frequency_filter_mask(df: pd.DataFrame, prediction_frequency=None):
    """Filters prediction origin index from df based on the forecast frequency setting.

    Filter based on timestamp last lag before targets start

    Parameters
    ----------
        prediction_frequency : int
            periodic interval in which forecasts should be made.
        Note
        ----
        E.g. if prediction_frequency=7, forecasts are only made on every 7th step (once in a week in case of daily
        resolution).

    Returns boolean mask where prediction origin indexes to be included are True, and the rest False.
    """
    # !! IMPORTANT
    # TODO: Adjust top level documentation to specify that the filter is applied to prediction ORIGIN, not targets start.
    # !! IMPORTANT

    mask = np.ones((len(df),), dtype=bool)

    # Basic case: no filter
    if prediction_frequency is None or prediction_frequency == 1:
        return mask

    # OLD: timestamps were created from "ds" column in tabularization and then re-converted here
    # timestamps = pd.to_datetime([x["timestamps"][0] for x in df])
    # OR
    # timestamps = df["timestamps"].apply(lambda x: pd.to_datetime(x[0]))

    timestamps = pd.to_datetime(df.loc[:, "ds"])
    filter_masks = []
    for key, value in prediction_frequency.items():
        if key == "daily-hour":
            mask = timestamps.dt.hour == value
        elif key == "weekly-day":
            mask = timestamps.dt.dayofweek == value
        elif key == "monthly-day":
            mask = timestamps.dt.day == value
        elif key == "yearly-month":
            mask = timestamps.dt.month == value
        elif key == "hourly-minute":
            mask = timestamps.dt.minute == value
        else:
            raise ValueError(f"Invalid prediction frequency: {key}")
        filter_masks.append(mask)
    for m in filter_masks:
        mask = np.logical_and(mask, m)
    return mask


def create_nan_mask(df, predict_steps, drop_missing):
    """Creates mask for each prediction origin,
    accounting for corresponding input lags / forecast targets containing any NaN values.

    Parameters
    ----------
        drop_missing : bool
            whether to automatically drop missing samples from the data
        predict_steps : int
            number of steps to predict
    """
    # IMPORTANT !!
    # TODO implement actual filtering
    return np.ones(len(df), dtype=bool)

    # Create index mapping of sample index to df index
    # - Filter missing samples (does not actually drop, but creates indexmapping)
    # -- drop nan analogous to `self.drop_nan_after_init(self.df, self.kwargs["predict_steps"], self.kwargs["config_missing"].drop_missing)
    # Note: needs to also account for NANs in lagged inputs or in n_forecasts, not just first target.
    # Implement a convolutional filter for targets and each lagged regressor.
    # Also account for future regressors and events.

    # Rewrite to return mask instead of filtering df:
    nan_idx = []
    # NaNs in inputs
    for key, data in self.inputs.items():
        if isinstance(data, torch.Tensor):
            nans = torch.where(torch.isnan(data))[0].tolist()
            if len(nans) > 0:
                nan_idx += nans
        elif isinstance(data, dict):
            for subkey, subdata in data.items():
                nans = torch.where(torch.isnan(subdata))[0].tolist()
                if len(nans) > 0:
                    nan_idx += nans

    # NaNs in targets that are not inserted for prediction at the end
    nans = torch.where(torch.isnan(self.targets))[0].tolist()
    if len(nans) > 0:
        for idx in nans:
            if idx not in nan_idx and idx < len(self) - predict_steps:
                nan_idx.append(idx)

    nan_idx = list(set(nan_idx))
    nan_idx.sort()
    if drop_missing and len(nan_idx) > 0:
        log.warning(f"{len(nan_idx)} samples with missing values were dropped from the data. ")
        for key, data in self.inputs.items():
            if key not in ["time", "lags"]:  # "time_lagged"
                for name, features in data.items():
                    self.inputs[key][name] = np.delete(self.inputs[key][name], nan_idx, 0)
            else:
                self.inputs[key] = np.delete(self.inputs[key], nan_idx, 0)
        self.targets = np.delete(self.targets, nan_idx, 0)
        self.length = self.inputs["time"].shape[0]
    if not drop_missing and len(nan_idx) > 0:
        raise ValueError(
            "Inputs/targets with missing values detected. "
            "Please either adjust imputation parameters, or set 'drop_missing' to True to drop those samples."
        )


def sort_regressor_names(config):
    additive_regressors_names = []
    multiplicative_regressors_names = []
    if config is not None:
        # sort and divide regressors into multiplicative and additive
        additive_regressors_names = []
        multiplicative_regressors_names = []
        for reg in sorted(list(config.keys())):
            mode = config[reg].mode
            if mode == "additive":
                additive_regressors_names.append(reg)
            else:
                multiplicative_regressors_names.append(reg)
    return additive_regressors_names, multiplicative_regressors_names


# ## TODO: rename - used elsewhere, not in this file.
# def make_country_specific_holidays_df(year_list, country):
#     return make_country_specific_holidays_dict(year_list, country)


# def split_nested_dict(inputs):
#     """Split nested dict into list of dicts.
#     Parameters
#     ----------
#         inputs : ordered dict
#             Nested dict to be split.
#     Returns
#     -------
#         list of dicts
#             List of dicts with same keys as inputs.
#     """

#     def split_dict(inputs, index):
#         return {k: v[index] if not isinstance(v, dict) else split_dict(v, index) for k, v in inputs.items()}

#     length = next(iter(inputs.values())).shape[0]
#     return [split_dict(inputs, i) for i in range(length)]

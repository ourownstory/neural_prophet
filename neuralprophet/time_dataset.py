import logging
from collections import OrderedDict
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data.dataset import Dataset

from neuralprophet import configure, utils
from neuralprophet.df_utils import get_max_num_lags
from neuralprophet.event_utils import get_all_holidays

log = logging.getLogger("NP.time_dataset")


class TimeDataset(Dataset):
    """Create a PyTorch dataset of a tabularized time-series"""

    def __init__(
        self,
        df,
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
        """
        # Outcome after a call to init (summary):
        # - add events and holidays columns to df
        # - calculated the number of usable samples (accounting for nan and filters)
        # - creates mapping of sample index to df index

        # Context Notes
        # Currently done to df before it arrives here:
        # -> fit calls prep_or_copy_df, _check_dataframe, and _handle_missing_data, passes to _train
        # -> _train calls prep_or_copy_df, then passes to init_train_loader, which returns the train_loader
        # -> init_train_loader calls prep_or_copy_df, _normalize, _create_dataset (returns TimeDataset), returns dataset wrapped in DataLoader
        # ->_create_dataset calls prep_or_copy_df, then returns GlobalTimeDataset
        # Future TODO: integrate some of these preprocessing steps happening outside?

        self.df = df.reset_index(drop=True)  # Needed for index based operations in __getitem__
        if "index" in list(self.df.columns):  # should not be the case
            self.df = self.df.drop("index", axis=1)
        df_names = list(np.unique(df.loc[:, "ID"].values))
        assert len(df_names) == 1
        assert isinstance(df_names[0], str)
        self.df_name = df_names[0]

        self.meta = OrderedDict({})
        self.meta["df_name"] = self.df_name

        self.predict_mode = predict_mode
        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self.prediction_frequency = prediction_frequency
        self.predict_steps = predict_steps  # currently unused
        self.config_seasonality = config_seasonality
        self.config_events = config_events
        self.config_country_holidays = config_country_holidays
        self.config_regressors = config_regressors
        self.config_lagged_regressors = config_lagged_regressors
        self.config_missing = config_missing

        self.max_lags = get_max_num_lags(n_lags=self.n_lags, config_lagged_regressors=self.config_lagged_regressors)
        if self.max_lags == 0:
            assert self.n_forecasts == 1
        self.two_level_inputs = ["seasonalities", "covariates", "events", "regressors"]

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

        # self.tensor_data = torch.tensor(df.values, dtype=torch.float32
        self.df["ds"] = self.df["ds"].astype(int) // 10**9  # Convert to Unix timestamp in seconds
        self.tensor_dict = {
            col: torch.tensor(self.df[col].values, dtype=torch.float32) for col in self.df if self.df[col].dtype != "O"
        }

        # Construct index map
        self.sample2index_map, self.length = self.create_sample2index_map(self.df, self.tensor_dict)

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
            tensor_dict=self.tensor_dict,
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
        return self.sample2index_map[sample_index]

    def create_sample2index_map(self, df, tensor_dict):
        """creates mapping of sample index to corresponding df index at prediction origin.
        (prediction origin: last observation before forecast / future period starts).
        return created mapping to sample2index_map and number of samples.
        """

        # Limit target range due to input lags and number of forecasts
        df_length = len(tensor_dict["ds"])
        origin_start_end_mask = create_origin_start_end_mask(
            df_length=df_length, max_lags=self.max_lags, n_forecasts=self.n_forecasts
        )

        # Prediction Frequency
        # Filter missing samples and prediction frequency (does not actually drop, but creates indexmapping)
        prediction_frequency_mask = create_prediction_frequency_filter_mask(
            tensor_dict["ds"], self.prediction_frequency
        )

        # Combine prediction origin masks
        valid_prediction_mask = prediction_frequency_mask & origin_start_end_mask

        # Create NAN-free index mapping of sample index to df index
        nan_mask = create_nan_mask(
            tensor_dict=tensor_dict,
            predict_mode=self.predict_mode,
            max_lags=self.max_lags,
            n_lags=self.n_lags,
            n_forecasts=self.n_forecasts,
            config_lagged_regressors=self.config_lagged_regressors,
            future_regressor_names=self.additive_regressors_names + self.multiplicative_regressors_names,
            event_names=self.additive_event_and_holiday_names + self.multiplicative_event_and_holiday_names,
        )  # boolean array where NAN are False

        # Filter NAN
        valid_sample_mask = valid_prediction_mask & nan_mask
        print(f"valid_prediction_mask = {valid_prediction_mask}")
        print(f"nan_mask = {nan_mask}")
        n_clean_data_samples = valid_prediction_mask.sum().item()
        n_real_data_samples = valid_sample_mask.sum().item()
        nan_samples_to_drop = n_clean_data_samples - n_real_data_samples
        if nan_samples_to_drop > 0 and not self.config_missing.drop_missing:
            raise ValueError(
                f"NANs found. {nan_samples_to_drop} samples affected. Set `drop_missing` to `True` to drop these samples."
            )

        # Convert boolean valid_sample to list of the positinal index of all true/one entries
        #   e.g. [0,0,1,1,0,1,0] -> [2,3,5]
        sample_index_2_df_origin_index = torch.arange(0, df_length)[valid_sample_mask]

        num_samples = sample_index_2_df_origin_index.size(0)
        assert num_samples == n_real_data_samples

        return sample_index_2_df_origin_index, num_samples


class GlobalTimeDataset(TimeDataset):
    def __init__(
        self,
        df,
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
                dataframe containing column ``ds``, ``y``, and optionally``ID`` and
                normalized columns normalized columns ``ds``, ``y``, ``t``, ``y_scaled``

        """
        self.df_names = sorted(list(np.unique(df.loc[:, "ID"].values)))
        self.datasets = OrderedDict({})
        for df_name in self.df_names:
            self.datasets[df_name] = TimeDataset(
                df=df[df["ID"] == df_name],
                predict_mode=predict_mode,
                n_lags=n_lags,
                n_forecasts=n_forecasts,
                prediction_frequency=prediction_frequency,
                predict_steps=predict_steps,
                config_seasonality=config_seasonality,
                config_events=config_events,
                config_country_holidays=config_country_holidays,
                config_regressors=config_regressors,
                config_lagged_regressors=config_lagged_regressors,
                config_missing=config_missing,
            )
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


def get_sample_targets(tensor_dict, origin_index, n_forecasts, max_lags, predict_mode):
    if predict_mode:
        return torch.zeros((n_forecasts, 1), dtype=torch.float32)
    else:
        if n_forecasts == 1:
            if max_lags == 0:
                targets = tensor_dict["y_scaled"][origin_index]
            if max_lags > 0:
                targets = tensor_dict["y_scaled"][origin_index + 1]
            targets = targets.unsqueeze(0).unsqueeze(1)
        else:
            targets = tensor_dict["y_scaled"][origin_index + 1 : origin_index + n_forecasts + 1]
            targets = targets.unsqueeze(1)
        return targets


def get_sample_lagged_regressors(tensor_dict, origin_index, config_lagged_regressors):
    lagged_regressors = OrderedDict({})
    # Future TODO: optimize this computation for many lagged_regressors
    for name, lagged_regressor in config_lagged_regressors.items():
        covar_lags = lagged_regressor.n_lags
        assert covar_lags > 0
        # Indexing tensors instead of DataFrame
        lagged_regressors[name] = tensor_dict[name][origin_index - covar_lags + 1 : origin_index + 1]
    return lagged_regressors


def get_sample_seasonalities(tensor_dict, origin_index, n_forecasts, max_lags, n_lags, config_seasonality):
    seasonalities = OrderedDict({})
    if max_lags == 0:
        dates = tensor_dict["ds"][origin_index].unsqueeze(0)
    else:
        dates = tensor_dict["ds"][origin_index - n_lags + 1 : origin_index + n_forecasts + 1]

    for name, period in config_seasonality.periods.items():
        if period.resolution > 0:
            if config_seasonality.computation == "fourier":
                t = (dates - datetime(1900, 1, 1).timestamp()).float() / (3600 * 24.0)
                features = torch.cat(
                    [
                        torch.sin(2.0 * (i + 1) * np.pi * t / period.period).unsqueeze(1)
                        for i in range(period.resolution)
                    ]
                    + [
                        torch.cos(2.0 * (i + 1) * np.pi * t / period.period).unsqueeze(1)
                        for i in range(period.resolution)
                    ],
                    dim=1,
                )
            else:
                raise NotImplementedError

            if period.condition_name is not None:
                if max_lags == 0:
                    condition_values = tensor_dict[period.condition_name][origin_index].unsqueeze(0).unsqueeze(1)
                else:
                    condition_values = tensor_dict[period.condition_name][
                        origin_index - n_lags + 1 : origin_index + n_forecasts + 1
                    ].unsqueeze(1)
                features = features * condition_values
            seasonalities[name] = features
    return seasonalities


def get_sample_future_regressors(
    tensor_dict, origin_index, n_forecasts, max_lags, n_lags, additive_regressors_names, multiplicative_regressors_names
):
    regressors = OrderedDict({})
    if max_lags == 0:
        if len(additive_regressors_names) > 0:
            features = torch.stack(
                [tensor_dict[name][origin_index].unsqueeze(0) for name in additive_regressors_names], dim=1
            )
            regressors["additive"] = features
        if len(multiplicative_regressors_names) > 0:
            features = torch.stack(
                [tensor_dict[name][origin_index].unsqueeze(0) for name in multiplicative_regressors_names], dim=1
            )
            regressors["multiplicative"] = features
    else:
        if len(additive_regressors_names) > 0:
            features = torch.stack(
                [
                    tensor_dict[name][origin_index + 1 - n_lags : origin_index + n_forecasts + 1]
                    for name in additive_regressors_names
                ],
                dim=1,
            )
            regressors["additive"] = features
        if len(multiplicative_regressors_names) > 0:
            features = torch.stack(
                [
                    tensor_dict[name][origin_index + 1 - n_lags : origin_index + n_forecasts + 1]
                    for name in multiplicative_regressors_names
                ],
                dim=1,
            )
            regressors["multiplicative"] = features
    return regressors


def get_sample_future_events(
    tensor_dict,
    origin_index,
    n_forecasts,
    max_lags,
    n_lags,
    additive_event_and_holiday_names,
    multiplicative_event_and_holiday_names,
):
    events = OrderedDict({})
    if max_lags == 0:
        if len(additive_event_and_holiday_names) > 0:
            features = torch.stack(
                [tensor_dict[name][origin_index].unsqueeze(0) for name in additive_event_and_holiday_names], dim=1
            )
            events["additive"] = features
        if len(multiplicative_event_and_holiday_names) > 0:
            features = torch.stack(
                [tensor_dict[name][origin_index].unsqueeze(0) for name in multiplicative_event_and_holiday_names], dim=1
            )
            events["multiplicative"] = features
    else:
        if len(additive_event_and_holiday_names) > 0:
            features = torch.stack(
                [
                    tensor_dict[name][origin_index + 1 - n_lags : origin_index + n_forecasts + 1]
                    for name in additive_event_and_holiday_names
                ],
                dim=1,
            )
            events["additive"] = features
        if len(multiplicative_event_and_holiday_names) > 0:
            features = torch.stack(
                [
                    tensor_dict[name][origin_index + 1 - n_lags : origin_index + n_forecasts + 1]
                    for name in multiplicative_event_and_holiday_names
                ],
                dim=1,
            )
            events["multiplicative"] = features
    return events


def log_input_shapes(inputs):
    tabularized_input_shapes_str = ""
    for key, value in inputs.items():
        if key in [
            "seasonalities",
            "covariates",
            "events",
            "regressors",
        ]:
            for name, period_features in value.items():
                tabularized_input_shapes_str += f"    {name} {key} {period_features.shape}\n"
        else:
            tabularized_input_shapes_str += f"    {key} {value.shape} \n"
    log.debug(f"Tabularized inputs shapes: \n{tabularized_input_shapes_str}")


def tabularize_univariate_datetime_single_index(
    tensor_dict: dict,
    origin_index: int,
    predict_mode: bool = False,
    n_lags: int = 0,
    max_lags: int = 0,
    n_forecasts: int = 1,
    config_seasonality: Optional[configure.ConfigSeasonality] = None,
    config_lagged_regressors: Optional[configure.ConfigLaggedRegressors] = None,
    additive_event_and_holiday_names: List[str] = [],
    multiplicative_event_and_holiday_names: List[str] = [],
    additive_regressors_names: List[str] = [],
    multiplicative_regressors_names: List[str] = [],
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
    # TODO: pre-process all type conversions (e.g. torch.float32) in __init__
    # Note: if max_lags == 0, then n_forecasts == 1

    # sample features are stored and returned in OrderedDict
    inputs = OrderedDict({})

    targets = get_sample_targets(
        tensor_dict=tensor_dict,
        origin_index=origin_index,
        n_forecasts=n_forecasts,
        max_lags=max_lags,
        predict_mode=predict_mode,
    )

    # TIME: the time at each sample's lags and forecasts
    if max_lags == 0:
        t = tensor_dict["t"][origin_index]
        inputs["time"] = t.unsqueeze(0)
    else:
        # extract time value of n_lags steps before  and icluding origin_index and n_forecasts steps after origin_index
        # Note: df.loc is inclusive of slice end, while df.iloc is not.
        t = tensor_dict["t"][origin_index - n_lags + 1 : origin_index + n_forecasts + 1]
        inputs["time"] = t

    # LAGS: From y-series, extract preceeding n_lags steps up to and including origin_index
    if n_lags >= 1 and "y_scaled" in tensor_dict:
        # Note: df.loc is inclusive of slice end, while df.iloc is not.
        lags = tensor_dict["y_scaled"][origin_index - n_lags + 1 : origin_index + 1]
        inputs["lags"] = lags

    # COVARIATES / LAGGED REGRESSORS: Lagged regressor inputs: analogous to LAGS
    if config_lagged_regressors is not None:  # and max_lags > 0:
        inputs["covariates"] = get_sample_lagged_regressors(
            tensor_dict=tensor_dict, origin_index=origin_index, config_lagged_regressors=config_lagged_regressors
        )

    # SEASONALITIES_
    if config_seasonality is not None:
        inputs["seasonalities"] = get_sample_seasonalities(
            tensor_dict=tensor_dict,
            origin_index=origin_index,
            n_forecasts=n_forecasts,
            max_lags=max_lags,
            n_lags=n_lags,
            config_seasonality=config_seasonality,
        )

    # FUTURE REGRESSORS: get the future regressors features
    # create numpy array of values of additive and multiplicative regressors, at correct indexes
    # features dims: (n_forecasts, n_features)
    any_future_regressors = 0 < len(additive_regressors_names + multiplicative_regressors_names)
    if any_future_regressors:  # if config_regressors.regressors is not None:
        inputs["regressors"] = get_sample_future_regressors(
            tensor_dict=tensor_dict,
            origin_index=origin_index,
            n_forecasts=n_forecasts,
            max_lags=max_lags,
            n_lags=n_lags,
            additive_regressors_names=additive_regressors_names,
            multiplicative_regressors_names=multiplicative_regressors_names,
        )

    # FUTURE EVENTS: get the events features
    # create numpy array of values of additive and multiplicative events, at correct indexes
    # features dims: (n_forecasts, n_features)
    any_events = 0 < len(additive_event_and_holiday_names + multiplicative_event_and_holiday_names)
    if any_events:
        inputs["events"] = get_sample_future_events(
            tensor_dict=tensor_dict,
            origin_index=origin_index,
            n_forecasts=n_forecasts,
            max_lags=max_lags,
            n_lags=n_lags,
            additive_event_and_holiday_names=additive_event_and_holiday_names,
            multiplicative_event_and_holiday_names=multiplicative_event_and_holiday_names,
        )

    # ONLY FOR DEBUGGING
    # if log.level == 0:
    #     log_input_shapes(inputs)
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

    def normalize_holiday_name(name):
        # Handle cases like "Independence Day (observed)" -> "Independence Day"
        if "(observed)" in name:
            return name.replace(" (observed)", "")
        return name

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
        country_holidays_dict = get_all_holidays(year_list, config_country_holidays.country)
        config = config_country_holidays
        mode = config.mode
        for holiday in config_country_holidays.holiday_names:
            feature = pd.Series(np.zeros(df.shape[0], dtype=np.float32))
            holiday = normalize_holiday_name(holiday)
            if holiday in country_holidays_dict.keys():
                dates = country_holidays_dict[holiday]
                feature[df.ds.isin(dates)] = 1.0
            else:
                raise ValueError(f"Holiday {holiday} not found in {config_country_holidays.country} holidays")
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


def create_origin_start_end_mask(df_length, max_lags, n_forecasts):
    """Creates a boolean mask for valid prediction origin positions.
    (based on limiting input lags and forecast targets at start and end of df)"""
    if max_lags >= 1:
        start_pad = torch.zeros(max_lags - 1, dtype=torch.bool)
        valid_targets = torch.ones(df_length - max_lags - n_forecasts + 1, dtype=torch.bool)
        end_pad = torch.zeros(n_forecasts, dtype=torch.bool)
        target_start_end_mask = torch.cat((start_pad, valid_targets, end_pad), dim=0)
    elif max_lags == 0 and n_forecasts == 1:
        # without lags, forecast targets and origins are identical
        target_start_end_mask = torch.ones(df_length, dtype=torch.bool)
    else:
        raise ValueError(f"max_lags value of {max_lags} not supported for n_forecasts {n_forecasts}.")
    return target_start_end_mask


def create_prediction_frequency_filter_mask(timestamps, prediction_frequency=None):
    """Filters prediction origin index from df based on the forecast frequency setting.

    Filter based on timestamp last lag before targets start

    Parameters
    ----------
        timestamps : torch.Tensor
            Tensor of timestamps in Unix epoch format
        prediction_frequency : dict
            periodic interval in which forecasts should be made.
        Note
        ----
        E.g. if prediction_frequency=7, forecasts are only made on every 7th step (once in a week in case of daily
        resolution).

    Returns boolean mask where prediction origin indexes to be included are True, and the rest False.
    """
    mask = torch.ones(len(timestamps), dtype=torch.bool)

    # Basic case: no filter
    if prediction_frequency is None:
        return mask
    else:
        assert isinstance(prediction_frequency, dict)

    timestamps = pd.to_datetime(timestamps.numpy(), unit="s")
    filter_masks = []
    for key, value in prediction_frequency.items():
        if key == "hourly-minute":
            filter_mask = timestamps.minute == value
        elif key == "daily-hour":
            filter_mask = timestamps.hour == value
        elif key == "weekly-day":
            filter_mask = timestamps.dayofweek == value
        elif key == "monthly-day":
            filter_mask = timestamps.day == value
        elif key == "yearly-month":
            filter_mask = timestamps.month == value
        else:
            raise ValueError(f"Invalid prediction frequency: {key}")
        filter_masks.append(filter_mask)

    combined_mask = filter_masks[0]
    for m in filter_masks[1:]:
        combined_mask = combined_mask & m
    return torch.tensor(combined_mask, dtype=torch.bool)


def create_nan_mask(
    tensor_dict,
    predict_mode,
    max_lags,
    n_lags,
    n_forecasts,
    config_lagged_regressors,
    future_regressor_names,
    event_names,
):
    """Creates mask for each prediction origin,
    accounting for corresponding input lags / forecast targets containing any NaN values.
    """
    tensor_length = len(tensor_dict["ds"])
    valid_origins = torch.ones(tensor_length, dtype=torch.bool)
    tensor_isna = {k: torch.isnan(v) for k, v in tensor_dict.items()}

    # TARGETS
    if predict_mode:
        # Targets not needed
        targets_valid = torch.ones(tensor_length, dtype=torch.bool)
    else:
        if max_lags == 0:  # y-series and origin index match
            targets_valid = ~tensor_isna["y_scaled"]
        else:
            if n_forecasts == 1:
                targets_nan = tensor_isna["y_scaled"][1:]
                targets_nan = torch.cat([targets_nan, torch.tensor([True], dtype=torch.bool)])
                targets_valid = ~targets_nan
            else:  # This is also correct for n_forecasts == 1, but slower.
                targets_nan = sliding_window_view(tensor_isna["y_scaled"], window_shape=n_forecasts).any(axis=-1)
                # first entry corresponds to origin_index -1, drop this.
                targets_nan = torch.tensor(targets_nan[1:])
                # pad last n_forecasts as missing, as forecast origins will have missing forecast-targets there.
                targets_nan = torch.cat([targets_nan, torch.ones(n_forecasts, dtype=torch.bool)])
                targets_valid = ~targets_nan

    valid_origins = valid_origins & targets_valid

    # AR LAGS
    if n_lags > 0:
        # boolean vector, starting at origin_index = n_lags -1
        y_lags_nan = torch.tensor(sliding_window_view(tensor_isna["y_scaled"], window_shape=n_lags).any(axis=-1))
        # fill first n_lags -1 positions with True
        # as there are missing lags for the corresponding origin_indexes
        y_lags_nan = torch.cat([torch.ones(n_lags - 1, dtype=torch.bool), y_lags_nan])
        y_lags_valid = ~y_lags_nan
        valid_origins = valid_origins & y_lags_valid

    # LAGGED REGRESSORS
    if config_lagged_regressors is not None:  # and max_lags > 0:
        reg_lags_valid = torch.ones(tensor_length, dtype=torch.bool)
        for name, lagged_regressor in config_lagged_regressors.items():
            n_reg_lags = lagged_regressor.n_lags
            if n_reg_lags > 0:
                # boolean vector, starting at origin_index = n_lags -1
                reg_lags_nan = torch.tensor(
                    sliding_window_view(tensor_isna[name].numpy(), window_shape=n_reg_lags).any(axis=-1)
                )
                # fill first n_reg_lags -1 positions with True,
                # as there are missing lags for the corresponding origin_indexes
                reg_lags_nan = torch.cat([torch.ones(n_reg_lags - 1, dtype=torch.bool), reg_lags_nan])
                reg_lags_valid_i = ~reg_lags_nan
                reg_lags_valid = reg_lags_valid & reg_lags_valid_i
        valid_origins = valid_origins & reg_lags_valid

    # TIME: TREND & SEASONALITY: the time at each sample's lags and forecasts
    # FUTURE REGRESSORS
    # EVENTS
    names = ["t"] + future_regressor_names + event_names
    valid_columns = mask_origin_without_nan_for_columns(tensor_isna, names, max_lags, n_lags, n_forecasts)
    valid_origins = valid_origins & valid_columns

    return valid_origins


def mask_origin_without_nan_for_columns(tensor_isna, names, max_lags, n_lags, n_forecasts):
    contains_nan = torch.stack([tensor_isna[name] for name in names], dim=1).any(dim=1)
    if max_lags > 0:
        if n_lags == 0 and n_forecasts == 1:
            contains_nan = contains_nan[1:]
            contains_nan = torch.cat([contains_nan, torch.tensor([True], dtype=torch.bool)])
        else:
            contains_nan = sliding_window_view(contains_nan.numpy(), window_shape=n_lags + n_forecasts).any(axis=-1)
            # first sample is at origin_index = n_lags -1,
            if n_lags == 0:  # first sample origin index is at -1
                contains_nan = contains_nan[1:]
            else:
                contains_nan = torch.cat([torch.ones(n_lags - 1, dtype=torch.bool), torch.tensor(contains_nan)])
            # there are n_forecasts origin_indexes missing at end
            contains_nan = torch.cat([torch.tensor(contains_nan), torch.ones(n_forecasts, dtype=torch.bool)])
    valid_origins = ~contains_nan
    return valid_origins


def sort_regressor_names(config):
    additive_regressors_names = []
    multiplicative_regressors_names = []
    if config is not None and config.regressors is not None:
        # sort and divide regressors into multiplicative and additive
        for reg in sorted(list(config.regressors.keys())):
            mode = config.regressors[reg].mode
            if mode == "additive":
                additive_regressors_names.append(reg)
            else:
                multiplicative_regressors_names.append(reg)
    return additive_regressors_names, multiplicative_regressors_names

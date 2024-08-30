import logging
from collections import OrderedDict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data.dataset import Dataset

from neuralprophet import configure, utils
from neuralprophet.event_utils import get_all_holidays
from neuralprophet.utils_time_dataset import (
    pack_additive_events_component,
    pack_additive_regressors_component,
    pack_lagged_regerssors_component,
    pack_lags_component,
    pack_multiplicative_events_component,
    pack_multiplicative_regressors_component,
    pack_seasonalities_component,
    pack_targets_component,
    pack_trend_component,
)

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
        config_model,
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
        self.config_model = config_model

        if self.config_model.max_lags == 0:
            assert self.n_forecasts == 1
        self.two_level_inputs = ["seasonalities", "covariates", "events", "regressors"]

        # Preprocessing of events and holidays features (added to self.df)
        (
            self.df,
            self.additive_event_and_holiday_names,
            self.multiplicative_event_and_holiday_names,
        ) = self.add_event_features_to_df(
            self.df,
            self.config_events,
            self.config_country_holidays,
        )
        # pre-sort additive/multiplicative regressors
        self.additive_regressors_names, self.multiplicative_regressors_names = self.sort_regressor_names(
            self.config_regressors
        )

        # skipping col "ID" is string type that is interpreted as object by torch (self.df[col].dtype == "O")
        # "ID" is stored in self.meta["df_name"]
        skip_cols = ["ID", "ds"]
        for col in self.df.columns:
            if col not in skip_cols:
                self.df[col] = self.df[col].astype(float)
        # Create the tensor dictionary with the correct data types
        self.df_tensors = {
            col: torch.tensor(self.df[col].values, dtype=torch.float32) for col in self.df if col not in skip_cols
        }
        self.df["ds"] = self.df["ds"].apply(lambda x: x.timestamp())  # Convert to Unix timestamp in seconds
        self.df_tensors["ds"] = torch.tensor(self.df["ds"].values, dtype=torch.int64)

        if self.additive_event_and_holiday_names:
            self.df_tensors["additive_event_and_holiday"] = torch.stack(
                [self.df_tensors[name] for name in self.additive_event_and_holiday_names], dim=1
            )
        if self.multiplicative_event_and_holiday_names:
            self.df_tensors["multiplicative_event_and_holiday"] = torch.stack(
                [self.df_tensors[name] for name in self.multiplicative_event_and_holiday_names], dim=1
            )

        if self.additive_regressors_names:
            self.df_tensors["additive_regressors"] = torch.stack(
                [self.df_tensors[name] for name in self.additive_regressors_names], dim=1
            )
        if self.multiplicative_regressors_names:
            self.df_tensors["multiplicative_regressors"] = torch.stack(
                [self.df_tensors[name] for name in self.multiplicative_regressors_names], dim=1
            )

        # Construct index map
        self.sample2index_map, self.length = self.create_sample2index_map(self.df, self.df_tensors)

        if self.config_seasonality is not None and hasattr(self.config_seasonality, "periods"):
            self.calculate_seasonalities()

        self.stack_all_features()

    def stack_all_features(self):
        """
        Stack all features into one large tensor by calling individual stacking methods.
        """
        feature_list = []
        feature_indices = {}

        current_idx = 0

        # Call individual stacking functions
        current_idx = pack_trend_component(self.df_tensors, feature_list, feature_indices, current_idx)
        current_idx = pack_targets_component(self.df_tensors, feature_list, feature_indices, current_idx)

        current_idx = pack_lags_component(self.df_tensors, feature_list, feature_indices, current_idx, self.n_lags)
        current_idx = pack_lagged_regerssors_component(
            self.df_tensors, feature_list, feature_indices, current_idx, self.config_lagged_regressors
        )
        current_idx = pack_additive_events_component(
            self.df_tensors, feature_list, feature_indices, current_idx, self.additive_event_and_holiday_names
        )
        current_idx = pack_multiplicative_events_component(
            self.df_tensors, feature_list, feature_indices, current_idx, self.multiplicative_event_and_holiday_names
        )
        current_idx = pack_additive_regressors_component(
            self.df_tensors, feature_list, feature_indices, current_idx, self.additive_regressors_names
        )
        current_idx = pack_multiplicative_regressors_component(
            self.df_tensors, feature_list, feature_indices, current_idx, self.multiplicative_regressors_names
        )

        if self.config_seasonality is not None and hasattr(self.config_seasonality, "periods"):
            current_idx = pack_seasonalities_component(
                feature_list, feature_indices, current_idx, self.config_seasonality, self.seasonalities
            )

        # Concatenate all features into one big tensor
        self.all_features = torch.cat(feature_list, dim=1)  # Concatenating along the second dimension

        # Update the model's features map if applicable
        if self.config_model is not None:
            self.config_model.features_map = feature_indices

        return feature_indices

    def calculate_seasonalities(self):
        self.seasonalities = OrderedDict({})
        dates = self.df_tensors["ds"]
        t = (dates - torch.tensor(datetime(1900, 1, 1).timestamp())).float() / (3600 * 24.0)

        def compute_fourier_features(t, period):
            factor = 2.0 * np.pi / period.period
            sin_terms = torch.sin(factor * t[:, None] * torch.arange(1, period.resolution + 1))
            cos_terms = torch.cos(factor * t[:, None] * torch.arange(1, period.resolution + 1))
            return torch.cat((sin_terms, cos_terms), dim=1)

        for name, period in self.config_seasonality.periods.items():
            if period.resolution > 0:
                features = compute_fourier_features(t, period)

                if period.condition_name is not None:
                    condition_values = self.df_tensors[period.condition_name].unsqueeze(1)
                    features *= condition_values
                self.seasonalities[name] = features

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

        # Extract features from dataframe at given target index position
        if self.config_model.max_lags > 0:
            min_start_index = df_index - self.config_model.max_lags + 1
            max_end_index = df_index + self.n_forecasts + 1
            inputs = self.all_features[min_start_index:max_end_index, :]
        else:
            inputs = self.all_features[df_index, :]

        return inputs, self.meta

    def __len__(self):
        """Overrides Parent class method to get data length."""
        return self.length

    def sample_index_to_df_index(self, sample_index):
        """Translates a single outer sample to dataframe index"""
        return self.sample2index_map[sample_index]

    def create_sample2index_map(self, df, df_tensors):
        """creates mapping of sample index to corresponding df index at prediction origin.
        (prediction origin: last observation before forecast / future period starts).
        return created mapping to sample2index_map and number of samples.
        """

        # Limit target range due to input lags and number of forecasts
        df_length = len(df_tensors["ds"])
        origin_start_end_mask = self.create_origin_start_end_mask(
            df_length=df_length, max_lags=self.config_model.max_lags, n_forecasts=self.n_forecasts
        )

        # Prediction Frequency
        # Filter missing samples and prediction frequency (does not actually drop, but creates indexmapping)
        prediction_frequency_mask = self.create_prediction_frequency_filter_mask(
            df_tensors["ds"], self.prediction_frequency
        )

        # Combine prediction origin masks
        valid_prediction_mask = prediction_frequency_mask & origin_start_end_mask

        # Create NAN-free index mapping of sample index to df index
        nan_mask = self.create_nan_mask(
            df_tensors=df_tensors,
            predict_mode=self.predict_mode,
            max_lags=self.config_model.max_lags,
            n_lags=self.n_lags,
            n_forecasts=self.n_forecasts,
            config_lagged_regressors=self.config_lagged_regressors,
            future_regressor_names=self.additive_regressors_names + self.multiplicative_regressors_names,
            event_names=self.additive_event_and_holiday_names + self.multiplicative_event_and_holiday_names,
        )  # boolean array where NAN are False

        # Filter NAN
        valid_sample_mask = valid_prediction_mask & nan_mask
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

    def log_input_shapes(self, inputs):
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

    def get_event_offset_features(self, event, config, feature):
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
        offsets = range(config.lower_window, config.upper_window + 1)
        offset_features = pd.concat(
            {
                utils.create_event_names_for_offsets(event, offset): feature.shift(periods=offset, fill_value=0.0)
                for offset in offsets
            },
            axis=1,
        )
        return offset_features

    def add_event_features_to_df(
        self,
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
            return name.replace(" (observed)", "") if "(observed)" in name else name

        def add_offset_features(feature, event_name, config):
            additive_names = []
            multiplicative_names = []
            for offset in range(config.lower_window, config.upper_window + 1):
                event_offset_name = utils.create_event_names_for_offsets(event_name, offset)
                df[event_offset_name] = feature.shift(periods=offset, fill_value=0.0)
                if config.mode == "additive":
                    additive_names.append(event_offset_name)
                else:
                    multiplicative_names.append(event_offset_name)
            return additive_names, multiplicative_names

        # Create all additional user-specified offset events
        additive_events_names = []
        multiplicative_events_names = []

        if config_events is not None:
            for event in sorted(config_events.keys()):
                feature = df[event]
                config = config_events[event]
                additive_names, multiplicative_names = add_offset_features(feature, event, config)
                additive_events_names.extend(additive_names)
                multiplicative_events_names.extend(multiplicative_names)

        # Create all country-specific holidays and their offsets
        additive_holiday_names = []
        multiplicative_holiday_names = []

        if config_country_holidays is not None:
            year_list = df["ds"].dt.year.unique()
            country_holidays_dict = get_all_holidays(year_list, config_country_holidays.country)
            config = config_country_holidays

            for holiday in config_country_holidays.holiday_names:
                feature = pd.Series(np.zeros(len(df)), index=df.index, dtype=np.float32)
                normalized_holiday = normalize_holiday_name(holiday)

                if normalized_holiday in country_holidays_dict:
                    dates = country_holidays_dict[normalized_holiday]
                    feature.loc[df["ds"].isin(dates)] = 1.0
                else:
                    raise ValueError(f"Holiday {holiday} not found in {config_country_holidays.country} holidays")

                additive_names, multiplicative_names = add_offset_features(feature, normalized_holiday, config)
                additive_holiday_names.extend(additive_names)
                multiplicative_holiday_names.extend(multiplicative_names)

        additive_event_and_holiday_names = sorted(additive_events_names + additive_holiday_names)
        multiplicative_event_and_holiday_names = sorted(multiplicative_events_names + multiplicative_holiday_names)

        return df, additive_event_and_holiday_names, multiplicative_event_and_holiday_names

    def create_origin_start_end_mask(self, df_length, max_lags, n_forecasts):
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

    def create_prediction_frequency_filter_mask(self, timestamps, prediction_frequency=None):
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
        if prediction_frequency is None:
            return torch.ones(len(timestamps), dtype=torch.bool)

        timestamps = pd.to_datetime(timestamps.numpy(), unit="s")
        mask = torch.ones(len(timestamps), dtype=torch.bool)

        filters = {
            "hourly-minute": timestamps.minute,
            "daily-hour": timestamps.hour,
            "weekly-day": timestamps.dayofweek,
            "monthly-day": timestamps.day,
            "yearly-month": timestamps.month,
        }

        for key, value in prediction_frequency.items():
            if key not in filters:
                raise ValueError(f"Invalid prediction frequency: {key}")
            mask &= filters[key] == value

        return torch.tensor(mask, dtype=torch.bool)

    def create_nan_mask(
        self,
        df_tensors,
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
        tensor_length = len(df_tensors["ds"])
        valid_origins = torch.ones(tensor_length, dtype=torch.bool)
        tensor_isna = {k: torch.isnan(v) for k, v in df_tensors.items()}

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

        valid_origins &= targets_valid

        # AR LAGS
        if n_lags > 0:
            # boolean vector, starting at origin_index = n_lags -1
            y_lags_nan = torch.tensor(sliding_window_view(tensor_isna["y_scaled"], window_shape=n_lags).any(axis=-1))
            # fill first n_lags -1 positions with True
            # as there are missing lags for the corresponding origin_indexes
            y_lags_nan = torch.cat([torch.ones(n_lags - 1, dtype=torch.bool), y_lags_nan])
            y_lags_valid = ~y_lags_nan
            valid_origins &= y_lags_valid

        # LAGGED REGRESSORS
        if (
            config_lagged_regressors is not None and config_lagged_regressors.regressors is not None
        ):  # and max_lags > 0:
            reg_lags_valid = torch.ones(tensor_length, dtype=torch.bool)
            for name, lagged_regressor in config_lagged_regressors.regressors.items():
                n_reg_lags = lagged_regressor.n_lags
                if n_reg_lags > 0:
                    # boolean vector, starting at origin_index = n_lags -1
                    reg_lags_nan = torch.tensor(
                        sliding_window_view(tensor_isna[name].numpy(), window_shape=n_reg_lags).any(axis=-1)
                    )
                    # fill first n_reg_lags -1 positions with True,
                    # as there are missing lags for the corresponding origin_indexes
                    reg_lags_nan = torch.cat([torch.ones(n_reg_lags - 1, dtype=torch.bool), reg_lags_nan])
                    reg_lags_valid &= ~reg_lags_nan
            valid_origins &= reg_lags_valid

        # TIME: TREND & SEASONALITY: the time at each sample's lags and forecasts
        # FUTURE REGRESSORS
        # EVENTS
        names = ["t"] + future_regressor_names + event_names
        valid_columns = self.mask_origin_without_nan_for_columns(tensor_isna, names, max_lags, n_lags, n_forecasts)
        valid_origins &= valid_columns

        return valid_origins

    def mask_origin_without_nan_for_columns(self, tensor_isna, names, max_lags, n_lags, n_forecasts):
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

    def sort_regressor_names(self, config):
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
        config_model,
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
                config_model=config_model,
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

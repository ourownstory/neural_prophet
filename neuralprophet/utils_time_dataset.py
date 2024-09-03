from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch


class ComponentStacker:
    def __init__(
        self,
        n_lags,
        n_forecasts,
        max_lags,
        feature_indices={},
        config_seasonality=None,
        lagged_regressor_config=None,
    ):
        """
        Initializes the ComponentStacker with the necessary parameters.

        Args:
            n_lags (int): Number of lags used in the model.
            n_forecasts (int): Number of forecasts to be made.
            max_lags (int): Maximum number of lags used in the model.
            feature_indices (dict): A dictionary containing the start and end indices of different features in the tensor.
            config_seasonality (object, optional): Configuration object that defines the seasonality periods.
            lagged_regressor_config (dict, optional): Configuration dictionary that defines the lagged regressors and their properties.
        """
        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self.max_lags = max_lags
        self.feature_indices = feature_indices
        self.config_seasonality = config_seasonality
        self.lagged_regressor_config = lagged_regressor_config

    def unstack_component(self, component_name, batch_tensor):
        """
        Routes the unstackion process to the appropriate function based on the component name.

        Args:
            component_name (str): The name of the component to unstack.

        Returns:
            Various: The output of the specific unstackion function.
        """
        if component_name == "targets":
            return self.unstack_targets(batch_tensor)
        elif component_name == "time":
            return self.unstack_time(batch_tensor)
        elif component_name == "seasonalities":
            return self.unstack_seasonalities(batch_tensor)
        elif component_name == "lagged_regressors":
            return self.unstack_lagged_regressors(batch_tensor)
        elif component_name == "lags":
            return self.unstack_lags(batch_tensor)
        elif component_name == "additive_events":
            return self.unstack_additive_events(batch_tensor)
        elif component_name == "multiplicative_events":
            return self.unstack_multiplicative_events(batch_tensor)
        elif component_name == "additive_regressors":
            return self.unstack_additive_regressors(batch_tensor)
        elif component_name == "multiplicative_regressors":
            return self.unstack_multiplicative_regressors(batch_tensor)
        else:
            raise ValueError(f"Unknown component name: {component_name}")

    def unstack_targets(self, batch_tensor):
        targets_start_idx, targets_end_idx = self.feature_indices["targets"]
        if self.max_lags > 0:
            return batch_tensor[:, self.max_lags : self.max_lags + self.n_forecasts, targets_start_idx].unsqueeze(2)
        else:
            return batch_tensor[:, targets_start_idx : targets_end_idx + 1].unsqueeze(1)

    def unstack_time(self, batch_tensor):
        start_idx, end_idx = self.feature_indices["time"]
        if self.max_lags > 0:
            return batch_tensor[:, self.max_lags - self.n_lags : self.max_lags + self.n_forecasts, start_idx]
        else:
            return batch_tensor[:, start_idx : end_idx + 1]

    def unstack_lags(self, batch_tensor):
        lags_start_idx, _ = self.feature_indices["lags"]
        return batch_tensor[:, self.max_lags - self.n_lags : self.max_lags, lags_start_idx]

    def unstack_lagged_regressors(self, batch_tensor):
        lagged_regressors = OrderedDict()
        if self.lagged_regressor_config is not None and self.lagged_regressor_config.regressors is not None:
            for name, lagged_regressor in self.lagged_regressor_config.regressors.items():
                lagged_regressor_key = f"lagged_regressor_{name}"
                if lagged_regressor_key in self.feature_indices:
                    lagged_regressor_start_idx, _ = self.feature_indices[lagged_regressor_key]
                    covar_lags = lagged_regressor.n_lags
                    lagged_regressor_offset = self.max_lags - covar_lags
                    lagged_regressors[name] = batch_tensor[
                        :,
                        lagged_regressor_offset : lagged_regressor_offset + covar_lags,
                        lagged_regressor_start_idx,
                    ]
        return lagged_regressors

    def unstack_seasonalities(self, batch_tensor):
        seasonalities = OrderedDict()
        if self.max_lags > 0:
            for seasonality_name in self.config_seasonality.periods.keys():
                seasonality_key = f"seasonality_{seasonality_name}"
                if seasonality_key in self.feature_indices:
                    seasonality_start_idx, seasonality_end_idx = self.feature_indices[seasonality_key]
                    seasonalities[seasonality_name] = batch_tensor[
                        :,
                        self.max_lags - self.n_lags : self.max_lags + self.n_forecasts,
                        seasonality_start_idx:seasonality_end_idx,
                    ]
        else:
            for seasonality_name in self.config_seasonality.periods.keys():
                seasonality_key = f"seasonality_{seasonality_name}"
                if seasonality_key in self.feature_indices:
                    seasonality_start_idx, seasonality_end_idx = self.feature_indices[seasonality_key]
                    seasonalities[seasonality_name] = batch_tensor[
                        :, seasonality_start_idx:seasonality_end_idx
                    ].unsqueeze(1)

        return seasonalities

    def unstack_additive_events(self, batch_tensor):
        if self.max_lags > 0:
            events_start_idx, events_end_idx = self.feature_indices["additive_events"]
            future_offset = self.max_lags - self.n_lags
            return batch_tensor[
                :, future_offset : future_offset + self.n_forecasts + self.n_lags, events_start_idx : events_end_idx + 1
            ]
        else:
            events_start_idx, events_end_idx = self.feature_indices["additive_events"]

            return batch_tensor[:, events_start_idx : events_end_idx + 1].unsqueeze(1)

    def unstack_multiplicative_events(self, batch_tensor):
        if self.max_lags > 0:
            events_start_idx, events_end_idx = self.feature_indices["multiplicative_events"]
            return batch_tensor[
                :, self.max_lags - self.n_lags : self.max_lags + self.n_forecasts, events_start_idx : events_end_idx + 1
            ]
        else:
            events_start_idx, events_end_idx = self.feature_indices["multiplicative_events"]
            return batch_tensor[:, events_start_idx : events_end_idx + 1].unsqueeze(1)

    def unstack_additive_regressors(self, batch_tensor):
        if self.max_lags > 0:
            regressors_start_idx, regressors_end_idx = self.feature_indices["additive_regressors"]
            return batch_tensor[
                :,
                self.max_lags - self.n_lags : self.max_lags + self.n_forecasts,
                regressors_start_idx : regressors_end_idx + 1,
            ]
        else:
            regressors_start_idx, regressors_end_idx = self.feature_indices["additive_regressors"]
            return batch_tensor[:, regressors_start_idx : regressors_end_idx + 1].unsqueeze(1)

    def unstack_multiplicative_regressors(self, batch_tensor):
        if self.max_lags > 0:
            regressors_start_idx, regressors_end_idx = self.feature_indices["multiplicative_regressors"]
            future_offset = self.max_lags - self.n_lags
            return batch_tensor[
                :,
                future_offset : future_offset + self.n_forecasts + self.n_lags,
                regressors_start_idx : regressors_end_idx + 1,
            ]
        else:
            regressors_start_idx, regressors_end_idx = self.feature_indices["multiplicative_regressors"]
            return batch_tensor[:, regressors_start_idx : regressors_end_idx + 1].unsqueeze(1)

    def stack_trend_component(self, df_tensors, feature_list, current_idx):
        """
        Stack the trend (time) feature.
        """
        time_tensor = df_tensors["t"].unsqueeze(-1)  # Shape: [T, 1]
        feature_list.append(time_tensor)
        self.feature_indices["time"] = (current_idx, current_idx)
        return current_idx + 1

    def stack_lags_component(self, df_tensors, feature_list, current_idx, n_lags):
        """
        Stack the lags feature.
        """
        if n_lags >= 1 and "y_scaled" in df_tensors:
            lags_tensor = df_tensors["y_scaled"].unsqueeze(-1)
            feature_list.append(lags_tensor)
            self.feature_indices["lags"] = (current_idx, current_idx)
            return current_idx + 1
        return current_idx

    def stack_targets_component(self, df_tensors, feature_list, current_idx):
        """
        Stack the targets feature.
        """
        if "y_scaled" in df_tensors:
            targets_tensor = df_tensors["y_scaled"].unsqueeze(-1)
            feature_list.append(targets_tensor)
            self.feature_indices["targets"] = (current_idx, current_idx)
            return current_idx + 1
        return current_idx

    def stack_lagged_regerssors_component(self, df_tensors, feature_list, current_idx, config_lagged_regressors):
        """
        Stack the lagged regressor features.
        """
        if config_lagged_regressors is not None and config_lagged_regressors.regressors is not None:
            lagged_regressor_tensors = [
                df_tensors[name].unsqueeze(-1) for name in config_lagged_regressors.regressors.keys()
            ]
            stacked_lagged_regressor_tensor = torch.cat(lagged_regressor_tensors, dim=-1)
            feature_list.append(stacked_lagged_regressor_tensor)
            num_features = stacked_lagged_regressor_tensor.size(-1)
            for i, name in enumerate(config_lagged_regressors.regressors.keys()):
                self.feature_indices[f"lagged_regressor_{name}"] = (
                    current_idx + i,
                    current_idx + i + 1,
                )
            return current_idx + num_features
        return current_idx

    def stack_additive_events_component(
        self,
        df_tensors,
        feature_list,
        current_idx,
        additive_event_and_holiday_names,
    ):
        """
        Stack the additive event and holiday features.
        """
        if additive_event_and_holiday_names:
            additive_events_tensor = torch.cat(
                [df_tensors[name].unsqueeze(-1) for name in additive_event_and_holiday_names],
                dim=1,
            )
            feature_list.append(additive_events_tensor)
            self.feature_indices["additive_events"] = (
                current_idx,
                current_idx + additive_events_tensor.size(1) - 1,
            )
            return current_idx + additive_events_tensor.size(1)
        return current_idx

    def stack_multiplicative_events_component(
        self, df_tensors, feature_list, current_idx, multiplicative_event_and_holiday_names
    ):
        """
        Stack the multiplicative event and holiday features.
        """
        if multiplicative_event_and_holiday_names:
            multiplicative_events_tensor = torch.cat(
                [df_tensors[name].unsqueeze(-1) for name in multiplicative_event_and_holiday_names], dim=1
            )
            feature_list.append(multiplicative_events_tensor)
            self.feature_indices["multiplicative_events"] = (
                current_idx,
                current_idx + multiplicative_events_tensor.size(1) - 1,
            )
            return current_idx + multiplicative_events_tensor.size(1)
        return current_idx

    def stack_additive_regressors_component(self, df_tensors, feature_list, current_idx, additive_regressors_names):
        """
        Stack the additive regressor features.
        """
        if additive_regressors_names:
            additive_regressors_tensor = torch.cat(
                [df_tensors[name].unsqueeze(-1) for name in additive_regressors_names], dim=1
            )
            feature_list.append(additive_regressors_tensor)
            self.feature_indices["additive_regressors"] = (
                current_idx,
                current_idx + additive_regressors_tensor.size(1) - 1,
            )
            return current_idx + additive_regressors_tensor.size(1)
        return current_idx

    def stack_multiplicative_regressors_component(
        self, df_tensors, feature_list, current_idx, multiplicative_regressors_names
    ):
        """
        Stack the multiplicative regressor features.
        """
        if multiplicative_regressors_names:
            multiplicative_regressors_tensor = torch.cat(
                [df_tensors[name].unsqueeze(-1) for name in multiplicative_regressors_names], dim=1
            )  # Shape: [batch_size, num_multiplicative_regressors, 1]
            feature_list.append(multiplicative_regressors_tensor)
            self.feature_indices["multiplicative_regressors"] = (
                current_idx,
                current_idx + len(multiplicative_regressors_names) - 1,
            )
            return current_idx + len(multiplicative_regressors_names)
        return current_idx

    def stack_seasonalities_component(self, feature_list, current_idx, config_seasonality, seasonalities):
        """
        Stack the seasonality features.
        """
        if config_seasonality and config_seasonality.periods:
            for seasonality_name, features in seasonalities.items():
                seasonal_tensor = features
                feature_list.append(seasonal_tensor)
                self.feature_indices[f"seasonality_{seasonality_name}"] = (
                    current_idx,
                    current_idx + seasonal_tensor.size(1),
                )
                current_idx += seasonal_tensor.size(1)
        return current_idx

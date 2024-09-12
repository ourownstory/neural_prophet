from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import torch

from neuralprophet.configure_components import LaggedRegressors, Seasonalities


@dataclass
class ComponentStacker:
    """
    ComponentStacker is a utility class that helps in stacking and unstacking the different components of the time series data.
    Args:
        n_lags (int): Number of lags used in the model.
        n_forecasts (int): Number of forecasts to be made.
        max_lags (int): Maximum number of lags used in the model.
        feature_indices (dict): A dictionary containing the start and end indices of different features in the tensor.
        config_seasonality (object, optional): Configuration object that defines the seasonality periods.
        lagged_regressor_config (dict, optional): Configuration dictionary that defines the lagged regressors and their properties.
    """

    n_lags: int
    n_forecasts: int
    max_lags: int
    feature_indices: dict = field(default_factory=dict)
    config_seasonality: Optional[Seasonalities] = None
    lagged_regressor_config: Optional[LaggedRegressors] = None
    stack_func: dict = field(init=False)
    unstack_func: dict = field(init=False)

    def __post_init__(self):
        """
        Initializes mappings to comonent stacking and unstacking functions.
        """
        self.stack_func = {
            "targets": self.stack_targets,
            "time": self.stack_time,
            "seasonalities": self.stack_seasonalities,
            "lagged_regressors": self.stack_lagged_regressors,
            "lags": self.stack_lags,
            "additive_events": self.stack_additive_events,
            "multiplicative_events": self.stack_multiplicative_events,
            "additive_regressors": self.stack_additive_regressors,
            "multiplicative_regressors": self.stack_multiplicative_regressors,
        }
        self.unstack_func = {
            "targets": self.unstack_targets,
            "time": self.unstack_time,
            "seasonalities": self.unstack_seasonalities,
            "lagged_regressors": self.unstack_lagged_regressors,
            "lags": self.unstack_lags,
            "additive_events": self.unstack_additive_events,
            "multiplicative_events": self.unstack_multiplicative_events,
            "additive_regressors": self.unstack_additive_regressors,
            "multiplicative_regressors": self.unstack_multiplicative_regressors,
        }

    def unstack(self, component_name, batch_tensor):
        """
        Routes the unstackion process to the appropriate function based on the component name.

        Args:
            component_name (str): The name of the component to unstack.

        Returns:
            Various: The output of the specific unstacking function.
        """
        assert component_name in self.unstack_func, f"Unknown component name: {component_name}"
        return self.unstack_func[component_name](batch_tensor)

    def stack(self, component_name, df_tensors, feature_list, current_idx, **kwargs):
        """
        Routes the unstackion process to the appropriate function based on the component name.

        Args:
            component_name (str): The name of the component to stack.
            df_tensors
            feature_list
            current_idx
            kwargs for specific component, mostly component configuration

        Returns:
            current_idx: the current index in the stack of features.
        """
        assert component_name in self.stack_func, f"Unknown component name: {component_name}"
        # this is currently not working for seasonalities
        return self.stack_func[component_name](
            df_tensors=df_tensors, feature_list=feature_list, current_idx=current_idx, **kwargs
        )

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

    def stack_time(self, df_tensors, feature_list, current_idx):
        """
        Stack the trend (time) feature.
        """
        time_tensor = df_tensors["t"].unsqueeze(-1)  # Shape: [T, 1]
        feature_list.append(time_tensor)
        self.feature_indices["time"] = (current_idx, current_idx)
        return current_idx + 1

    def stack_lags(self, df_tensors, feature_list, current_idx, n_lags):
        """
        Stack the lags feature.
        """
        if n_lags >= 1 and "y_scaled" in df_tensors:
            lags_tensor = df_tensors["y_scaled"].unsqueeze(-1)
            feature_list.append(lags_tensor)
            self.feature_indices["lags"] = (current_idx, current_idx)
            return current_idx + 1
        return current_idx

    def stack_targets(self, df_tensors, feature_list, current_idx):
        """
        Stack the targets feature.
        """
        if "y_scaled" in df_tensors:
            targets_tensor = df_tensors["y_scaled"].unsqueeze(-1)
            feature_list.append(targets_tensor)
            self.feature_indices["targets"] = (current_idx, current_idx)
            return current_idx + 1
        return current_idx

    def stack_lagged_regressors(self, df_tensors, feature_list, current_idx, config):
        """
        Stack the lagged regressor features.
        """
        if config is not None and config.regressors is not None:
            lagged_regressor_tensors = [df_tensors[name].unsqueeze(-1) for name in config.regressors.keys()]
            stacked_lagged_regressor_tensor = torch.cat(lagged_regressor_tensors, dim=-1)
            feature_list.append(stacked_lagged_regressor_tensor)
            num_features = stacked_lagged_regressor_tensor.size(-1)
            for i, name in enumerate(config.regressors.keys()):
                self.feature_indices[f"lagged_regressor_{name}"] = (
                    current_idx + i,
                    current_idx + i + 1,
                )
            return current_idx + num_features
        return current_idx

    def stack_additive_events(self, df_tensors, feature_list, current_idx, names):
        """
        Stack the additive event and holiday features.
        """
        if names:
            additive_events_tensor = torch.cat(
                [df_tensors[name].unsqueeze(-1) for name in names],
                dim=1,
            )
            feature_list.append(additive_events_tensor)
            self.feature_indices["additive_events"] = (
                current_idx,
                current_idx + additive_events_tensor.size(1) - 1,
            )
            return current_idx + additive_events_tensor.size(1)
        return current_idx

    def stack_multiplicative_events(self, df_tensors, feature_list, current_idx, names):
        """
        Stack the multiplicative event and holiday features.
        """
        if names:
            multiplicative_events_tensor = torch.cat([df_tensors[name].unsqueeze(-1) for name in names], dim=1)
            feature_list.append(multiplicative_events_tensor)
            self.feature_indices["multiplicative_events"] = (
                current_idx,
                current_idx + multiplicative_events_tensor.size(1) - 1,
            )
            return current_idx + multiplicative_events_tensor.size(1)
        return current_idx

    def stack_additive_regressors(self, df_tensors, feature_list, current_idx, names):
        """
        Stack the additive regressor features.
        """
        if names:
            additive_regressors_tensor = torch.cat([df_tensors[name].unsqueeze(-1) for name in names], dim=1)
            feature_list.append(additive_regressors_tensor)
            self.feature_indices["additive_regressors"] = (
                current_idx,
                current_idx + additive_regressors_tensor.size(1) - 1,
            )
            return current_idx + additive_regressors_tensor.size(1)
        return current_idx

    def stack_multiplicative_regressors(self, df_tensors, feature_list, current_idx, names):
        """
        Stack the multiplicative regressor features.
        """
        if names:
            multiplicative_regressors_tensor = torch.cat(
                [df_tensors[name].unsqueeze(-1) for name in names], dim=1
            )  # Shape: [batch_size, num_multiplicative_regressors, 1]
            feature_list.append(multiplicative_regressors_tensor)
            self.feature_indices["multiplicative_regressors"] = (
                current_idx,
                current_idx + len(names) - 1,
            )
            return current_idx + len(names)
        return current_idx

    def stack_seasonalities(self, df_tensors, feature_list, current_idx, config, seasonalities):
        """
        Stack the seasonality features.
        """
        # TODO conform to other stack functions, using df_tensors
        if config and config.periods:
            for seasonality_name, features in seasonalities.items():
                seasonal_tensor = features
                feature_list.append(seasonal_tensor)
                self.feature_indices[f"seasonality_{seasonality_name}"] = (
                    current_idx,
                    current_idx + seasonal_tensor.size(1),
                )
                current_idx += seasonal_tensor.size(1)
        return current_idx

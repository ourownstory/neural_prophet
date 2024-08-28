from collections import OrderedDict


class FeatureExtractor:
    def __init__(
        self,
        n_lags,
        n_forecasts,
        max_lags,
        data_tensor=None,
        feature_indices=None,
        config_seasonality=None,
        lagged_regressor_config=None,
    ):
        """
        Initializes the FeatureExtractor with the necessary parameters.

        Args:
            data_tensor (torch.Tensor): The tensor containing all features, sliced according to indices.
            n_lags (int): Number of lags used in the model.
            n_forecasts (int): Number of forecasts to be made.
            max_lags (int): Maximum number of lags used in the model.
            feature_indices (dict): A dictionary containing the start and end indices of different features in the tensor.
            config_seasonality (object, optional): Configuration object that defines the seasonality periods.
            lagged_regressor_config (dict, optional): Configuration dictionary that defines the lagged regressors and their properties.
        """
        self.data_tensor = data_tensor
        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self.max_lags = max_lags
        self.feature_indices = feature_indices
        self.config_seasonality = config_seasonality
        self.lagged_regressor_config = lagged_regressor_config

    def update_data_inputs(self, data_tensor, feature_indices):
        """
        Updates the data tensor with a new tensor.

        Args:
            data_tensor (torch.Tensor): The new tensor containing all features.
        """
        self.data_tensor = data_tensor
        self.feature_indices = feature_indices

    def extract(self, component_name):
        """
        Routes the extraction process to the appropriate function based on the component name.

        Args:
            component_name (str): The name of the component to extract.

        Returns:
            Various: The output of the specific extraction function.
        """
        if component_name == "targets":
            return self.extract_targets()
        elif component_name == "time":
            return self.extract_time()
        elif component_name == "seasonalities":
            return self.extract_seasonalities()
        elif component_name == "lagged_regressors":
            return self.extract_lagged_regressors()
        elif component_name == "lags":
            return self.extract_lags()
        elif component_name == "additive_events":
            return self.extract_additive_events()
        elif component_name == "multiplicative_events":
            return self.extract_multiplicative_events()
        elif component_name == "additive_regressors":
            return self.extract_additive_regressors()
        elif component_name == "multiplicative_regressors":
            return self.extract_multiplicative_regressors()
        else:
            raise ValueError(f"Unknown component name: {component_name}")

    def extract_targets(self):
        targets_start_idx, targets_end_idx = self.feature_indices["targets"]
        if self.max_lags > 0:
            return self.data_tensor[:, self.max_lags : self.max_lags + self.n_forecasts, targets_start_idx].unsqueeze(2)
        else:
            return self.data_tensor[:, targets_start_idx : targets_end_idx + 1].unsqueeze(1)

    def extract_time(self):
        start_idx, end_idx = self.feature_indices["time"]
        if self.max_lags > 0:
            return self.data_tensor[:, self.max_lags - self.n_lags : self.max_lags + self.n_forecasts, start_idx]
        else:
            return self.data_tensor[:, start_idx : end_idx + 1]

    def extract_lags(self):
        lags_start_idx, _ = self.feature_indices["lags"]
        return self.data_tensor[:, self.max_lags - self.n_lags : self.max_lags, lags_start_idx]

    def extract_lagged_regressors(self):
        lagged_regressors = OrderedDict()
        if self.lagged_regressor_config:
            for name, lagged_regressor in self.lagged_regressor_config.items():
                lagged_regressor_key = f"lagged_regressor_{name}"
                if lagged_regressor_key in self.feature_indices:
                    lagged_regressor_start_idx, _ = self.feature_indices[lagged_regressor_key]
                    covar_lags = lagged_regressor.n_lags
                    lagged_regressor_offset = self.max_lags - covar_lags
                    lagged_regressors[name] = self.data_tensor[
                        :,
                        lagged_regressor_offset : lagged_regressor_offset + covar_lags,
                        lagged_regressor_start_idx,
                    ]
        return lagged_regressors

    def extract_seasonalities(self):
        seasonalities = OrderedDict()
        if self.max_lags > 0:
            for seasonality_name in self.config_seasonality.periods.keys():
                seasonality_key = f"seasonality_{seasonality_name}"
                if seasonality_key in self.feature_indices:
                    seasonality_start_idx, seasonality_end_idx = self.feature_indices[seasonality_key]
                    seasonalities[seasonality_name] = self.data_tensor[
                        :,
                        self.max_lags - self.n_lags : self.max_lags + self.n_forecasts,
                        seasonality_start_idx:seasonality_end_idx,
                    ]
        else:
            for seasonality_name in self.config_seasonality.periods.keys():
                seasonality_key = f"seasonality_{seasonality_name}"
                if seasonality_key in self.feature_indices:
                    seasonality_start_idx, seasonality_end_idx = self.feature_indices[seasonality_key]
                    seasonalities[seasonality_name] = self.data_tensor[
                        :, seasonality_start_idx:seasonality_end_idx
                    ].unsqueeze(1)

        return seasonalities

    def extract_additive_events(self):
        if self.max_lags > 0:
            events_start_idx, events_end_idx = self.feature_indices["additive_events"]
            future_offset = self.max_lags - self.n_lags
            return self.data_tensor[
                :, future_offset : future_offset + self.n_forecasts + self.n_lags, events_start_idx : events_end_idx + 1
            ]
        else:
            events_start_idx, events_end_idx = self.feature_indices["additive_events"]
            return self.data_tensor[:, events_start_idx : events_end_idx + 1].unsqueeze(1)

    def extract_multiplicative_events(self):
        if self.max_lags > 0:
            events_start_idx, events_end_idx = self.feature_indices["multiplicative_events"]
            return self.data_tensor[
                :, self.max_lags - self.n_lags : self.max_lags + self.n_forecasts, events_start_idx : events_end_idx + 1
            ]
        else:
            events_start_idx, events_end_idx = self.feature_indices["multiplicative_events"]
            return self.data_tensor[:, events_start_idx : events_end_idx + 1].unsqueeze(1)

    def extract_additive_regressors(self):
        if self.max_lags > 0:
            regressors_start_idx, regressors_end_idx = self.feature_indices["additive_regressors"]
            return self.data_tensor[
                :,
                self.max_lags - self.n_lags : self.max_lags + self.n_forecasts,
                regressors_start_idx : regressors_end_idx + 1,
            ]
        else:
            regressors_start_idx, regressors_end_idx = self.feature_indices["additive_regressors"]
            return self.data_tensor[:, regressors_start_idx : regressors_end_idx + 1].unsqueeze(1)

    def extract_multiplicative_regressors(self):
        if self.max_lags > 0:
            regressors_start_idx, regressors_end_idx = self.feature_indices["multiplicative_regressors"]
            future_offset = self.max_lags - self.n_lags
            return self.data_tensor[
                :,
                future_offset : future_offset + self.n_forecasts + self.n_lags,
                regressors_start_idx : regressors_end_idx + 1,
            ]
        else:
            regressors_start_idx, regressors_end_idx = self.feature_indices["multiplicative_regressors"]
            return self.data_tensor[:, regressors_start_idx : regressors_end_idx + 1].unsqueeze(1)

from collections import OrderedDict


def unpack_targets(
    sliced_tensor,
    n_forecasts,
    max_lags,
    feature_indices,
):
    """
    Unpacks the target values from the sliced tensor based on the given feature indices.

    Args:
        sliced_tensor (torch.Tensor): The tensor containing all features, sliced according to indices.
        n_forecasts (int): Number of forecasts to be made.
        max_lags (int): Maximum number of lags used in the model.
        feature_indices (dict): A dictionary containing the start and end indices of different features in the tensor.

    Returns:
        torch.Tensor: A tensor containing the target values, with an extra dimension added.
    """
    targets_start_idx, targets_end_idx = feature_indices["targets"]
    if max_lags > 0:
        return sliced_tensor[:, max_lags : max_lags + n_forecasts, targets_start_idx].unsqueeze(2)
    else:
        return sliced_tensor[:, targets_start_idx : targets_end_idx + 1].unsqueeze(1)


def unpack_time(sliced_tensor, n_lags, n_forecasts, max_lags, feature_indices):
    """
    Unpacks the time features from the sliced tensor.

    Args:
        sliced_tensor (torch.Tensor): The tensor containing all features, sliced according to indices.
        n_lags (int): Number of lags used in the model.
        n_forecasts (int): Number of forecasts to be made.
        max_lags (int): Maximum number of lags used in the model.
        feature_indices (dict): A dictionary containing the start and end indices of different features in the tensor.

    Returns:
        torch.Tensor: A tensor containing the time features.
    """
    start_idx, end_idx = feature_indices["time"]
    if max_lags > 0:
        return sliced_tensor[:, max_lags - n_lags : max_lags + n_forecasts, start_idx]
    else:
        return sliced_tensor[:, start_idx : end_idx + 1]


def unpack_seasonalities(sliced_tensor, n_lags, n_forecasts, max_lags, feature_indices, config_seasonality):
    """
    Unpacks the seasonality features from the sliced tensor.

    Args:
        sliced_tensor (torch.Tensor): The tensor containing all features, sliced according to indices.
        n_lags (int): Number of lags used in the model.
        n_forecasts (int): Number of forecasts to be made.
        max_lags (int): Maximum number of lags used in the model.
        feature_indices (dict): A dictionary containing the start and end indices of different features in the tensor.
        config_seasonality (object): Configuration object that defines the seasonality periods.

    Returns:
        OrderedDict: A dictionary containing the seasonality features for each period.
    """
    seasonalities = OrderedDict()
    if max_lags > 0:
        for seasonality_name in config_seasonality.periods.keys():
            seasonality_key = f"seasonality_{seasonality_name}"
            if seasonality_key in feature_indices:
                seasonality_start_idx, seasonality_end_idx = feature_indices[seasonality_key]
                seasonalities[seasonality_name] = sliced_tensor[
                    :,
                    max_lags - n_lags : max_lags + n_forecasts,
                    seasonality_start_idx:seasonality_end_idx,
                ]
    else:
        for seasonality_name in config_seasonality.periods.keys():
            seasonality_key = f"seasonality_{seasonality_name}"
            if seasonality_key in feature_indices:
                seasonality_start_idx, seasonality_end_idx = feature_indices[seasonality_key]
                seasonalities[seasonality_name] = sliced_tensor[:, seasonality_start_idx:seasonality_end_idx].unsqueeze(
                    1
                )
    return seasonalities


def unpack_lagged_regressors(sliced_tensor, max_lags, feature_indices, config_lagged_regressors):
    """
    Unpacks the lagged regressors from the sliced tensor.

    Args:
        sliced_tensor (torch.Tensor): The tensor containing all features, sliced according to indices.
        max_lags (int): Maximum number of lags used in the model.
        feature_indices (dict): A dictionary containing the start and end indices of different features in the tensor.
        config_lagged_regressors (dict): Configuration dictionary that defines the lagged regressors and their properties.

    Returns:
        OrderedDict: A dictionary containing the lagged regressor features.
    """
    lagged_regressors = OrderedDict()
    if config_lagged_regressors:
        for name, lagged_regressor in config_lagged_regressors.items():
            lagged_regressor_key = f"lagged_regressor_{name}"
            if lagged_regressor_key in feature_indices:
                lagged_regressor_start_idx, _ = feature_indices[lagged_regressor_key]
                covar_lags = lagged_regressor.n_lags
                lagged_regressor_offset = max_lags - covar_lags
                lagged_regressors[name] = sliced_tensor[
                    :,
                    lagged_regressor_offset : lagged_regressor_offset + covar_lags,
                    lagged_regressor_start_idx,
                ]
    return lagged_regressors


def unpack_lags(sliced_tensor, n_lags, max_lags, feature_indices):
    """
    Unpacks the lagged features from the sliced tensor.

    Args:
        sliced_tensor (torch.Tensor): The tensor containing all features, sliced according to indices.
        n_lags (int): Number of lags used in the model.
        max_lags (int): Maximum number of lags used in the model.
        feature_indices (dict): A dictionary containing the start and end indices of different features in the tensor.

    Returns:
        torch.Tensor: A tensor containing the lagged features.
    """
    lags_start_idx, _ = feature_indices["lags"]
    return sliced_tensor[:, max_lags - n_lags : max_lags, lags_start_idx]


def unpack_additive_events(sliced_tensor, n_lags, n_forecasts, max_lags, feature_indices):
    """
    Unpacks the additive events features from the sliced tensor.

    Args:
        sliced_tensor (torch.Tensor): The tensor containing all features, sliced according to indices.
        n_lags (int): Number of lags used in the model.
        n_forecasts (int): Number of forecasts to be made.
        max_lags (int): Maximum number of lags used in the model.
        feature_indices (dict): A dictionary containing the start and end indices of different features in the tensor.

    Returns:
        torch.Tensor: A tensor containing the additive events features.
    """
    if max_lags > 0:
        events_start_idx, events_end_idx = feature_indices["additive_events"]
        future_offset = max_lags - n_lags
        return sliced_tensor[
            :, future_offset : future_offset + n_forecasts + n_lags, events_start_idx : events_end_idx + 1
        ]
    else:
        events_start_idx, events_end_idx = feature_indices["additive_events"]
        return sliced_tensor[:, events_start_idx : events_end_idx + 1].unsqueeze(1)


def unpack_multiplicative_events(sliced_tensor, n_lags, n_forecasts, max_lags, feature_indices):
    """
    Unpacks the multiplicative events features from the sliced tensor.

    Args:
        sliced_tensor (torch.Tensor): The tensor containing all features, sliced according to indices.
        n_lags (int): Number of lags used in the model.
        n_forecasts (int): Number of forecasts to be made.
        max_lags (int): Maximum number of lags used in the model.
        feature_indices (dict): A dictionary containing the start and end indices of different features in the tensor.

    Returns:
        torch.Tensor: A tensor containing the multiplicative events features.
    """
    if max_lags > 0:
        events_start_idx, events_end_idx = feature_indices["multiplicative_events"]
        return sliced_tensor[:, max_lags - n_lags : max_lags + n_forecasts, events_start_idx : events_end_idx + 1]
    else:
        events_start_idx, events_end_idx = feature_indices["multiplicative_events"]
        return sliced_tensor[:, events_start_idx : events_end_idx + 1].unsqueeze(1)


def unpack_additive_regressor(sliced_tensor, n_lags, n_forecasts, max_lags, feature_indices):
    """
    Unpacks the additive regressor features from the sliced tensor.

    Args:
        sliced_tensor (torch.Tensor): The tensor containing all features, sliced according to indices.
        n_lags (int): Number of lags used in the model.
        n_forecasts (int): Number of forecasts to be made.
        max_lags (int): Maximum number of lags used in the model.
        feature_indices (dict): A dictionary containing the start and end indices of different features in the tensor.

    Returns:
        torch.Tensor: A tensor containing the additive regressor features.
    """
    if max_lags > 0:
        regressors_start_idx, regressors_end_idx = feature_indices["additive_regressors"]
        return sliced_tensor[
            :,
            max_lags - n_lags : max_lags + n_forecasts,
            regressors_start_idx : regressors_end_idx + 1,
        ]
    else:
        regressors_start_idx, regressors_end_idx = feature_indices["additive_regressors"]
        return sliced_tensor[:, regressors_start_idx : regressors_end_idx + 1].unsqueeze(1)


def unpack_multiplicative_regressor(sliced_tensor, n_lags, n_forecasts, max_lags, feature_indices):
    """
    Unpacks the multiplicative regressor features from the sliced tensor.

    Args:
        sliced_tensor (torch.Tensor): The tensor containing all features, sliced according to indices.
        n_lags (int): Number of lags used in the model.
        n_forecasts (int): Number of forecasts to be made.
        max_lags (int): Maximum number of lags used in the model.
        feature_indices (dict): A dictionary containing the start and end indices of different features in the tensor.

    Returns:
        torch.Tensor: A tensor containing the multiplicative regressor features.
    """
    if max_lags > 0:
        regressors_start_idx, regressors_end_idx = feature_indices["multiplicative_regressors"]
        future_offset = max_lags - n_lags
        return sliced_tensor[
            :,
            future_offset : future_offset + n_forecasts + n_lags,
            regressors_start_idx : regressors_end_idx + 1,
        ]
    else:
        regressors_start_idx, regressors_end_idx = feature_indices["multiplicative_regressors"]
        return sliced_tensor[:, regressors_start_idx : regressors_end_idx + 1].unsqueeze(1)

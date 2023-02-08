from neuralprophet.components.future_regressors.linear import LinearFutureRegressors
from neuralprophet.components.seasonality.fourier import GlobalFourierSeasonality, LocalFourierSeasonality
from neuralprophet.components.trend.linear import GlobalLinearTrend, LocalLinearTrend
from neuralprophet.components.trend.piecewise_linear import GlobalPiecewiseLinearTrend, LocalPiecewiseLinearTrend
from neuralprophet.components.trend.static import StaticTrend


def get_trend(config, n_forecasts, quantiles, id_list, num_trends_modelled, device):
    """
    Router for all trend classes.

    Based on the conditions provided, the correct trend class is returned and initialized using the provided args.

    Parameters
        ----------
            config : configure.Trend
            n_forecasts : int
                number of steps to forecast. Aka number of model outputs
            quantiles : list
                the set of quantiles estimated
            id_list : list
                List of different time series IDs, used for global-local modelling (if enabled)

                Note
                ----
                This parameter is set to  ``['__df__']`` if only one time series is input.
            num_trends_modelled : int
                Number of different trends modelled.

                Note
                ----
                If only 1 time series is modelled, it will be always 1.

                Note
                ----
                For multiple time series. If trend is modelled globally the value is set
                to 1, otherwise it is set to the number of time series modelled.
            device : torch.device
                Device that tensors are stored on.

                Note
                ----
                This is set to ``torch.device("cpu")`` if no GPU is available.
    """
    args = {
        "config": config,
        "id_list": id_list,
        "quantiles": quantiles,
        "num_trends_modelled": num_trends_modelled,
        "n_forecasts": n_forecasts,
        "device": device,
    }

    if config.growth == "off":
        # No trend
        return StaticTrend(**args)
    elif config.growth in ["linear", "discontinuous"]:
        # Linear trend
        if num_trends_modelled == 1:
            # Global trend
            if int(config.n_changepoints) == 0:
                # Linear trend
                return GlobalLinearTrend(**args)
            else:
                # Piecewise trend
                return GlobalPiecewiseLinearTrend(**args)
        else:
            # Local trend
            if int(config.n_changepoints) == 0:
                # Linear trend
                return LocalLinearTrend(**args)
            else:
                # Piecewise trend
                return LocalPiecewiseLinearTrend(**args)
    else:
        raise ValueError(f"Growth type {config.growth} is not supported.")


def get_future_regressors(config, id_list, quantiles, n_forecasts, device, config_trend_none_bool):
    """
    Router for all future regressor classes.
    """
    args = {
        "config": config,
        "id_list": id_list,
        "quantiles": quantiles,
        "n_forecasts": n_forecasts,
        "device": device,
        "config_trend_none_bool": config_trend_none_bool,
    }

    return LinearFutureRegressors(**args)


def get_seasonality(config, id_list, quantiles, num_seasonalities_modelled, n_forecasts, device):
    """
    Router for all seasonality classes.
    """
    args = {
        "config": config,
        "id_list": id_list,
        "quantiles": quantiles,
        "n_forecasts": n_forecasts,
        "device": device,
        "num_seasonalities_modelled": num_seasonalities_modelled,
    }

    if config.global_local == "global":
        # Global seasonality
        return GlobalFourierSeasonality(**args)
    elif config.global_local == "local":
        # Local seasonality
        return LocalFourierSeasonality(**args)
    else:
        raise ValueError(f"Seasonality mode {config.global_local} is not supported.")

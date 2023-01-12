from neuralprophet.components.trend.linear import GlobalLinearTrend, LocalLinearTrend
from neuralprophet.components.trend.piecewise_linear import GlobalPiecewiseLinearTrend, LocalPiecewiseLinearTrend
from neuralprophet.components.trend.static import StaticTrend


def get_trend(config, id_list, quantiles, num_trends_modelled, n_forecasts, device):
    """
    Router for all trend classes.

    Based on the conditions provided, the correct trend class is returned and initialized using the provided args.
    """
    if config.growth == "off":
        # No trend
        return StaticTrend(
            config=config,
            id_list=id_list,
            quantiles=quantiles,
            num_trends_modelled=num_trends_modelled,
            n_forecasts=n_forecasts,
            device=device,
        )
    elif config.growth in ["linear", "discontinuous"]:
        # Linear trend
        if num_trends_modelled == 1:
            # Global trend
            if int(config.n_changepoints) == 0:
                # Linear trend
                return GlobalLinearTrend(
                    config=config,
                    id_list=id_list,
                    quantiles=quantiles,
                    num_trends_modelled=num_trends_modelled,
                    n_forecasts=n_forecasts,
                    device=device,
                )
            else:
                # Piecewise trend
                return GlobalPiecewiseLinearTrend(
                    config=config,
                    id_list=id_list,
                    quantiles=quantiles,
                    num_trends_modelled=num_trends_modelled,
                    n_forecasts=n_forecasts,
                    device=device,
                )
        else:
            # Local trend
            if int(config.n_changepoints) == 0:
                # Linear trend
                return LocalLinearTrend(
                    config=config,
                    id_list=id_list,
                    quantiles=quantiles,
                    num_trends_modelled=num_trends_modelled,
                    n_forecasts=n_forecasts,
                    device=device,
                )
            else:
                # Piecewise trend
                return LocalPiecewiseLinearTrend(
                    config=config,
                    id_list=id_list,
                    quantiles=quantiles,
                    num_trends_modelled=num_trends_modelled,
                    n_forecasts=n_forecasts,
                    device=device,
                )
    else:
        raise ValueError(f"Growth type {config.growth} is not supported.")

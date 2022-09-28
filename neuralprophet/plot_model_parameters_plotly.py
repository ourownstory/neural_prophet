import numpy as np
import pandas as pd
import logging
from neuralprophet.utils import set_y_as_percent
from neuralprophet.plot_model_parameters import predict_season_from_dates, predict_one_season
import datetime

log = logging.getLogger("NP.plotly")

try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
except ImportError:
    log.error("Importing plotly failed. Interactive plots will not work.")

# UI Configuration
color = "#2d92ff"
xaxis_args = {
    "showline": True,
    "mirror": True,
    "linewidth": 1.5,
}
yaxis_args = {
    "showline": True,
    "mirror": True,
    "linewidth": 1.5,
}
layout_args = {
    "autosize": True,
    "template": "plotly_white",
    "margin": go.layout.Margin(l=0, r=10, b=0, t=10, pad=0),
    "font": dict(size=10),
    "title": dict(font=dict(size=12)),
    "hovermode": "x unified",
}


def get_dynamic_axis_range(df_range, type, pad=0.05, inverse=False):
    """Adds a percentage of values at both ends of a list for plotting.

    Parameters
    ----------
        df_range: list
            List of axis values to pad
        type : str
            Type of values in the list to pad
        pad : float
            Percentage of padding to add to each end of the range
        inverse : bool
            Flag for list sorted in an inverted order

    Returns
    -------
        Padded range of values
    """
    if inverse:
        df_range = df_range[::-1]
    delta = df_range[round(len(df_range) * pad)]
    if type == "dt":
        range_min = min(df_range) + (min(df_range) - delta)
        range_max = max(df_range) + (delta - min(df_range))
    elif type == "numeric":

        range_min = min(df_range) - delta
        range_max = max(df_range) + delta
    else:
        raise NotImplementedError(f"The type {type} is not implemented.")
    if inverse:
        df_range = df_range[::-1]
    return [range_min, range_max]


def get_parameter_components(m, forecast_in_focus, df_name="__df__"):
    """Provides the components for plotting parameters.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        forecast_in_focus : int
            n-th step ahead forecast AR-coefficients to plot
        df_name : str
            Name of dataframe to refer to data params from original keys of train dataframes

            Note
            ----
            Only used for local normalization in global modeling

    Returns
    -------
        List of dicts consisting the parameter plot components.
    """
    # Identify components to be plotted
    components = [{"plot_name": "Trend"}]
    if m.config_trend.n_changepoints > 0:
        components.append({"plot_name": "Trend Rate Change"})

    # Plot  seasonalities, if present
    if m.config_season is not None:
        for name in m.config_season.periods:
            components.append({"plot_name": "seasonality", "comp_name": name})

    if m.n_lags > 0:
        components.append(
            {
                "plot_name": "lagged weights",
                "comp_name": "AR",
                "weights": m.model.ar_weights.detach().numpy(),
                "focus": forecast_in_focus,
            }
        )

    # all scalar regressors will be plotted together
    # collected as tuples (name, weights)

    # Add Regressors
    additive_future_regressors = []
    multiplicative_future_regressors = []
    if m.config_regressors is not None:
        for regressor, configs in m.config_regressors.items():
            mode = configs.mode
            regressor_param = m.model.get_reg_weights(regressor)
            if mode == "additive":
                additive_future_regressors.append((regressor, regressor_param.detach().numpy()))
            else:
                multiplicative_future_regressors.append((regressor, regressor_param.detach().numpy()))

    # Add Events
    additive_events = []
    multiplicative_events = []

    # add the country holidays
    if m.config_country_holidays is not None:
        for country_holiday in m.config_country_holidays.holiday_names:
            event_params = m.model.get_event_weights(country_holiday)
            weight_list = [(key, param.detach().numpy()) for key, param in event_params.items()]
            mode = m.config_country_holidays.mode
            if mode == "additive":
                additive_events = additive_events + weight_list
            else:
                multiplicative_events = multiplicative_events + weight_list

    # add the user specified events
    if m.config_events is not None:
        for event, configs in m.config_events.items():
            event_params = m.model.get_event_weights(event)
            weight_list = [(key, param.detach().numpy()) for key, param in event_params.items()]
            mode = configs.mode
            if mode == "additive":
                additive_events = additive_events + weight_list
            else:
                multiplicative_events = multiplicative_events + weight_list

    # Add Covariates
    lagged_scalar_regressors = []
    if m.config_covar is not None:
        for name in m.config_covar.keys():
            if m.config_covar[name].as_scalar:
                lagged_scalar_regressors.append((name, m.model.get_covar_weights(name).detach().numpy()))
            else:
                components.append(
                    {
                        "plot_name": "lagged weights",
                        "comp_name": f'Lagged Regressor "{name}"',
                        "weights": m.model.get_covar_weights(name).detach().numpy(),
                        "focus": forecast_in_focus,
                    }
                )

    if len(additive_future_regressors) > 0:
        components.append({"plot_name": "Additive future regressor"})
    if len(multiplicative_future_regressors) > 0:
        components.append({"plot_name": "Multiplicative future regressor"})
    if len(lagged_scalar_regressors) > 0:
        components.append({"plot_name": "Lagged scalar regressor"})
    if len(additive_events) > 0:
        data_params = m.config_normalization.get_data_params(df_name)
        scale = data_params["y"].scale
        additive_events = [(key, weight * scale) for (key, weight) in additive_events]

        components.append({"plot_name": "Additive event"})
    if len(multiplicative_events) > 0:
        components.append({"plot_name": "Multiplicative event"})

    output_dict = {
        "components": components,
        "additive_future_regressors": additive_future_regressors,
        "additive_events": additive_events,
        "multiplicative_future_regressors": multiplicative_future_regressors,
        "multiplicative_events": multiplicative_events,
        "lagged_scalar_regressors": lagged_scalar_regressors,
    }

    return output_dict


def plot_trend_change(m, plot_name="Trend Change", df_name="__df__"):
    """Make a barplot of the magnitudes of trend-changes.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        plot_name : str
            Name of the plot Title
        df_name : str
            Name of dataframe to refer to data params from original keys of train dataframes

            Note
            ----
            Only used for local normalization in global modeling

    Returns
    -------
        Dictionary with plotly traces, xaxis and yaxis
    """
    data_params = m.config_normalization.get_data_params(df_name)
    start = data_params["ds"].shift
    scale = data_params["ds"].scale
    time_span_seconds = scale.total_seconds()
    cp_t = []
    for cp in m.model.config_trend.changepoints:
        cp_t.append(start + datetime.timedelta(seconds=cp * time_span_seconds))
    weights = m.model.get_trend_deltas.detach().numpy()
    # add end-point to force scale to match trend plot
    cp_t.append(start + scale)
    weights = np.append(weights, [0.0])

    traces = []
    traces.append(
        go.Bar(
            name=plot_name,
            x=cp_t,
            y=weights,
            marker_color=color,
        )
    )

    padded_range = get_dynamic_axis_range(cp_t, type="dt")
    xaxis = go.layout.XAxis(
        title="Trend segment",
        type="date",
        range=padded_range,
    )
    yaxis = go.layout.YAxis(
        rangemode="normal",
        title=go.layout.yaxis.Title(text=plot_name),
    )

    return {"traces": traces, "xaxis": xaxis, "yaxis": yaxis}


def plot_trend(m, plot_name="Trend Change", df_name="__df__"):
    """Make a barplot of the magnitudes of trend-changes.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        plot_name : str
            Name of the plot Title
        df_name : str
            Name of dataframe to refer to data params from original keys of train dataframes

            Note
            ----
            Only used for local normalization in global modeling

    Returns
    -------
        Dictionary with plotly traces, xaxis and yaxis
    """
    traces = []
    line_width = 2

    data_params = m.config_normalization.get_data_params(df_name)
    t_start = data_params["ds"].shift
    t_end = t_start + data_params["ds"].scale

    if m.config_trend.n_changepoints == 0:
        fcst_t = pd.Series([t_start, t_end]).dt.to_pydatetime()
        trend_0 = m.model.bias.detach().numpy()
        if m.config_trend.growth == "off":
            trend_1 = trend_0
        else:
            trend_1 = trend_0 + m.model.trend_k0.detach().numpy()

        data_params = m.config_normalization.get_data_params(df_name)
        shift = data_params["y"].shift
        scale = data_params["y"].scale
        trend_0 = trend_0 * scale + shift
        trend_1 = trend_1 * scale + shift

        traces.append(
            go.Scatter(
                name=plot_name,
                x=fcst_t,
                y=np.concatenate([trend_0, trend_1]),
                mode="lines",
                line=dict(color=color, width=line_width),
                fill="none",
            )
        )
        extended_daterange = pd.date_range(
            start=fcst_t[0].strftime("%Y-%m-%d"), end=fcst_t[1].strftime("%Y-%m-%d")
        ).to_pydatetime()
        padded_range = get_dynamic_axis_range(extended_daterange, type="dt")
    else:
        days = pd.date_range(start=t_start, end=t_end, freq=m.data_freq)
        df_y = pd.DataFrame({"ds": days})
        df_trend = m.predict_trend(df_y)
        traces.append(
            go.Scatter(
                name=plot_name,
                x=df_y["ds"].dt.to_pydatetime(),
                y=df_trend["trend"],
                mode="lines",
                line=dict(color=color, width=line_width),
                fill="none",
            )
        )
        padded_range = get_dynamic_axis_range(df_y["ds"].dt.to_pydatetime(), type="dt")

    xaxis = go.layout.XAxis(
        title="ds",
        type="date",
        range=padded_range,
    )
    yaxis = go.layout.YAxis(
        rangemode="normal",
        title=go.layout.yaxis.Title(text=plot_name),
    )

    return {"traces": traces, "xaxis": xaxis, "yaxis": yaxis}


def plot_scalar_weights(weights, plot_name, focus=None, multiplicative=False):
    """Make a barplot of the regressor weights.

    Parameters
    ----------
        weights : list
            tuples of (name, weights)
        plot_name : str
            Name of the plot Title
        focus : int
            Show weights for this forecast, if provided
        multiplicative : bool
            Flag to set y axis as percentage

    Returns
    -------
        Dictionary with Plotly traces, xaxis and yaxis
    """
    traces = []

    names = []
    values = []
    for name, weights in weights:
        names.append(name)
        weight = np.squeeze(weights)
        if len(weight.shape) > 1:
            raise ValueError("Not scalar " + plot_name)
        if len(weight.shape) == 1 and len(weight) > 1:
            if focus is not None:
                weight = weight[focus - 1]
            else:
                weight = np.mean(weight)
        values.append(weight)

    traces.append(
        go.Bar(
            name=plot_name,
            x=names,
            y=values,
            marker_color=color,
            width=0.8,
        )
    )

    xaxis = go.layout.XAxis(title=f"{plot_name} name")

    if "lagged" in plot_name.lower():
        if focus is None:
            yaxis = go.layout.YAxis(
                rangemode="normal",
                title=go.layout.yaxis.Title(text=f"{plot_name} weight (avg)"),
            )
        else:
            yaxis = go.layout.YAxis(
                rangemode="normal",
                title=go.layout.yaxis.Title(text=f"{plot_name} weight ({focus})-ahead"),
            )
    else:
        yaxis = go.layout.YAxis(
            rangemode="normal",
            title=go.layout.yaxis.Title(text=f"{plot_name} weight"),
        )

    if multiplicative:
        yaxis.update(tickformat=".1%", hoverformat=".4%")

    return {"traces": traces, "xaxis": xaxis, "yaxis": yaxis}


def plot_lagged_weights(weights, comp_name, focus=None):
    """Make a barplot of the importance of lagged inputs.

    Parameters
    ----------
        weights : list
            tuples of (name, weights)
        comp_name : str
            Name of lagged inputs
        focus : int
            Show weights for this forecast, if provided

    Returns
    -------
        Dictionary with plotly traces, xaxis and yaxis
    """
    traces = []

    n_lags = weights.shape[1]
    lags_range = list(range(1, 1 + n_lags))[::-1]
    if focus is None:
        weights = np.sum(np.abs(weights), axis=0)
        weights = weights / np.sum(weights)

        traces.append(
            go.Bar(
                name=comp_name,
                x=lags_range,
                y=weights,
                marker_color=color,
            )
        )

    else:
        if len(weights.shape) == 2:
            weights = weights[focus - 1, :]

        traces.append(go.Bar(name=comp_name, x=lags_range, y=weights, marker_color=color, width=0.8))

    padded_range = get_dynamic_axis_range(lags_range, type="numeric", inverse=True)
    xaxis = go.layout.XAxis(title=f"{comp_name} lag number", range=padded_range)

    if focus is None:
        yaxis = go.layout.YAxis(
            rangemode="normal",
            title=go.layout.yaxis.Title(text=f"{comp_name} relevance"),
            tickformat=",.0%",
        )
        # ax = set_y_as_percent(ax)
    else:
        yaxis = go.layout.YAxis(
            rangemode="normal",
            title=go.layout.yaxis.Title(text=f"{comp_name} weight ({focus})-ahead"),
        )

    return {"traces": traces, "xaxis": xaxis, "yaxis": yaxis}


def plot_yearly(m, comp_name="yearly", yearly_start=0, quick=True, multiplicative=False):
    """Plot the yearly component of the forecast.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        comp_name : str
            Name of seasonality component
        yearly_start : int
            Specifying the start day of the yearly seasonality plot

            Options
                * (default) ``yearly_start = 0``: starts the year on Jan 1
                * ``yearly_start = 1``: shifts by 1 day to Jan 2, and so on
        quick : bool
            Use quick low-level call of model
        multiplicative : bool
            Flag to set y axis as percentage

    """
    traces = []
    line_width = 2

    # Compute yearly seasonality for a Jan 1 - Dec 31 sequence of dates.
    days = pd.date_range(start="2017-01-01", periods=365) + pd.Timedelta(days=yearly_start)
    df_y = pd.DataFrame({"ds": days})
    if quick:
        predicted = predict_season_from_dates(m, dates=df_y["ds"], name=comp_name)
    else:
        predicted = m.predict_seasonal_components(df_y)[comp_name]

    traces.append(
        go.Scatter(
            name=comp_name,
            x=df_y["ds"].dt.to_pydatetime(),
            y=predicted,
            mode="lines",
            line=dict(color=color, width=line_width),
            fill="none",
        )
    )

    padded_range = get_dynamic_axis_range(df_y["ds"].dt.to_pydatetime(), type="dt")
    xaxis = go.layout.XAxis(title="Day of year", range=padded_range)
    yaxis = go.layout.YAxis(
        rangemode="normal",
        title=go.layout.yaxis.Title(text=f"Seasonality: {comp_name}"),
    )

    if multiplicative:
        yaxis.update(tickformat=".1%", hoverformat=".4%")

    return {"traces": traces, "xaxis": xaxis, "yaxis": yaxis}


def plot_weekly(m, comp_name="weekly", weekly_start=0, quick=True, multiplicative=False):
    """Plot the weekly component of the forecast.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        comp_name : str
            Name of seasonality component
        weekly_start : int
            Specifying the start day of the weekly seasonality plot

            Options
                * (default) ``weekly_start = 0``: starts the week on Sunday
                * ``weekly_start = 1``: shifts by 1 day to Monday, and so on
        quick : bool
            Use quick low-level call of model
        multplicative : bool
            Flag to set y axis as percentage

    Returns
    -------
        Dictionary with plotly traces, xaxis and yaxis

    """
    traces = []
    line_width = 2

    # Compute weekly seasonality for a Sun-Sat sequence of dates.
    days_i = pd.date_range(start="2017-01-01", periods=7 * 24, freq="H") + pd.Timedelta(days=weekly_start)
    df_w = pd.DataFrame({"ds": days_i})
    if quick:
        predicted = predict_season_from_dates(m, dates=df_w["ds"], name=comp_name)
    else:
        predicted = m.predict_seasonal_components(df_w)[comp_name]
    days = pd.date_range(start="2017-01-01", periods=8) + pd.Timedelta(days=weekly_start)
    days = days.day_name()

    traces.append(
        go.Scatter(
            name=comp_name,
            x=list(range(len(days_i))),
            y=predicted,
            mode="lines",
            line=dict(color=color, width=line_width),
            fill="none",
        )
    )
    padded_range = get_dynamic_axis_range(list(range(len(days_i))), type="numeric")
    xaxis = go.layout.XAxis(
        title="Day of week",
        tickmode="array",
        range=padded_range,
        tickvals=[x * 24 for x in range(len(days) + 1)],
        ticktext=list(days) + [days[0]],
    )
    yaxis = go.layout.YAxis(
        rangemode="normal",
        title=go.layout.yaxis.Title(text=f"Seasonality: {comp_name}"),
    )

    if multiplicative:
        yaxis.update(tickformat=".1%", hoverformat=".4%")

    return {"traces": traces, "xaxis": xaxis, "yaxis": yaxis}


def plot_daily(m, comp_name="daily", quick=True, multiplicative=False):
    """Plot the daily component of the forecast.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        comp_name : str
            Name of seasonality component if previously changed from default ``daily``
        quick : bool
            Use quick low-level call of model
        ax : matplotlib axis
            Matplotlib Axes to plot on
        multiplicative: bool
            Flag whether to set y axis as percentage

    Returns
    -------
        Dictionary with plotly traces, xaxis and yaxis
    """
    traces = []
    line_width = 2

    # Compute daily seasonality
    dates = pd.date_range(start="2017-01-01", periods=24 * 12, freq="5min")
    df = pd.DataFrame({"ds": dates})
    if quick:
        predicted = predict_season_from_dates(m, dates=df["ds"], name=comp_name)
    else:
        predicted = m.predict_seasonal_components(df)[comp_name]

    traces.append(
        go.Scatter(
            name=comp_name,
            x=np.array(range(len(dates))),
            y=predicted,
            mode="lines",
            line=dict(color=color, width=line_width),
            fill="none",
        ),
    )
    padded_range = get_dynamic_axis_range(list(range(len(dates))), type="numeric")
    xaxis = go.layout.XAxis(
        title="Hour of day",
        tickmode="array",
        range=padded_range,
        tickvals=list(np.arange(25) * 12),
        ticktext=list(np.arange(25)),
    )
    yaxis = go.layout.YAxis(
        rangemode="normal",
        title=go.layout.yaxis.Title(text=f"Seasonality: {comp_name}"),
    )

    if multiplicative:
        yaxis.update(tickformat=".1%", hoverformat=".4%")

    return {"traces": traces, "xaxis": xaxis, "yaxis": yaxis}


def plot_custom_season(m, comp_name, multiplicative=False):
    """Plot any seasonal component of the forecast.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        comp_name : str
            Name of seasonality component
        multiplicative : bool
            Flag whether to set y axis as percentage

    Returns
    -------
        Dictionary with plotly traces, xaxis and yaxis

    """

    traces = []
    line_width = 2

    t_i, predicted = predict_one_season(m, name=comp_name, n_steps=300)
    traces = []

    print(t_i)
    traces.append(
        go.Scatter(
            name=comp_name,
            x=t_i,
            y=predicted,
            mode="lines",
            line=dict(color=color, width=line_width),
            fill="none",
        )
    )
    padded_range = get_dynamic_axis_range(list(range(len(t_i))), type="numeric")
    xaxis = go.layout.XAxis(
        title=f"One period: {comp_name}",
        range=padded_range,
    )
    yaxis = go.layout.YAxis(
        rangemode="normal",
        title=go.layout.yaxis.Title(text=f"Seasonality: {comp_name}"),
    )

    if multiplicative:
        yaxis.update(tickformat=".1%", hoverformat=".4%")

    return {"traces": traces, "xaxis": xaxis, "yaxis": yaxis}


def plot_parameters(m, forecast_in_focus=None, weekly_start=0, yearly_start=0, figsize=(700, 210), df_name=None):
    """Plot the parameters that the model is composed of, visually.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        forecast_in_focus : int
            n-th step ahead forecast AR-coefficients to plot
        weekly_start : int
            Specifying the start day of the weekly seasonality plot

            Options
                * (default) ``weekly_start = 0``: starts the week on Sunday
                * ``weekly_start = 1``: shifts by 1 day to Monday, and so on
        yearly_start : int
            Specifying the start day of the yearly seasonality plot.

            Options
                * (default) ``yearly_start = 0``: starts the year on Jan 1
                * ``yearly_start = 1``: shifts by 1 day to Jan 2, and so on
        figsize : tuple
            Width, height in inches.

            Note
            ----
            Default value is set to ``None`` ->  automatic ``figsize = (700, 210 * npanel)``
        df_name : str
            Name of dataframe to refer to data params from original keys of train dataframes

            Note
            ----
            Only used for local normalization in global modeling

    Returns:
        Plotly figure
    """

    if m.config_normalization.global_normalization:
        if df_name is None:
            df_name = "__df__"
        else:
            log.debug("Global normalization set - ignoring given df_name for normalization")
    else:
        if df_name is None:
            log.warning("Local normalization set, but df_name is None. Using global data params instead.")
            df_name = "__df__"
        elif df_name not in m.config_normalization.local_data_params:
            log.warning(
                f"Local normalization set, but df_name '{df_name}' not found. Using global data params instead."
            )
            df_name = "__df__"
        else:
            log.debug(f"Local normalization set. Data params for {df_name} will be used to denormalize.")

    parameter_components = get_parameter_components(m, forecast_in_focus, df_name)

    components = parameter_components["components"]
    additive_future_regressors = parameter_components["additive_future_regressors"]
    additive_events = parameter_components["additive_events"]
    multiplicative_future_regressors = parameter_components["multiplicative_future_regressors"]
    multiplicative_events = parameter_components["multiplicative_events"]
    lagged_scalar_regressors = parameter_components["lagged_scalar_regressors"]

    npanel = len(components)
    figsize = figsize if figsize else (700, 210 * npanel)

    # Create Plotly subplot figure and add the components to it
    fig = make_subplots(npanel, cols=1, print_grid=False)
    fig.update_layout(go.Layout(showlegend=False, width=figsize[0], height=figsize[1] * npanel, **layout_args))

    if npanel == 1:
        axes = [axes]

    for i, comp in enumerate(components):
        is_multiplicative = False
        plot_name = comp["plot_name"].lower()
        if plot_name.startswith("trend"):
            if "change" in plot_name:
                # plot_trend_change(m=m, ax=ax, plot_name=comp["plot_name"])
                trace_object = plot_trend_change(m, plot_name=comp["plot_name"], df_name=df_name)
            else:
                # plot_trend(m=m, ax=ax, plot_name=comp["plot_name"])
                trace_object = plot_trend(m, plot_name=comp["plot_name"], df_name=df_name)

        elif plot_name.startswith("seasonality"):
            name = comp["comp_name"]
            if m.config_season.mode == "multiplicative":
                is_multiplicative = True
            if name.lower() == "weekly" or m.config_season.periods[name].period == 7:
                trace_object = plot_weekly(
                    m=m, weekly_start=weekly_start, comp_name=name, multiplicative=is_multiplicative
                )
            elif name.lower() == "yearly" or m.config_season.periods[name].period == 365.25:
                trace_object = plot_yearly(
                    m=m, yearly_start=yearly_start, comp_name=name, multiplicative=is_multiplicative
                )
            elif name.lower() == "daily" or m.config_season.periods[name].period == 1:
                trace_object = plot_daily(m=m, comp_name=name, multiplicative=is_multiplicative)
            else:
                trace_object = plot_custom_season(m=m, comp_name=name, multiplicative=is_multiplicative)
        elif plot_name == "lagged weights":
            trace_object = plot_lagged_weights(
                weights=comp["weights"], comp_name=comp["comp_name"], focus=comp["focus"]
            )
        else:
            if plot_name == "additive future regressor":
                weights = additive_future_regressors
            elif plot_name == "multiplicative future regressor":
                is_multiplicative = True
                weights = multiplicative_future_regressors
            elif plot_name == "lagged scalar regressor":
                weights = lagged_scalar_regressors
            elif plot_name == "additive event":
                weights = additive_events
            elif plot_name == "multiplicative event":
                is_multiplicative = True
                weights = multiplicative_events
            trace_object = plot_scalar_weights(
                weights=weights, plot_name=comp["plot_name"], focus=forecast_in_focus, multiplicative=is_multiplicative
            )

        if i == 0:
            xaxis = fig["layout"]["xaxis"]
            yaxis = fig["layout"]["yaxis"]
        else:
            xaxis = fig["layout"][f"xaxis{i + 1}"]
            yaxis = fig["layout"][f"yaxis{i + 1}"]

        xaxis.update(trace_object["xaxis"])
        yaxis.update(trace_object["yaxis"])
        xaxis.update(**xaxis_args)
        yaxis.update(**yaxis_args)
        for trace in trace_object["traces"]:
            fig.add_trace(trace, i + 1, 1)

    return fig

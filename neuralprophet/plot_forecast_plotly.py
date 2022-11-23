import logging

import numpy as np
import pandas as pd

from neuralprophet.plot_model_parameters_plotly import get_dynamic_axis_range
from neuralprophet.utils import set_y_as_percent

log = logging.getLogger("NP.plotly")

try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
except ImportError:
    log.error("Importing plotly failed. Interactive plots will not work.")

# UI Configuration
prediction_color = "#2d92ff"
actual_color = "black"
trend_color = "#B23B00"
line_width = 2
marker_size = 4
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


def plot(fcst, quantiles, xlabel="ds", ylabel="y", highlight_forecast=None, line_per_origin=False, figsize=(700, 210)):
    """
    Plot the NeuralProphet forecast

    Parameters
    ---------
        fcst : pd.DataFrame
            Output of m.predict
        quantiles: list
            Quantiles for which the forecasts are to be plotted.
        xlabel : str
            Label name on X-axis
        ylabel : str
            Label name on Y-axis
        highlight_forecast : int
            i-th step ahead forecast to highlight.
        line_per_origin : bool
            Print a line per forecast of one per forecast age
        figsize : tuple
            Width, height in inches.

    Returns
    -------
        Plotly figure
    """
    cross_marker_color = "blue"
    cross_symbol = "x"

    fcst = fcst.fillna(value=np.nan)

    ds = fcst["ds"].dt.to_pydatetime()
    colname = "yhat"
    step = 1
    # if plot_latest_forecast(), column names become "origin-x", with origin-0 being the latest forecast
    if line_per_origin:
        colname = "origin-"
        step = 0
    # all yhat column names, including quantiles
    yhat_col_names = [col_name for col_name in fcst.columns if f"{colname}" in col_name]
    # without quants
    yhat_col_names_no_qts = [
        col_name for col_name in yhat_col_names if f"{colname}" in col_name and "%" not in col_name
    ]
    data = []

    if highlight_forecast is None or line_per_origin:
        for i, yhat_col_name in enumerate(yhat_col_names_no_qts):
            data.append(
                go.Scatter(
                    name=yhat_col_name,
                    x=ds,
                    y=fcst[f"{colname}{i if line_per_origin else i + 1}"],
                    mode="lines",
                    line=dict(color=f"rgba(45, 146, 255, {0.2 + 2.0 / (i + 2.5)})", width=line_width),
                    fill="none",
                )
            )
    if len(quantiles) > 1:
        for i in range(1, len(quantiles)):
            # skip fill="tonexty" for the first quantile
            if i == 1:
                data.append(
                    go.Scatter(
                        name=f"{colname}{highlight_forecast if highlight_forecast else step} {round(quantiles[i] * 100, 1)}%",
                        x=ds,
                        y=fcst[
                            f"{colname}{highlight_forecast if highlight_forecast else step} {round(quantiles[i] * 100, 1)}%"
                        ],
                        mode="lines",
                        line=dict(color="rgba(45, 146, 255, 0.2)", width=1),
                        fillcolor="rgba(45, 146, 255, 0.2)",
                    )
                )
            else:
                data.append(
                    go.Scatter(
                        name=f"{colname}{highlight_forecast if highlight_forecast else step} {round(quantiles[i] * 100, 1)}%",
                        x=ds,
                        y=fcst[
                            f"{colname}{highlight_forecast if highlight_forecast else step} {round(quantiles[i] * 100, 1)}%"
                        ],
                        mode="lines",
                        line=dict(color="rgba(45, 146, 255, 0.2)", width=1),
                        fill="tonexty",
                        fillcolor="rgba(45, 146, 255, 0.2)",
                    )
                )

    if highlight_forecast is not None:
        if line_per_origin:
            num_forecast_steps = sum(fcst["origin-0"].notna())
            steps_from_last = num_forecast_steps - highlight_forecast
            for i, yhat_col_name in enumerate(yhat_col_names_no_qts):
                x = [ds[-(1 + i + steps_from_last)]]
                y = [fcst[f"origin-{i}"].values[-(1 + i + steps_from_last)]]
                data.append(
                    go.Scatter(
                        name=yhat_col_name,
                        x=x,
                        y=y,
                        mode="markers",
                        marker=dict(color=cross_marker_color, size=marker_size, symbol=cross_symbol),
                    )
                )
        else:
            x = ds
            y = fcst[f"yhat{highlight_forecast}"]
            data.append(
                go.Scatter(
                    name="Predicted",
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color=prediction_color, width=line_width),
                )
            )
            data.append(
                go.Scatter(
                    name="Predicted",
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(color=cross_marker_color, size=marker_size, symbol=cross_symbol),
                )
            )

    # Add actual
    data.append(
        go.Scatter(name="Actual", x=ds, y=fcst["y"], marker=dict(color=actual_color, size=marker_size), mode="markers")
    )

    # Plot trend
    # if trend:
    #    data.append(
    #        go.Scatter(
    #            name="Trend",
    #            x=fcst["ds"],
    #            y=fcst["trend"],
    #            mode="lines",
    #            line=dict(color=trend_color, width=line_width),
    #        )
    #    )

    layout = go.Layout(
        showlegend=True,
        width=figsize[0],
        height=figsize[1],
        xaxis=go.layout.XAxis(
            title=xlabel,
            type="date",
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            rangeslider=dict(visible=True),
            **xaxis_args,
        ),
        yaxis=go.layout.YAxis(title=ylabel, **yaxis_args),
        **layout_args,
    )
    fig = go.Figure(data=data, layout=layout)

    return fig


def plot_components(m, fcst, plot_configuration, df_name="__df__", one_period_per_season=False, figsize=(700, 210)):
    """
    Plot the NeuralProphet forecast components.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        fcst : pd.DataFrame
            Output of m.predict
        plot_configuration: dict
            dict of configured components to plot
        df_name : str
            ID from time series that should be plotted
        one_period_per_season : bool
            Plot one period per season, instead of the true seasonal components of the forecast.
        figsize : tuple
            Width, height in inches.

    Returns
    -------
        Plotly figure
    """
    log.debug("Plotting forecast components")
    fcst = fcst.fillna(value=np.nan)
    components_to_plot = plot_configuration["components_list"]

    # set number of axes based on selected plot_names and sort them according to order in components
    panel_names = list(set(next(iter(dic.values())).lower() for dic in components_to_plot))
    panel_order = [x for dic in components_to_plot for x in panel_names if x in dic["plot_name"].lower()]
    npanel = len(panel_names)
    figsize = figsize if figsize else (700, 210 * npanel)

    # Create Plotly subplot figure and add the components to it
    fig = make_subplots(npanel, cols=1, print_grid=False)
    fig.update_layout(
        go.Layout(
            # showlegend=False, #set individually instead
            width=figsize[0],
            height=figsize[1] * npanel,
            **layout_args,
        )
    )

    multiplicative_axes = []
    for comp in components_to_plot:
        name = comp["plot_name"].lower()
        j = panel_order.index(name)

        if (
            name in ["trend"]
            or ("ar" in name and "ahead" in name)
            or ("lagged_regressor" in name and "ahead" in name)
            or ("uncertainty" in name)
        ):
            trace_object = get_forecast_component_props(fcst=fcst, **comp)

        elif "event" in name or "future regressor" in name:
            trace_object = get_forecast_component_props(fcst=fcst, **comp)

        elif "season" in name:
            if m.config_season.mode == "multiplicative":
                comp.update({"multiplicative": True})
            if one_period_per_season:
                comp_name = comp["comp_name"]
                trace_object = get_seasonality_props(m, fcst, df_name, **comp)
            else:
                comp_name = f"season_{comp['comp_name']}"
                trace_object = get_forecast_component_props(fcst=fcst, comp_name=comp_name, plot_name=comp["plot_name"])

        elif "auto-regression" in name or "lagged regressor" in name:
            trace_object = get_multiforecast_component_props(fcst=fcst, **comp)
            fig.update_layout(barmode="overlay")

        if j == 0:
            xaxis = fig["layout"]["xaxis"]
            yaxis = fig["layout"]["yaxis"]
        else:
            xaxis = fig["layout"][f"xaxis{j + 1}"]
            yaxis = fig["layout"][f"yaxis{j + 1}"]

        xaxis.update(trace_object["xaxis"])
        xaxis.update(**xaxis_args)
        yaxis.update(trace_object["yaxis"])
        yaxis.update(**yaxis_args)
        for trace in trace_object["traces"]:
            fig.add_trace(trace, j + 1, 1)
        fig.update_layout(legend={"y": 0.1, "traceorder": "reversed"})

    # Reset multiplicative axes labels after tight_layout adjustment
    for ax in multiplicative_axes:
        ax = set_y_as_percent(ax)
    return fig


def get_forecast_component_props(
    fcst,
    comp_name,
    plot_name=None,
    multiplicative=False,
    bar=False,
    rolling=None,
    add_x=False,
    fill=False,
    num_overplot=None,
    **kwargs,
):
    """
    Prepares a dictionary for plotting the selected forecast component with plotly.

    Parameters
    ----------
        fcst : pd.DataFrame
            Output of m.predict
        comp_name : str
            Name of the component to plot
        plot_name : str
            Name of the plot
        multiplicative : bool
            Flag whetther to plot the y-axis as percentage
        bar : bool
            Flag whether to plot the component as a bar
        rolling : int
            Rolling average to underplot
        add_x : bool
            Flag whether to add x-symbols to the plotted points
        fill : bool
            Add fill between signal and x(y=0) axis
        num_overplot: int
            the number of forecast in focus

    Returns
    -------
        Dictionary with plotly traces, xaxis and yaxis
    """
    cross_symbol = "x"
    cross_marker_color = "blue"

    if plot_name is None:
        plot_name = comp_name

    # Remove empty rows for the respective component
    fcst = fcst.loc[fcst[comp_name].notna()]

    text = None
    mode = "lines"
    fcst_t = fcst["ds"].dt.to_pydatetime()

    traces = []
    if rolling is not None:
        rolling_avg = fcst[comp_name].rolling(rolling, min_periods=1, center=True).mean()
        if bar:
            traces.append(
                go.Bar(
                    name=plot_name,
                    x=fcst_t,
                    y=rolling_avg,
                    text=text,
                    color=prediction_color,
                    opacity=0.5,
                    showlegend=False,
                )
            )
        else:
            traces.append(
                go.Scatter(
                    name=plot_name,
                    x=fcst_t,
                    y=rolling_avg,
                    mode=mode,
                    line=go.scatter.Line(color=prediction_color, width=line_width),
                    text=text,
                    opacity=0.5,
                    showlegend=False,
                )
            )

            if add_x:
                traces.append(
                    go.Scatter(
                        x=fcst_t,
                        y=fcst[comp_name],
                        mode="markers",
                        marker=dict(color=cross_marker_color, size=marker_size, symbol=cross_symbol),
                        showlegend=False,
                    )
                )

    y = fcst[comp_name].values

    if "uncertainty" in plot_name.lower():
        if num_overplot is not None:
            y = fcst[comp_name].values - fcst[f"yhat{num_overplot}"].values
        else:
            y = fcst[comp_name].values - fcst["yhat1"].values
    if bar:
        traces.append(
            go.Bar(
                name=plot_name,
                x=fcst_t,
                y=y,
                text=text,
                marker_color=prediction_color,
                showlegend=False,
            )
        )
    elif "uncertainty" in plot_name.lower() and fill:
        filling = "tozeroy"
        traces.append(
            go.Scatter(
                name=comp_name,
                x=fcst_t,
                y=y,
                text=text,
                fill=filling,
                mode="lines",
                line=dict(color="rgba(45, 146, 255, 0.2)", width=1),
                fillcolor="rgba(45, 146, 255, 0.2)",
                showlegend=True,
            )
        )
    else:
        traces.append(
            go.Scatter(
                name=plot_name,
                x=fcst_t,
                y=y,
                mode=mode,
                line=go.scatter.Line(color=prediction_color, width=line_width),
                text=text,
                showlegend=False,
            )
        )

        if add_x:
            traces.append(
                go.Scatter(
                    x=fcst_t,
                    y=fcst[comp_name],
                    mode="markers",
                    marker=dict(color=cross_marker_color, size=marker_size, symbol=cross_symbol),
                    showlegend=False,
                )
            )
    padded_range = get_dynamic_axis_range(list(fcst["ds"]), type="dt")
    xaxis = go.layout.XAxis(title="ds", type="date", range=padded_range)
    yaxis = go.layout.YAxis(
        title=plot_name,
        rangemode="normal" if comp_name == "trend" else "tozero",
    )

    if multiplicative:
        yaxis.update(tickformat=".1%", hoverformat=".4%")

    return {"traces": traces, "xaxis": xaxis, "yaxis": yaxis}


def get_multiforecast_component_props(
    fcst, comp_name, plot_name=None, multiplicative=False, bar=False, focus=1, num_overplot=None, **kwargs
):
    """
    Prepares a dictionary for plotting the selected multi forecast component with plotly

    Parameters
    ----------
        fcst : pd.DataFrame
            Output of m.predict
        comp_name : str
            Name of the component to plot
        plot_name : str
            Name of the plot
        multiplicative : bool
            Flag whetther to plot the y-axis as percentage
        bar : bool
            Flag whether to plot the component as a bar
        focus : int
            Id of the forecast to display
        add_x : bool
            Flag whether to add x-symbols to the plotted points

    Returns
    -------
        Dictionary with plotly traces, xaxis and yaxis
    """
    if plot_name is None:
        plot_name = comp_name

    # Remove empty rows for the respective components
    if num_overplot:
        fcst = fcst.loc[(fcst[f"{comp_name}1"].notna()) | (fcst[f"{comp_name}{num_overplot}"].notna())]
    else:
        fcst = fcst.loc[fcst[comp_name].notna()]

    text = None
    mode = "lines"
    fcst_t = fcst["ds"].dt.to_pydatetime()
    col_names = [col_name for col_name in fcst.columns if col_name.startswith(comp_name)]
    traces = []

    if num_overplot is not None:
        assert num_overplot <= len(col_names)
        for i in list(range(num_overplot))[::-1]:
            y = fcst[f"{comp_name}{i+1}"]
            y = y.values
            alpha_min = 0.2
            alpha_softness = 1.2
            alpha = alpha_min + alpha_softness * (1.0 - alpha_min) / (i + 1.0 * alpha_softness)
            y[-1] = 0

            if bar:
                traces.append(
                    go.Bar(
                        name=plot_name,
                        x=fcst_t,
                        y=y,
                        text=text,
                        marker_color=prediction_color,
                        opacity=alpha,
                        showlegend=False,
                    )
                )

            else:
                traces.append(
                    go.Scatter(
                        name=plot_name,
                        x=fcst_t,
                        y=y,
                        mode=mode,
                        line=go.scatter.Line(color=prediction_color, width=line_width),
                        text=text,
                        opacity=alpha,
                        showlegend=False,
                    )
                )

    if num_overplot is None or focus > 1:

        y = fcst[f"{comp_name}"]
        y = y.values
        y[-1] = 0
        if bar:
            traces.append(
                go.Bar(
                    name=plot_name,
                    x=fcst_t,
                    y=y,
                    text=text,
                    marker_color=prediction_color,
                    showlegend=False,
                )
            )
        else:
            traces.append(
                go.Scatter(
                    name=plot_name,
                    x=fcst_t,
                    y=y,
                    mode=mode,
                    line=go.scatter.Line(color=prediction_color, width=line_width),
                    text=text,
                    showlegend=False,
                )
            )

    padded_range = get_dynamic_axis_range(list(fcst["ds"]), type="dt")
    xaxis = go.layout.XAxis(title="ds", type="date", range=padded_range)
    yaxis = go.layout.YAxis(
        rangemode="normal" if comp_name == "trend" else "tozero",
        title=plot_name,
    )

    if multiplicative:
        yaxis.update(tickformat=".1%", hoverformat=".4%")

    return {"traces": traces, "xaxis": xaxis, "yaxis": yaxis}


def get_seasonality_props(m, fcst, df_name="__df__", comp_name="weekly", multiplicative=False, quick=False, **kwargs):
    """
    Prepares a dictionary for plotting the selected seasonality with plotly

    Parameters
    ----------
        m : NeuralProphet
            Fitted NeuralProphet model
        fcst : pd.DataFrame
            Output of m.predict
        df_name : str
            ID from time series that should be plotted
        comp_name : str
            Name of the component to plot
        multiplicative : bool
            Flag whetther to plot the y-axis as percentage
        quick : bool
            Use quick low-level call of model

    Returns
    -------
        Dictionary with plotly traces, xaxis and yaxis
    """
    # Compute seasonality from Jan 1 through a single period.
    start = pd.to_datetime("2017-01-01 0000")

    period = m.config_season.periods[comp_name].period
    if m.data_freq == "B":
        period = 5
        start += pd.Timedelta(days=1)

    end = start + pd.Timedelta(days=period)
    if (fcst["ds"].dt.hour == 0).all():  # Day Precision
        plot_points = np.floor(period * 24).astype(int)
    elif (fcst["ds"].dt.minute == 0).all():  # Hour Precision
        plot_points = np.floor(period * 24 * 24).astype(int)
    else:  # Minute Precision
        plot_points = np.floor(period * 24 * 60).astype(int)
    days = pd.to_datetime(np.linspace(start.value, end.value, plot_points, endpoint=False))
    df_y = pd.DataFrame({"ds": days})
    df_y["ID"] = df_name

    if quick:
        predicted = m.predict_season_from_dates(m, dates=df_y["ds"], name=comp_name)
    else:
        predicted = m.predict_seasonal_components(df_y)[comp_name]

    traces = []
    traces.append(
        go.Scatter(
            name="Seasonality: " + comp_name,
            x=df_y["ds"],
            y=predicted,
            mode="lines",
            line=go.scatter.Line(color=prediction_color, width=line_width, shape="spline", smoothing=1),
            showlegend=False,
        )
    )

    # Set tick formats (examples are based on 2017-01-06 21:15)
    if period <= 2:
        tickformat = "%H:%M"  # "21:15"
    elif period < 7:
        tickformat = "%A %H:%M"  # "Friday 21:15"
    elif period < 14:
        tickformat = "%A"  # "Friday"
    else:
        tickformat = "%B"  # "January  6"

    padded_range = get_dynamic_axis_range(list(df_y["ds"]), type="dt")
    xaxis = go.layout.XAxis(
        title=f"Day of {comp_name[:-2]}" if comp_name[-2:] == "ly" else f"Day of {comp_name}",
        tickformat=tickformat,
        type="date",
        range=padded_range,
    )

    yaxis = go.layout.YAxis(
        title="Seasonality: " + comp_name,
    )

    if multiplicative:
        yaxis.update(tickformat=".1%", hoverformat=".4%")

    return {"traces": traces, "xaxis": xaxis, "yaxis": yaxis}


def check_if_configured(m, components, error_flag=False):
    """Check if components were set in the model configuration by the user.

    Parameters
    ----------
        m : NeuralProphet
            Fitted NeuralProphet model
        components : str or list, optional
            name or list of names of components to check

            Options
            ----
            * ``trend``
            * ``trend_rate_change``
            * ``seasonality``
            * ``autoregression``
            * ``lagged_regressors```
            * ``events``
            * ``future_regressors`
            * ``uncertainty``
        error_flag : bool
            Activate to raise a ValueError if component has not been configured

    Returns
    -------
        components
            list of components only including the components set in the model configuration
    """
    invalid_components = []
    if "trend_rate_change" in components and m.model.config_trend.changepoints is None:
        components.remove("trend_rate_change")
        invalid_components.append("trend_rate_change")
    if "seasonality" in components and m.config_season is None:
        components.remove("seasonality")
        invalid_components.append("seasonality")
    if "autoregression" in components and not m.config_ar.n_lags > 0:
        components.remove("autoregression")
        invalid_components.append("autoregression")
    if "lagged_regressors" in components and m.config_lagged_regressors is None:
        components.remove("lagged_regressors")
        invalid_components.append("lagged_regressors")
    if "events" in components and (m.config_events and m.config_country_holidays) is None:
        components.remove("events")
        invalid_components.append("events")
    if "future_regressors" in components and m.config_regressors is None:
        components.remove("future_regressors")
        invalid_components.append("future_regressors")
    if "uncertainty" in components and not len(m.model.quantiles) > 1:
        components.remove("uncertainty")
        invalid_components.append("uncertainty")
    if error_flag and len(invalid_components) != 0:
        raise ValueError(
            f" Selected component(s) {(invalid_components)} for plotting not specified in the model configuration."
        )
    return components


def get_valid_configuration(
    m, components=None, df_name=None, valid_set=None, validator=None, forecast_in_focus=None, quantile=0.5
):
    """Validate and adapt the selected components to be plotted.

    Parameters
    ----------
        m : NeuralProphet
            Fitted NeuralProphet model
        components : str or list, optional
            name or list of names of components to validate and adapt
        df_name: str
            ID from time series that should be plotted
        valid_set : str or list, optional
            name or list of names of components that are defined as valid option

            Options
            ----
             * (default)``None``:  All components the user set in the model configuration are validated and adapted
            * ``trend``
            * ``seasonality``
            * ``autoregression``
            * ``lagged_regressors``
            * ``future_regressors``
            * ``events``
            * ``uncertainty``
        validator: str
            specifies the validation purpose to customize

            Options
            ----
            * ``plot_parameters``: customize for plot_parameters() function
            * ``plot_components``: customize for plot_components() function
        forecast_in_focus: int
            optinal, i-th step ahead forecast to plot

            Note
            ----
            None (default): plot self.highlight_forecast_step_n by default
        quantile: float
            The quantile for which the model parameters are to be plotted

            Note
            ----
            0.5 (default):  Parameters will be plotted for the median quantile.

    Returns
    -------
        valid_configuration: dict
            dict of validated components and values to be plotted
    """
    if type(valid_set) is not list:
        valid_set = [valid_set]

    if components is None:
        components = valid_set
        components = check_if_configured(m=m, components=components)
    else:
        if type(components) is not list:
            components = [components]
        components = [comp.lower() for comp in components]
        for comp in components:
            if comp not in valid_set:
                raise ValueError(
                    f" Selected component {comp} is either mis-spelled or not an available "
                    f"option for this function."
                )
        components = check_if_configured(m=m, components=components, error_flag=True)
    if validator is None:
        raise ValueError("Specify a validator from the available options")
    # Adapt Normalization
    if validator == "plot_parameters":
        # Set to True in case of local normalization and unknown_data_params is not True
        overwriting_unknown_data_normalization = False
        if m.config_normalization.global_normalization:
            if df_name is None:
                df_name = "__df__"
            else:
                log.debug("Global normalization set - ignoring given df_name for normalization")
        else:
            if df_name is None:
                log.warning("Local normalization set, but df_name is None. Using global data params instead.")
                df_name = "__df__"
                if not m.config_normalization.unknown_data_normalization:
                    m.config_normalization.unknown_data_normalization = True
                    overwriting_unknown_data_normalization = True
            elif df_name not in m.config_normalization.local_data_params:
                log.warning(
                    f"Local normalization set, but df_name '{df_name}' not found. Using global data params instead."
                )
                df_name = "__df__"
                if not m.config_normalization.unknown_data_normalization:
                    m.config_normalization.unknown_data_normalization = True
                    overwriting_unknown_data_normalization = True
            else:
                log.debug(f"Local normalization set. Data params for {df_name} will be used to denormalize.")

    # Identify components to be plotted
    # as dict, minimum: {plot_name}
    plot_components = []
    if validator == "plot_parameters":
        quantile_index = m.model.quantiles.index(quantile)

    # Plot trend
    if "trend" in components:
        plot_components.append({"plot_name": "Trend", "comp_name": "trend"})
    if "trend_rate_change" in components:
        plot_components.append({"plot_name": "Trend Rate Change"})

    # Plot  seasonalities, if present
    if "seasonality" in components:
        for name in m.config_season.periods:
            if validator == "plot_components":
                plot_components.append(
                    {
                        "plot_name": f"{name} seasonality",
                        "comp_name": name,
                    }
                )
            elif validator == "plot_parameters":
                plot_components.append({"plot_name": "seasonality", "comp_name": name})

    # AR
    if "autoregression" in components:
        if validator == "plot_components":
            if forecast_in_focus is None:
                plot_components.append(
                    {
                        "plot_name": "Auto-Regression",
                        "comp_name": "ar",
                        "num_overplot": m.n_forecasts,
                        "bar": True,
                    }
                )
            else:
                plot_components.append(
                    {
                        "plot_name": f"AR ({forecast_in_focus})-ahead",
                        "comp_name": f"ar{forecast_in_focus}",
                    }
                )
        elif validator == "plot_parameters":
            plot_components.append(
                {
                    "plot_name": "lagged weights",
                    "comp_name": "AR",
                    "weights": m.model.ar_weights.detach().numpy(),
                    "focus": forecast_in_focus,
                }
            )

    # Add lagged regressors
    lagged_scalar_regressors = []
    if "lagged_regressors" in components:
        if validator == "plot_components":
            if forecast_in_focus is None:
                for name in m.config_lagged_regressors.keys():
                    plot_components.append(
                        {
                            "plot_name": f'Lagged Regressor "{name}"',
                            "comp_name": f"lagged_regressor_{name}",
                            "num_overplot": m.n_forecasts,
                            "bar": True,
                        }
                    )
            else:
                for name in m.config_lagged_regressors.keys():
                    plot_components.append(
                        {
                            "plot_name": f'Lagged Regressor "{name}" ({forecast_in_focus})-ahead',
                            "comp_name": f"lagged_regressor_{name}{forecast_in_focus}",
                        }
                    )
        elif validator == "plot_parameters":
            for name in m.config_lagged_regressors.keys():
                if m.config_lagged_regressors[name].as_scalar:
                    lagged_scalar_regressors.append((name, m.model.get_covar_weights(name).detach().numpy()))
                else:
                    plot_components.append(
                        {
                            "plot_name": "lagged weights",
                            "comp_name": f'Lagged Regressor "{name}"',
                            "weights": m.model.get_covar_weights(name).detach().numpy(),
                            "focus": forecast_in_focus,
                        }
                    )

    # Add Events
    additive_events = []
    multiplicative_events = []
    if "events" in components:
        additive_events_flag = False
        muliplicative_events_flag = False
        for event, configs in m.config_events.items():
            if validator == "plot_components" and configs.mode == "additive":
                additive_events_flag = True
            elif validator == "plot_components" and configs.mode == "multiplicative":
                muliplicative_events_flag = True
            elif validator == "plot_parameters":
                event_params = m.model.get_event_weights(event)
                weight_list = [(key, param.detach().numpy()[quantile_index, :]) for key, param in event_params.items()]
                if configs.mode == "additive":
                    additive_events = additive_events + weight_list
                elif configs.mode == "multiplicative":
                    multiplicative_events = multiplicative_events + weight_list

        for country_holiday in m.config_country_holidays.holiday_names:
            if validator == "plot_components" and m.config_country_holidays.mode == "additive":
                additive_events_flag = True
            elif validator == "plot_components" and m.config_country_holidays.mode == "multiplicative":
                muliplicative_events_flag = True
            elif validator == "plot_parameters":
                event_params = m.model.get_event_weights(country_holiday)
                weight_list = [(key, param.detach().numpy()[quantile_index, :]) for key, param in event_params.items()]
                if m.config_country_holidays.mode == "additive":
                    additive_events = additive_events + weight_list
                elif m.config_country_holidays.mode == "multiplicative":
                    multiplicative_events = multiplicative_events + weight_list

        if additive_events_flag:
            plot_components.append(
                {
                    "plot_name": "Additive Events",
                    "comp_name": "events_additive",
                }
            )
        if muliplicative_events_flag:
            plot_components.append(
                {
                    "plot_name": "Multiplicative Events",
                    "comp_name": "events_multiplicative",
                    "multiplicative": True,
                }
            )

    # Add Regressors
    additive_future_regressors = []
    multiplicative_future_regressors = []
    if "future_regressors" in components:
        for regressor, configs in m.config_regressors.items():
            if validator == "plot_components" and configs.mode == "additive":
                plot_components.append(
                    {
                        "plot_name": "Additive Future Regressors",
                        "comp_name": "future_regressors_additive",
                    }
                )
            elif validator == "plot_components" and configs.mode == "multiplicative":
                plot_components.append(
                    {
                        "plot_name": "Multiplicative Future Regressors",
                        "comp_name": "future_regressors_multiplicative",
                        "multiplicative": True,
                    }
                )
            elif validator == "plot_parameters":
                regressor_param = m.model.get_reg_weights(regressor)[quantile_index, :]
                if configs.mode == "additive":
                    additive_future_regressors.append((regressor, regressor_param.detach().numpy()))
                elif configs.mode == "multiplicative":
                    multiplicative_future_regressors.append((regressor, regressor_param.detach().numpy()))

    # Plot  quantiles as a separate component, if present
    # If multiple steps in the future are predicted, only plot quantiles if highlight_forecast_step_n is set
    if (
        "quantiles" in components
        and validator == "plot_components"
        and len(m.model.quantiles) > 1
        and forecast_in_focus is None
    ):
        if len(m.config_train.quantiles) > 1 and (
            m.n_forecasts > 1 or m.config_ar.n_lags > 0
        ):  # rather query if n_forecasts >1 than n_lags>1
            raise ValueError(
                "Please specify step_number using the highlight_nth_step_ahead_of_each_forecast function"
                " for quantiles plotting when autoregression enabled."
            )
        for i in range(1, len(m.model.quantiles)):
            plot_components.append(
                {
                    "plot_name": "Uncertainty",
                    "comp_name": f"yhat1 {round(m.model.quantiles[i] * 100, 1)}%",
                    "fill": True,
                }
            )
    elif (
        "uncertainty" in components
        and validator == "plot_components"
        and len(m.model.quantiles) > 1
        and forecast_in_focus is not None
    ):
        for i in range(1, len(m.model.quantiles)):
            plot_components.append(
                {
                    "plot_name": "Uncertainty",
                    "comp_name": f"yhat{forecast_in_focus} {round(m.model.quantiles[i] * 100, 1)}%",
                    "fill": True,
                }
            )
    if validator == "plot_parameters":
        if len(additive_future_regressors) > 0:
            plot_components.append({"plot_name": "Additive future regressor"})
        if len(multiplicative_future_regressors) > 0:
            plot_components.append({"plot_name": "Multiplicative future regressor"})
        if len(lagged_scalar_regressors) > 0:
            plot_components.append({"plot_name": "Lagged scalar regressor"})
        if len(additive_events) > 0:
            data_params = m.config_normalization.get_data_params(df_name)
            scale = data_params["y"].scale
            additive_events = [(key, weight * scale) for (key, weight) in additive_events]
            plot_components.append({"plot_name": "Additive event"})
        if len(multiplicative_events) > 0:
            plot_components.append({"plot_name": "Multiplicative event"})

        valid_configuration = {
            "components_list": plot_components,
            "additive_future_regressors": additive_future_regressors,
            "additive_events": additive_events,
            "multiplicative_future_regressors": multiplicative_future_regressors,
            "multiplicative_events": multiplicative_events,
            "lagged_scalar_regressors": lagged_scalar_regressors,
            "overwriting_unknown_data_normalization": overwriting_unknown_data_normalization,
            "df_name": df_name,
        }
    elif validator == "plot_components":
        valid_configuration = {
            "components_list": plot_components,
        }
    return valid_configuration

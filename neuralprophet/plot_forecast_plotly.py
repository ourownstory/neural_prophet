import logging

import numpy as np
import pandas as pd

from neuralprophet.plot_model_parameters_plotly import get_dynamic_axis_range
from neuralprophet.plot_utils import set_y_as_percent

log = logging.getLogger("NP.plotly")

try:
    import plotly.express as px
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
            trace_object = get_forecast_component_props(fcst=fcst, df_name=df_name, **comp)

        elif "event" in name or "future regressor" in name:
            trace_object = get_forecast_component_props(fcst=fcst, df_name=df_name, **comp)

        elif "season" in name:
            if m.config_season.mode == "multiplicative":
                comp.update({"multiplicative": True})
            if one_period_per_season:
                comp_name = comp["comp_name"]
                trace_object = get_seasonality_props(m, fcst, df_name, **comp)
            else:
                comp_name = f"season_{comp['comp_name']}"
                trace_object = get_forecast_component_props(
                    fcst=fcst, df_name=df_name, comp_name=comp_name, plot_name=comp["plot_name"]
                )

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
        predicted = m.predict_seasonal_components(df_y)[["ds", "ID", comp_name]]

    traces = []
    traces.append(
        go.Scatter(
            name="Seasonality: " + comp_name,
            x=df_y["ds"],
            y=predicted[comp_name],
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


def plot_nonconformity_scores(scores, alpha, q, method):
    """Plot the NeuralProphet forecast components.

    Parameters
    ----------
        scores : list
            nonconformity scores
        alpha : float
            user-specified significance level of the prediction interval
        q : float
            prediction interval width (or q)
        method : str
            name of conformal prediction technique used

            Options
                * (default) ``naive``: Naive or Absolute Residual
                * ``cqr``: Conformalized Quantile Regression

    Returns
    -------
        plotly.graph_objects.Figure
            Figure showing the nonconformity score with horizontal line for q-value based on the significance level or alpha
    """
    confidence_levels = np.arange(len(scores)) / len(scores)
    fig = px.line(
        pd.DataFrame({"Confidence Level": confidence_levels, "One-Sided Interval Width": scores}),
        x="Confidence Level",
        y="One-Sided Interval Width",
        title=f"{method} One-Sided Interval Width with q",
        width=600,
        height=400,
    )
    fig.add_vline(
        x=1 - alpha,
        annotation_text=f"(1-alpha) = {1-alpha}",
        annotation_position="top left",
        line_width=1,
        line_color="green",
    )
    fig.add_hline(
        y=q, annotation_text=f"q1 = {round(q, 2)}", annotation_position="top left", line_width=1, line_color="red"
    )
    fig.update_layout(margin=dict(l=70, r=70, t=60, b=50))
    return fig

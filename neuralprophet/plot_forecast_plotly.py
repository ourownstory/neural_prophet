import numpy as np
import pandas as pd
import logging
from neuralprophet.utils import set_y_as_percent
from neuralprophet.plot_model_parameters_plotly import get_dynamic_axis_range

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
    yhat_col_names = [col_name for col_name in fcst.columns if "yhat" in col_name]

    data = []

    if highlight_forecast is None or line_per_origin:
        for i, yhat_col_name in enumerate(yhat_col_names):
            if "%" not in yhat_col_name:
                data.append(
                    go.Scatter(
                        name=yhat_col_name,
                        x=ds,
                        y=fcst[f"yhat{i + 1}"],
                        mode="lines",
                        line=dict(color=f"rgba(45, 146, 255, {0.2 + 2.0 / (i + 2.5)})", width=line_width),
                        fill="none",
                    )
                )
    if len(quantiles) > 1 and not line_per_origin:
        for i in range(1, len(quantiles)):
            # skip fill="tonexty" for the first quantile
            if i == 1:
                data.append(
                    go.Scatter(
                        name=f"yhat{highlight_forecast if highlight_forecast else 1} {quantiles[i] * 100}%",
                        x=ds,
                        y=fcst[f"yhat{highlight_forecast if highlight_forecast else 1} {quantiles[i] * 100}%"],
                        mode="lines",
                        line=dict(color="rgba(45, 146, 255, 0.2)", width=1),
                        fillcolor="rgba(45, 146, 255, 0.2)",
                    )
                )
            else:
                data.append(
                    go.Scatter(
                        name=f"yhat{highlight_forecast if highlight_forecast else 1} {quantiles[i] * 100}%",
                        x=ds,
                        y=fcst[f"yhat{highlight_forecast if highlight_forecast else 1} {quantiles[i] * 100}%"],
                        mode="lines",
                        line=dict(color="rgba(45, 146, 255, 0.2)", width=1),
                        fill="tonexty",
                        fillcolor="rgba(45, 146, 255, 0.2)",
                    )
                )

    if highlight_forecast is not None:
        if line_per_origin:
            num_forecast_steps = sum(fcst["yhat1"].notna())
            steps_from_last = num_forecast_steps - highlight_forecast
            for i, yhat_col_name in enumerate(yhat_col_names):
                x = [ds[-(1 + i + steps_from_last)]]
                y = [fcst[f"yhat{(i + 1)}"].values[-(1 + i + steps_from_last)]]
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


def plot_components(m, fcst, forecast_in_focus=None, one_period_per_season=True, residuals=False, figsize=(700, 210)):
    """
    Plot the NeuralProphet forecast components.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        fcst : pd.DataFrame
            Output of m.predict
        forecast_in_focus : int
            n-th step ahead forecast AR-coefficients to plot
        one_period_per_season : bool
            Plot one period per season, instead of the true seasonal components of the forecast.
        residuals : bool
            Flag whether to plot the residuals or not.
        figsize : tuple
            Width, height in inches.

    Returns
    -------
        Plotly figure
    """
    log.debug("Plotting forecast components")
    fcst = fcst.fillna(value=np.nan)

    # Identify components to be plotted
    # as dict, minimum: {plot_name, comp_name}
    components = []

    # Plot  trend
    components.append({"plot_name": "Trend", "comp_name": "trend"})

    # Plot  seasonalities, if present
    if m.model.config_season is not None:
        for name in m.model.config_season.periods:
            components.append(
                {
                    "plot_name": f"{name} seasonality",
                    "comp_name": name,
                }
            )
    # AR
    if m.model.n_lags > 0:
        if forecast_in_focus is None:
            components.append(
                {
                    "plot_name": "Auto-Regression",
                    "comp_name": "ar",
                    "num_overplot": m.n_forecasts,
                    "bar": True,
                }
            )
        else:
            components.append(
                {
                    "plot_name": f"AR ({forecast_in_focus})-ahead",
                    "comp_name": f"ar{forecast_in_focus}",
                }
            )

    # Add Covariates
    if m.model.config_covar is not None:
        for name in m.model.config_covar.keys():
            if forecast_in_focus is None:
                components.append(
                    {
                        "plot_name": f'Lagged Regressor "{name}"',
                        "comp_name": f"lagged_regressor_{name}",
                        "num_overplot": m.n_forecasts,
                        "bar": True,
                    }
                )
            else:
                components.append(
                    {
                        "plot_name": f'Lagged Regressor "{name}" ({forecast_in_focus})-ahead',
                        "comp_name": f"lagged_regressor_{name}{forecast_in_focus}",
                    }
                )
                # 'add_x': True})
    # Add Events
    if "events_additive" in fcst.columns:
        components.append(
            {
                "plot_name": "Additive Events",
                "comp_name": "events_additive",
            }
        )
    if "events_multiplicative" in fcst.columns:
        components.append(
            {
                "plot_name": "Multiplicative Events",
                "comp_name": "events_multiplicative",
                "multiplicative": True,
            }
        )

    # Add Regressors
    if "future_regressors_additive" in fcst.columns:
        components.append(
            {
                "plot_name": "Additive Future Regressors",
                "comp_name": "future_regressors_additive",
            }
        )
    if "future_regressors_multiplicative" in fcst.columns:
        components.append(
            {
                "plot_name": "Multiplicative Future Regressors",
                "comp_name": "future_regressors_multiplicative",
                "multiplicative": True,
            }
        )
    if residuals:
        if forecast_in_focus is None and m.n_forecasts > 1:
            if fcst["residual1"].count() > 0:
                components.append(
                    {
                        "plot_name": "Residuals",
                        "comp_name": "residual",
                        "num_overplot": m.n_forecasts,
                        "bar": True,
                    }
                )
        else:
            ahead = 1 if forecast_in_focus is None else forecast_in_focus
            if fcst[f"residual{ahead}"].count() > 0:
                components.append(
                    {
                        "plot_name": f"Residuals ({ahead})-ahead",
                        "comp_name": f"residual{ahead}",
                        "bar": True,
                    }
                )

    npanel = len(components)
    figsize = figsize if figsize else (700, 210 * npanel)

    # Create Plotly subplot figure and add the components to it
    fig = make_subplots(npanel, cols=1, print_grid=False)
    fig.update_layout(
        go.Layout(
            showlegend=False,
            width=figsize[0],
            height=figsize[1] * npanel,
            **layout_args,
        )
    )

    multiplicative_axes = []
    for i, comp in enumerate(components):
        name = comp["plot_name"].lower()
        ploty_trace = None

        if (
            name in ["trend"]
            or ("residuals" in name and "ahead" in name)
            or ("ar" in name and "ahead" in name)
            or ("lagged_regressor" in name and "ahead" in name)
        ):
            trace_object = get_forecast_component_props(fcst=fcst, **comp)

        elif "event" in name or "future regressor" in name:
            trace_object = get_forecast_component_props(fcst=fcst, **comp)

        elif "season" in name:
            if m.config_season.mode == "multiplicative":
                comp.update({"multiplicative": True})
            if one_period_per_season:
                comp_name = comp["comp_name"]
                trace_object = get_seasonality_props(m, fcst, **comp)
            else:
                comp_name = f"season_{comp['comp_name']}"
                trace_object = get_forecast_component_props(fcst=fcst, comp_name=comp_name, plot_name=comp["plot_name"])

        elif "auto-regression" in name or "lagged regressor" in name or "residuals" in name:
            trace_object = get_multiforecast_component_props(fcst=fcst, **comp)
            fig.update_layout(barmode="overlay")

        if i == 0:
            xaxis = fig["layout"]["xaxis"]
            yaxis = fig["layout"]["yaxis"]
        else:
            xaxis = fig["layout"][f"xaxis{i + 1}"]
            yaxis = fig["layout"][f"yaxis{i + 1}"]

        xaxis.update(trace_object["xaxis"])
        xaxis.update(**xaxis_args)
        yaxis.update(trace_object["yaxis"])
        yaxis.update(**yaxis_args)
        for trace in trace_object["traces"]:
            fig.add_trace(trace, i + 1, 1)

    # Reset multiplicative axes labels after tight_layout adjustment
    for ax in multiplicative_axes:
        ax = set_y_as_percent(ax)

    return fig


def get_forecast_component_props(
    fcst, comp_name, plot_name=None, multiplicative=False, bar=False, rolling=None, add_x=False, **kwargs
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
                go.Bar(name=plot_name, x=fcst_t, y=rolling_avg, text=text, color=prediction_color, opacity=0.5)
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
                )
            )

            if add_x:
                traces.append(
                    go.Scatter(
                        x=fcst_t,
                        y=fcst[comp_name],
                        mode="markers",
                        marker=dict(color=cross_marker_color, size=marker_size, symbol=cross_symbol),
                    )
                )

    y = fcst[comp_name].values

    if "residual" in comp_name:
        y[-1] = 0

    if bar:
        traces.append(
            go.Bar(
                name=plot_name,
                x=fcst_t,
                y=y,
                text=text,
                marker_color=prediction_color,
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
            )
        )

        if add_x:
            traces.append(
                go.Scatter(
                    x=fcst_t,
                    y=fcst[comp_name],
                    mode="markers",
                    marker=dict(color=cross_marker_color, size=marker_size, symbol=cross_symbol),
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
            notnull = y.notnull()
            y = y.values
            alpha_min = 0.2
            alpha_softness = 1.2
            alpha = alpha_min + alpha_softness * (1.0 - alpha_min) / (i + 1.0 * alpha_softness)
            if "residual" not in comp_name:
                pass
            else:
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
                    )
                )

    if num_overplot is None or focus > 1:

        y = fcst[f"{comp_name}"]
        notnull = y.notnull()
        y = y.values
        if "residual" not in comp_name:
            fcst_t = fcst_t[notnull]
            y = y[notnull]
        else:
            y[-1] = 0
        if bar:
            traces.append(
                go.Bar(
                    name=plot_name,
                    x=fcst_t,
                    y=y,
                    text=text,
                    marker_color=prediction_color,
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


def get_seasonality_props(m, fcst, comp_name="weekly", multiplicative=False, quick=False, **kwargs):
    """
    Prepares a dictionary for plotting the selected seasonality with plotly

    Parameters
    ----------
        m : NeuralProphet
            Fitted NeuralProphet model
        fcst : pd.DataFrame
            Output of m.predict
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

    end = start + pd.Timedelta(days=period)
    if (fcst["ds"].dt.hour == 0).all():  # Day Precision
        plot_points = np.floor(period * 24).astype(int)
    elif (fcst["ds"].dt.minute == 0).all():  # Hour Precision
        plot_points = np.floor(period * 24 * 24).astype(int)
    else:  # Minute Precision
        plot_points = np.floor(period * 24 * 60).astype(int)
    days = pd.to_datetime(np.linspace(start.value, end.value, plot_points, endpoint=False))
    df_y = pd.DataFrame({"ds": days})

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

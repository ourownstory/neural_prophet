import numpy as np
import pandas as pd
import logging
from neuralprophet.utils import set_y_as_percent
import datetime

log = logging.getLogger("NP.plotly")

try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
except ImportError:
    log.error("Importing plotly failed. Interactive plots will not work.")


def get_parameter_components(m, forecast_in_focus):
    """Provides the components for plotting parameters.

    Args:
        m (NeuralProphet): fitted model.

    Returns:
        A list of dicts consisting the parameter plot components.
    """

    # Identify components to be plotted
    # as dict: {plot_name, }
    components = [{"plot_name": "Trend"}]
    if m.config_trend.n_changepoints > 0:
        components.append({"plot_name": "Trend Rate Change"})

    # Plot  seasonalities, if present
    if m.season_config is not None:
        for name in m.season_config.periods:
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
    if m.regressors_config is not None:
        for regressor, configs in m.regressors_config.items():
            mode = configs["mode"]
            regressor_param = m.model.get_reg_weights(regressor)
            if mode == "additive":
                additive_future_regressors.append((regressor, regressor_param.detach().numpy()))
            else:
                multiplicative_future_regressors.append((regressor, regressor_param.detach().numpy()))

    additive_events = []
    multiplicative_events = []
    # Add Events
    # add the country holidays
    if m.country_holidays_config is not None:
        for country_holiday in m.country_holidays_config["holiday_names"]:
            event_params = m.model.get_event_weights(country_holiday)
            weight_list = [(key, param.detach().numpy()) for key, param in event_params.items()]
            mode = m.country_holidays_config["mode"]
            if mode == "additive":
                additive_events = additive_events + weight_list
            else:
                multiplicative_events = multiplicative_events + weight_list

    # add the user specified events
    if m.events_config is not None:
        for event, configs in m.events_config.items():
            event_params = m.model.get_event_weights(event)
            weight_list = [(key, param.detach().numpy()) for key, param in event_params.items()]
            mode = configs["mode"]
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
                        "comp_name": 'Lagged Regressor "{}"'.format(name),
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
        additive_events = [(key, weight * m.data_params["y"].scale) for (key, weight) in additive_events]

        components.append({"plot_name": "Additive event"})
    if len(multiplicative_events) > 0:
        components.append({"plot_name": "Multiplicative event"})

    return components


def get_trend_change(m, plot_name="Trend Change"):
    """Make a barplot of the magnitudes of trend-changes.

    Args:
        m (NeuralProphet): fitted model.
        plot_name (str): Name of the plot Title.

    Returns:
        A dictionary with Plotly traces, xaxis and yaxis
    """
    zeroline_color = "#AAA"
    color = "#0072B2"

    start = m.data_params["ds"].shift
    scale = m.data_params["ds"].scale
    time_span_seconds = scale.total_seconds()
    cp_t = []
    for cp in m.model.config_trend.changepoints:
        cp_t.append(start + datetime.timedelta(seconds=cp * time_span_seconds))
    weights = m.model.get_trend_deltas.detach().numpy()
    # add end-point to force scale to match trend plot
    cp_t.append(start + scale)
    weights = np.append(weights, [0.0])
    width = time_span_seconds / 175000 / m.config_trend.n_changepoints

    text = None

    traces = []
    traces.append(
        go.Bar(
            name=plot_name,
            x=cp_t,
            y=weights,
            marker_color=color,
        )
    )
    xaxis = go.layout.XAxis(type="date")
    yaxis = go.layout.YAxis(
        rangemode="normal",
        title=go.layout.yaxis.Title(text=plot_name),
        zerolinecolor=zeroline_color,
    )

    return {"traces": traces, "xaxis": xaxis, "yaxis": yaxis}


def get_trend(m, plot_name="Trend Change"):
    """Make a barplot of the magnitudes of trend-changes.

    Args:
        m (NeuralProphet): fitted model.
        plot_name (str): Name of the plot Title.

    Returns:
        A dictionary with Plotly traces, xaxis and yaxis
    """

    traces = []
    color = "#0072B2"
    zeroline_color = "#AAA"
    line_width = 1

    t_start = m.data_params["ds"].shift
    t_end = t_start + m.data_params["ds"].scale

    if m.config_trend.n_changepoints == 0:
        fcst_t = pd.Series([t_start, t_end]).dt.to_pydatetime()
        trend_0 = m.model.bias.detach().numpy()
        if m.config_trend.growth == "off":
            trend_1 = trend_0
        else:
            trend_1 = trend_0 + m.model.trend_k0.detach().numpy()
        trend_0 = trend_0 * m.data_params["y"].scale + m.data_params["y"].shift
        trend_1 = trend_1 * m.data_params["y"].scale + m.data_params["y"].shift
        traces.append(
            go.Scatter(
                name=plot_name,
                x=fcst_t,
                y=trend_0,
                mode="lines",
                line=dict(color=color, width=line_width),
                fill="none",
            )
        )

        traces.append(
            go.Scatter(
                name=plot_name,
                x=fcst_t,
                y=trend_1,
                mode="lines",
                line=dict(color=color, width=line_width),
                fill="none",
            )
        )
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

    xaxis = go.layout.XAxis(type="date")
    yaxis = go.layout.YAxis(
        rangemode="normal",
        title=go.layout.yaxis.Title(text=plot_name),
        zerolinecolor=zeroline_color,
    )

    return {"traces": traces, "xaxis": xaxis, "yaxis": yaxis}


def plot_parameters_plotly(m, forecast_in_focus=None, weekly_start=0, yearly_start=0, figsize=(900, 200)):
    """Plot the parameters that the model is composed of, visually.

    Args:
        m (NeuralProphet): fitted model.
        forecast_in_focus (int): n-th step ahead forecast AR-coefficients to plot
        weekly_start (int):  specifying the start day of the weekly seasonality plot.
            0 (default) starts the week on Sunday.
            1 shifts by 1 day to Monday, and so on.
        yearly_start (int): specifying the start day of the yearly seasonality plot.
            0 (default) starts the year on Jan 1.
            1 shifts by 1 day to Jan 2, and so on.
        figsize (tuple): width, height in inches.
            None (default):  automatic (10, 3 * npanel)

    Returns:
        A plotly figure.
    """

    components = get_parameter_components(m, forecast_in_focus)

    npanel = len(components)
    figsize = figsize if figsize else (10, 3 * npanel)

    # Create Plotly subplot figure and add the components to it
    fig = make_subplots(npanel, cols=1, print_grid=False)
    fig["layout"].update(go.Layout(showlegend=False, width=figsize[0], height=figsize[1] * npanel))

    if npanel == 1:
        axes = [axes]
    multiplicative_axes = []

    for i, comp in enumerate(components):
        plot_name = comp["plot_name"].lower()
        if plot_name.startswith("trend"):
            if "change" in plot_name:
                # plot_trend_change(m=m, ax=ax, plot_name=comp["plot_name"])
                trace_object = get_trend_change(m, plot_name=comp["plot_name"])
            else:
                # plot_trend(m=m, ax=ax, plot_name=comp["plot_name"])
                trace_object = get_trend(m, plot_name=comp["plot_name"])

        else:
            continue

        if i == 0:
            xaxis = fig["layout"]["xaxis"]
            yaxis = fig["layout"]["yaxis"]
        else:
            xaxis = fig["layout"]["xaxis{}".format(i + 1)]
            yaxis = fig["layout"]["yaxis{}".format(i + 1)]

        xaxis.update(trace_object["xaxis"])
        yaxis.update(trace_object["yaxis"])
        for trace in trace_object["traces"]:
            fig.add_trace(trace, i + 1, 1)

    return fig

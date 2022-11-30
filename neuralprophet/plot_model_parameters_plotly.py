import datetime
import logging

import numpy as np
import pandas as pd

from neuralprophet.plot_utils import predict_one_season, predict_season_from_dates

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


def plot_trend_change(m, quantile, plot_name="Trend Change", df_name="__df__"):
    """Make a barplot of the magnitudes of trend-changes.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        quantile : float
            The quantile for which the yearly seasonality is plotted
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
    if isinstance(df_name, list):
        df_name = df_name[0]
    data_params = m.config_normalization.get_data_params(df_name)
    start = data_params["ds"].shift
    scale = data_params["ds"].scale
    time_span_seconds = scale.total_seconds()
    cp_t = []
    for cp in m.model.config_trend.changepoints:
        cp_t.append(start + datetime.timedelta(seconds=cp * time_span_seconds))
    # Global/Local Mode
    if m.model.config_trend.trend_global_local == "local":
        quantile_index = m.model.quantiles.index(quantile)
        weights = m.model.get_trend_deltas.detach()[quantile_index, m.model.id_dict[df_name], :].numpy()
    else:
        quantile_index = m.model.quantiles.index(quantile)
        weights = m.model.get_trend_deltas.detach()[quantile_index, 0, :].numpy()
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


def plot_trend(m, quantile, plot_name="Trend Change", df_name="__df__"):
    """Make a barplot of the magnitudes of trend-changes.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        quantile : float
            The quantile for which the yearly seasonality is plotted
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

    if m.config_trend.n_changepoints == 0:
        if isinstance(df_name, list):
            df_name = df_name[0]
        data_params = m.config_normalization.get_data_params(df_name)
        t_start = data_params["ds"].shift
        t_end = t_start + data_params["ds"].scale
        quantile_index = m.model.quantiles.index(quantile)

        fcst_t = pd.Series([t_start, t_end]).dt.to_pydatetime()
        trend_0 = m.model.bias[quantile_index].detach().numpy().squeeze().reshape(1)
        if m.config_trend.growth == "off":
            trend_1 = trend_0
        else:
            if m.model.config_trend.trend_global_local == "local":
                trend_1 = trend_0 + m.model.trend_k0[quantile_index, m.model.id_dict[df_name]].detach().numpy()
            else:
                trend_1 = trend_0 + m.model.trend_k0[quantile_index, 0].detach().numpy()

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
        mean_std = True
        if not isinstance(df_name, list):
            df_name = [df_name]
            # if global df with no specified df_name: plot mean and std, otherwise: don't
            mean_std = False
        df_y = pd.DataFrame()
        for df_name_i in df_name:
            data_params = m.config_normalization.get_data_params(df_name_i)
            t_start = data_params["ds"].shift
            t_end = t_start + data_params["ds"].scale
            quantile_index = m.model.quantiles.index(quantile)

            days = pd.date_range(start=t_start, end=t_end, freq=m.data_freq)
            df_i = pd.DataFrame({"ds": days})
            df_i["ID"] = df_name_i
            df_y = pd.concat((df_y, df_i), ignore_index=True)

        df_trend = m.predict_trend(df=df_y, quantile=quantile)
        if mean_std:
            df_trend_q90 = df_trend.groupby("ds")[["trend"]].apply(lambda x: x.quantile(0.9))
            df_trend_q10 = df_trend.groupby("ds")[["trend"]].apply(lambda x: x.quantile(0.1))
            df_trend = df_trend.groupby("ds")[["trend"]].apply(lambda x: x.mean())
            df_trend["ID"] = m.id_list[0]
            df_y = df_y[df_y["ID"] == m.id_list[0]]

        traces.append(
            go.Scatter(
                name=plot_name + " mean" if mean_std else plot_name,
                x=df_y["ds"].dt.to_pydatetime(),
                y=df_trend["trend"],
                mode="lines",
                line=dict(color=color, width=line_width),
                fill="none",
            )
        )
        if mean_std:
            # If more than on ID has been provided, and no df_name has been specified: plot mean and quants of the component
            filling = "tonexty"
            traces.append(
                go.Scatter(
                    name="Quants: 10%",
                    x=df_y["ds"].dt.to_pydatetime(),
                    y=df_trend_q10["trend"],
                    mode="lines",
                    line=dict(color="rgba(45, 146, 255, 0.2)", width=1),
                    fillcolor="rgba(45, 146, 255, 0.2)",
                    showlegend=True,
                )
            )
            traces.append(
                go.Scatter(
                    name="Quants: 90%",
                    x=df_y["ds"].dt.to_pydatetime(),
                    y=df_trend_q90["trend"],
                    fill=filling,
                    mode="lines",
                    line=dict(color="rgba(45, 146, 255, 0.2)", width=1),
                    fillcolor="rgba(45, 146, 255, 0.2)",
                    showlegend=True,
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


def plot_yearly(m, quantile, comp_name="yearly", yearly_start=0, quick=True, multiplicative=False, df_name="__df__"):
    """Plot the yearly component of the forecast.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        quantile : float
            The quantile for which the yearly seasonality is plotted
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
        df_name : str
            Name of dataframe to refer to data params from original keys of train dataframes

            Note
            ----
            Only used for local normalization in global modeling

    """
    traces = []
    line_width = 2

    # Compute yearly seasonality for a Jan 1 - Dec 31 sequence of dates.
    days = pd.date_range(start="2017-01-01", periods=365) + pd.Timedelta(days=yearly_start)
    df_y = pd.DataFrame({"ds": days})
    if not isinstance(df_name, list):
        df_y["ID"] = df_name
    mean_std = False  # Indicates whether mean and std of global df shall be plotted
    if isinstance(df_name, list):
        mean_std = True
        quick = False
        df_y = pd.DataFrame()
        for i in range(m.id_list.__len__()):
            df_i = pd.DataFrame({"ds": days})
            df_i["ID"] = m.id_list[i]
            df_y = pd.concat((df_y, df_i), ignore_index=True)
    if quick:
        predicted = predict_season_from_dates(m, dates=df_y["ds"], name=comp_name, quantile=quantile, df_name=df_name)
    else:
        predicted = m.predict_seasonal_components(df_y, quantile=quantile)[["ds", "ID", comp_name]]

    if mean_std:
        # If more than on ID has been provided, and no df_name has been specified: plot median and quants across all IDs
        predicted_q90 = predicted[["ds", comp_name]].groupby("ds").apply(lambda x: x.quantile(0.9))
        predicted_q10 = predicted[["ds", comp_name]].groupby("ds").apply(lambda x: x.quantile(0.1))
        predicted = predicted[["ds", comp_name]].groupby("ds").apply(lambda x: x.mean())
        predicted["ID"] = m.id_list[0]
        df_y = df_y[df_y["ID"] == m.id_list[0]]

    traces.append(
        go.Scatter(
            name=comp_name + " Mean" if mean_std else comp_name,
            x=df_y["ds"].dt.to_pydatetime(),
            y=predicted[comp_name],
            mode="lines",
            line=dict(color=color, width=line_width),
            fill="none",
        )
    )
    if mean_std:
        filling = "tonexty"
        traces.append(
            go.Scatter(
                name="Quant 10%",
                x=df_y["ds"],
                y=predicted_q10[comp_name],
                mode="lines",
                line=dict(color="rgba(45, 146, 255, 0.2)", width=1),
                fillcolor="rgba(45, 146, 255, 0.2)",
                showlegend=True,
            )
        )
        traces.append(
            go.Scatter(
                name="Quant 90%",
                x=df_y["ds"],
                y=predicted_q90[comp_name],
                fill=filling,
                mode="lines",
                line=dict(color="rgba(45, 146, 255, 0.2)", width=1),
                fillcolor="rgba(45, 146, 255, 0.2)",
                showlegend=False,
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


def plot_weekly(m, quantile, comp_name="weekly", weekly_start=0, quick=True, multiplicative=False, df_name="__df__"):
    """Plot the weekly component of the forecast.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        quantile : float
            The quantile for which the yearly seasonality is plotted
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

    week_days = 7
    if m.data_freq == "B":
        week_days = 5
        weekly_start = 1
    days_i = pd.date_range(start="2017-01-01", periods=week_days * 24, freq="H") + pd.Timedelta(days=weekly_start)
    df_w = pd.DataFrame({"ds": days_i})
    if not isinstance(df_name, list):
        df_w["ID"] = df_name
    mean_std = False  # Indicates whether mean and std of global df shall be plotted
    if isinstance(df_name, list):
        df_w = pd.DataFrame()
        quick = False
        mean_std = True
        for i in range(m.id_list.__len__()):
            df_i = pd.DataFrame({"ds": days_i})
            df_i["ID"] = m.id_list[i]
            df_w = pd.concat((df_w, df_i), ignore_index=True)
    if quick:
        predicted = predict_season_from_dates(m, dates=df_w["ds"], name=comp_name, quantile=quantile, df_name=df_name)
    else:
        predicted = m.predict_seasonal_components(df_w, quantile=quantile)[["ds", "ID", comp_name]]
    days = pd.date_range(start="2017-01-01", periods=week_days) + pd.Timedelta(days=weekly_start)

    if mean_std:
        # If more than on ID has been provided, and no df_name has been specified: plot median and quants across all IDs
        predicted_q90 = predicted.groupby("ds")[[comp_name]].apply(lambda x: x.quantile(0.9))
        predicted_q10 = predicted.groupby("ds")[[comp_name]].apply(lambda x: x.quantile(0.1))
        predicted = predicted.groupby("ds")[[comp_name]].apply(lambda x: x.mean())
        predicted["ID"] = m.id_list[0]
        df_w = df_w[df_w["ID"] == m.id_list[0]]

    days = pd.date_range(start="2017-01-01", periods=8) + pd.Timedelta(days=weekly_start)
    days = days.day_name()

    traces.append(
        go.Scatter(
            name=comp_name + " Mean" if mean_std else comp_name,
            x=list(range(len(days_i))),
            # x=df_w['ds'].dt.to_pydatetime(),
            y=predicted[comp_name],
            mode="lines",
            line=dict(color=color, width=line_width),
            fill="none",
        )
    )
    if mean_std:
        filling = "tonexty"
        traces.append(
            go.Scatter(
                name="Quant 10%",
                x=list(range(len(days_i))),
                y=predicted_q10[comp_name],
                mode="lines",
                line=dict(color="rgba(45, 146, 255, 0.2)", width=1),
                fillcolor="rgba(45, 146, 255, 0.2)",
                showlegend=True,
            )
        )
        traces.append(
            go.Scatter(
                name="Quant 90%",
                x=list(range(len(days_i))),
                y=predicted_q90[comp_name],
                fill=filling,
                mode="lines",
                line=dict(color="rgba(45, 146, 255, 0.2)", width=1),
                fillcolor="rgba(45, 146, 255, 0.2)",
                showlegend=False,
            )
        )
    padded_range = get_dynamic_axis_range(list(range(len(days_i))), type="numeric")
    xaxis = go.layout.XAxis(
        title="Day of week",
        tickmode="array",
        range=padded_range,
        tickvals=[x * 24 for x in range(len(days) + 1 - weekly_start)],
        ticktext=list(days) + [days[0]] if m.data_freq != "B" else list(days),
    )
    yaxis = go.layout.YAxis(
        rangemode="normal",
        title=go.layout.yaxis.Title(text=f"Seasonality: {comp_name}"),
    )

    if multiplicative:
        yaxis.update(tickformat=".1%", hoverformat=".4%")

    return {"traces": traces, "xaxis": xaxis, "yaxis": yaxis}


def plot_daily(m, quantile, comp_name="daily", quick=True, multiplicative=False, df_name="__df__"):
    """Plot the daily component of the forecast.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        quantile : float
            The quantile for which the yearly seasonality is plotted
        comp_name : str
            Name of seasonality component if previously changed from default ``daily``
        quick : bool
            Use quick low-level call of model
        ax : matplotlib axis
            Matplotlib Axes to plot on
        multiplicative: bool
            Flag whether to set y axis as percentage
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

    # Compute daily seasonality
    days_i = pd.date_range(start="2017-01-01", periods=24 * 12, freq="5min")
    df_d = pd.DataFrame({"ds": days_i})
    if not isinstance(df_name, list):
        df_d["ID"] = df_name
    mean_std = False  # Indicates whether mean and std of global df shall be plotted
    if isinstance(df_name, list):
        df_d = pd.DataFrame()
        quick = False
        mean_std = True
        for i in range(m.id_list.__len__()):
            df_i = pd.DataFrame({"ds": days_i})
            df_i["ID"] = m.id_list[i]
            df_d = pd.concat((df_d, df_i), ignore_index=True)
    if quick:
        predicted = predict_season_from_dates(m, dates=df_d["ds"], name=comp_name, quantile=quantile, df_name=df_name)
    else:
        predicted = m.predict_seasonal_components(df_d, quantile=quantile)[["ds", "ID", comp_name]]

    if mean_std:
        # If more than on ID has been provided, and no df_name has been specified: plot median and quants across all IDs
        predicted_q90 = predicted.groupby("ds")[[comp_name]].apply(lambda x: x.quantile(0.9))
        predicted_q10 = predicted.groupby("ds")[[comp_name]].apply(lambda x: x.quantile(0.1))
        predicted = predicted.groupby("ds")[[comp_name]].apply(lambda x: x.mean())
        predicted["ID"] = m.id_list[0]
        df_d = df_d[df_d["ID"] == m.id_list[0]]

    traces.append(
        go.Scatter(
            name=comp_name + " Mean" if mean_std else comp_name,
            x=np.array(range(len(days_i))),
            # x=df_d['ds'].dt.to_pydatetime(),
            y=predicted[comp_name],
            mode="lines",
            line=dict(color=color, width=line_width),
            fill="none",
        ),
    )
    if mean_std:
        filling = "tonexty"
        traces.append(
            go.Scatter(
                name="Quant 10%",
                x=np.array(range(len(days_i))),
                y=predicted_q10[comp_name],
                mode="lines",
                line=dict(color="rgba(45, 146, 255, 0.2)", width=1),
                fillcolor="rgba(45, 146, 255, 0.2)",
                showlegend=True,
            )
        )
        traces.append(
            go.Scatter(
                name="Quant 90%",
                x=np.array(range(len(days_i))),
                y=predicted_q90[comp_name],
                fill=filling,
                mode="lines",
                line=dict(color="rgba(45, 146, 255, 0.2)", width=1),
                fillcolor="rgba(45, 146, 255, 0.2)",
                showlegend=False,
            )
        )
    padded_range = get_dynamic_axis_range(list(range(len(days_i))), type="numeric")
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


def plot_custom_season(m, comp_name, quantile, multiplicative=False, df_name="__df__"):
    """Plot any seasonal component of the forecast.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        comp_name : str
            Name of seasonality component
        quantile : float
            The quantile for which the custom season is plotted
        multiplicative : bool
            Flag whether to set y axis as percentage
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

    t_i, predicted = predict_one_season(m, quantile=quantile, name=comp_name, n_steps=300, df_name=df_name)
    traces = []

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


def plot_parameters(
    m,
    plot_configuration,
    quantile=0.5,
    weekly_start=0,
    yearly_start=0,
    figsize=(700, 210),
    df_name=None,
    forecast_in_focus=None,
):
    """Plot the parameters that the model is composed of, visually.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        plot_configuration: dict
            dict of configured parameters to plot
        quantile : float
            The quantile for which the model parameters are to be plotted
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
        forecast_in_focus: int
            optinal, i-th step ahead forecast to plot

            Note
            ----
            None (default): plot self.highlight_forecast_step_n by default

    Returns:
        Plotly figure
    """
    compnents_to_plot = plot_configuration["components_list"]
    additive_future_regressors = plot_configuration["additive_future_regressors"]
    additive_events = plot_configuration["additive_events"]
    multiplicative_future_regressors = plot_configuration["multiplicative_future_regressors"]
    multiplicative_events = plot_configuration["multiplicative_events"]
    lagged_scalar_regressors = plot_configuration["lagged_scalar_regressors"]

    npanel = len(compnents_to_plot)
    figsize = figsize if figsize else (700, 210 * npanel)

    # Create Plotly subplot figure and add the components to it
    fig = make_subplots(npanel, cols=1, print_grid=False)
    fig.update_layout(go.Layout(showlegend=False, width=figsize[0], height=figsize[1] * npanel, **layout_args))

    for i, comp in enumerate(compnents_to_plot):
        is_multiplicative = False
        plot_name = comp["plot_name"].lower()
        if plot_name.startswith("trend"):
            if "change" in plot_name:
                trace_object = plot_trend_change(m, quantile=quantile, plot_name=comp["plot_name"], df_name=df_name)
            else:
                trace_object = plot_trend(m, quantile=quantile, plot_name=comp["plot_name"], df_name=df_name)

        elif plot_name.startswith("seasonality"):
            name = comp["comp_name"]
            if m.config_season.mode == "multiplicative":
                is_multiplicative = True
            if name.lower() == "weekly" or m.config_season.periods[name].period == 7:
                trace_object = plot_weekly(
                    m=m,
                    quantile=quantile,
                    weekly_start=weekly_start,
                    comp_name=name,
                    multiplicative=is_multiplicative,
                    df_name=df_name,
                )
            elif name.lower() == "yearly" or m.config_season.periods[name].period == 365.25:
                trace_object = plot_yearly(
                    m=m,
                    quantile=quantile,
                    yearly_start=yearly_start,
                    comp_name=name,
                    multiplicative=is_multiplicative,
                    df_name=df_name,
                )
            elif name.lower() == "daily" or m.config_season.periods[name].period == 1:
                trace_object = plot_daily(
                    m=m, quantile=quantile, comp_name=name, multiplicative=is_multiplicative, df_name=df_name
                )
            else:
                trace_object = plot_custom_season(
                    m=m, quantile=quantile, comp_name=name, multiplicative=is_multiplicative, df_name=df_name
                )
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

import datetime
import time
import numpy as np
import pandas as pd
import logging
import torch
from neuralprophet import time_dataset
from neuralprophet.utils import set_y_as_percent

log = logging.getLogger("nprophet.plotting")

try:
    from matplotlib import pyplot as plt
    from matplotlib.dates import (
        MonthLocator,
        num2date,
        AutoDateLocator,
        AutoDateFormatter,
    )
    from matplotlib.ticker import FuncFormatter

    from pandas.plotting import deregister_matplotlib_converters

    deregister_matplotlib_converters()
except ImportError:
    log.error("Importing matplotlib failed. Plotting will not work.")


def plot_parameters(m, forecast_in_focus=None, weekly_start=0, yearly_start=0, figsize=None):
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
        A matplotlib figure.
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

    npanel = len(components)
    figsize = figsize if figsize else (10, 3 * npanel)
    fig, axes = plt.subplots(npanel, 1, facecolor="w", figsize=figsize)
    if npanel == 1:
        axes = [axes]
    multiplicative_axes = []
    for ax, comp in zip(axes, components):
        plot_name = comp["plot_name"].lower()
        if plot_name.startswith("trend"):
            if "change" in plot_name:
                plot_trend_change(m=m, ax=ax, plot_name=comp["plot_name"])
            else:
                plot_trend(m=m, ax=ax, plot_name=comp["plot_name"])
        elif plot_name.startswith("seasonality"):
            name = comp["comp_name"]
            if m.season_config.mode == "multiplicative":
                multiplicative_axes.append(ax)
            if name.lower() == "weekly" or m.season_config.periods[name].period == 7:
                plot_weekly(m=m, ax=ax, weekly_start=weekly_start, comp_name=name)
            elif name.lower() == "yearly" or m.season_config.periods[name].period == 365.25:
                plot_yearly(m=m, ax=ax, yearly_start=yearly_start, comp_name=name)
            elif name.lower() == "daily" or m.season_config.periods[name].period == 1:
                plot_daily(m=m, ax=ax, comp_name=name)
            else:
                plot_custom_season(m=m, ax=ax, comp_name=name)
        elif plot_name == "lagged weights":
            plot_lagged_weights(weights=comp["weights"], comp_name=comp["comp_name"], focus=comp["focus"], ax=ax)
        else:
            if plot_name == "additive future regressor":
                weights = additive_future_regressors
            elif plot_name == "multiplicative future regressor":
                multiplicative_axes.append(ax)
                weights = multiplicative_future_regressors
            elif plot_name == "lagged scalar regressor":
                weights = lagged_scalar_regressors
            elif plot_name == "additive event":
                weights = additive_events
            elif plot_name == "multiplicative event":
                multiplicative_axes.append(ax)
                weights = multiplicative_events
            plot_scalar_weights(weights=weights, plot_name=comp["plot_name"], focus=forecast_in_focus, ax=ax)
    fig.tight_layout()
    # Reset multiplicative axes labels after tight_layout adjustment
    for ax in multiplicative_axes:
        ax = set_y_as_percent(ax)
    return fig


def plot_trend_change(m, ax=None, plot_name="Trend Change", figsize=(10, 6)):
    """Make a barplot of the magnitudes of trend-changes.

    Args:
        m (NeuralProphet): fitted model.
        ax (matplotlib axis): matplotlib Axes to plot on.
            One will be created if this is not provided.
        plot_name (str): Name of the plot Title.
        figsize (tuple): width, height in inches. Ignored if ax is not None.
             default: (10, 6)

    Returns:
        a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)

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
    artists += ax.bar(cp_t, weights, width=width, color="#0072B2")
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    ax.set_xlabel("Trend Segment")
    ax.set_ylabel(plot_name)
    return artists


def plot_trend(m, ax=None, plot_name="Trend", figsize=(10, 6)):
    """Make a barplot of the magnitudes of trend-changes.

    Args:
        m (NeuralProphet): fitted model.
        ax (matplotlib axis): matplotlib Axes to plot on.
            One will be created if this is not provided.
        plot_name (str): Name of the plot Title.
        figsize (tuple): width, height in inches. Ignored if ax is not None.
             default: (10, 6)

    Returns:
        a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
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
        artists += ax.plot(fcst_t, [trend_0, trend_1], ls="-", c="#0072B2")
    else:
        days = pd.date_range(start=t_start, end=t_end, freq=m.data_freq)
        df_y = pd.DataFrame({"ds": days})
        df_trend = m.predict_trend(df_y)
        artists += ax.plot(df_y["ds"].dt.to_pydatetime(), df_trend["trend"], ls="-", c="#0072B2")
    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    ax.set_xlabel("ds")
    ax.set_ylabel(plot_name)
    return artists


def plot_scalar_weights(weights, plot_name, focus=None, ax=None, figsize=(10, 6)):
    """Make a barplot of the regressor weights.

    Args:
        weights (list): tuples (name, weights)
        plot_name (string): name of the plot
        ax (matplotlib axis): matplotlib Axes to plot on.
            One will be created if this is not provided.
        focus (int): if provided, show weights for this forecast
            None (default) plot average
        figsize (tuple): width, height in inches. Ignored if ax is not None.
             default: (10, 6)
    Returns:
        a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
    # if len(regressors) == 1:
    # else:
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
    artists += ax.bar(names, values, width=0.8, color="#0072B2")
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    ax.set_xlabel(plot_name + " name")
    xticks = ax.get_xticklabels()
    if len("_".join(names)) > 100:
        for tick in xticks:
            tick.set_ha("right")
            tick.set_rotation(20)
    if "lagged" in plot_name.lower():
        if focus is None:
            ax.set_ylabel(plot_name + " weight (avg)")
        else:
            ax.set_ylabel(plot_name + " weight ({})-ahead".format(focus))
    else:
        ax.set_ylabel(plot_name + " weight")
    return artists


def plot_lagged_weights(weights, comp_name, focus=None, ax=None, figsize=(10, 6)):
    """Make a barplot of the importance of lagged inputs.

    Args:
        weights (np.array): model weights as matrix or vector
        comp_name (str): name of lagged inputs
        focus (int): if provided, show weights for this forecast
            None (default) sum over all forecasts and plot as relative percentage
        ax (matplotlib axis): matplotlib Axes to plot on.
            One will be created if this is not provided.
        figsize (tuple): width, height in inches. Ignored if ax is not None.
             default: (10, 6)
    Returns:
        a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
    n_lags = weights.shape[1]
    lags_range = list(range(1, 1 + n_lags))[::-1]
    if focus is None:
        weights = np.sum(np.abs(weights), axis=0)
        weights = weights / np.sum(weights)
        artists += ax.bar(lags_range, weights, width=1.00, color="#0072B2")
    else:
        if len(weights.shape) == 2:
            weights = weights[focus - 1, :]
        artists += ax.bar(lags_range, weights, width=0.80, color="#0072B2")
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    ax.set_xlabel("{} lag number".format(comp_name))
    if focus is None:
        ax.set_ylabel("{} relevance".format(comp_name))
        ax = set_y_as_percent(ax)
    else:
        ax.set_ylabel("{} weight ({})-ahead".format(comp_name, focus))
    return artists


def predict_one_season(m, name, n_steps=100):
    config = m.season_config.periods[name]
    t_i = np.arange(n_steps + 1) / float(n_steps)
    features = time_dataset.fourier_series_t(
        t=t_i * config.period, period=config.period, series_order=config.resolution
    )
    features = torch.from_numpy(np.expand_dims(features, 1))
    predicted = m.model.seasonality(features=features, name=name)
    predicted = predicted.squeeze().detach().numpy()
    if m.season_config.mode == "additive":
        predicted = predicted * m.data_params["y"].scale
    return t_i, predicted


def predict_season_from_dates(m, dates, name):
    config = m.season_config.periods[name]
    features = time_dataset.fourier_series(dates=dates, period=config.period, series_order=config.resolution)
    features = torch.from_numpy(np.expand_dims(features, 1))
    predicted = m.model.seasonality(features=features, name=name)
    predicted = predicted.squeeze().detach().numpy()
    if m.season_config.mode == "additive":
        predicted = predicted * m.data_params["y"].scale
    return predicted


def plot_custom_season(m, comp_name, ax=None, figsize=(10, 6)):
    """Plot any seasonal component of the forecast.

    Args:
        m (NeuralProphet): fitted model.
        comp_name (str): Name of seasonality component.
        ax (matplotlib axis): matplotlib Axes to plot on.
            One will be created if this is not provided.
        figsize (tuple): width, height in inches. Ignored if ax is not None.
             default: (10, 6)

    Returns:
        a list of matplotlib artists
    """
    t_i, predicted = predict_one_season(m, name=comp_name, n_steps=300)
    artists = []
    if not ax:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
    artists += ax.plot(t_i, predicted, ls="-", c="#0072B2")
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    ax.set_xlabel("One period: {}".format(comp_name))
    ax.set_ylabel("Seasonality: {}".format(comp_name))
    return artists


def plot_yearly(m, comp_name="yearly", yearly_start=0, quick=True, ax=None, figsize=(10, 6)):
    """Plot the yearly component of the forecast.

    Args:
        m (NeuralProphet): fitted model.
        ax (matplotlib axis): matplotlib Axes to plot on.
            One will be created if this is not provided.
        yearly_start (int): specifying the start day of the yearly seasonality plot.
            0 (default) starts the year on Jan 1.
            1 shifts by 1 day to Jan 2, and so on.
        quick (bool): use quick low-evel call of model. might break in future.
        figsize (tuple): width, height in inches. Ignored if ax is not None.
             default: (10, 6)
        comp_name (str): Name of seasonality component if previously changed from default 'yearly'.

    Returns:
        a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
    # Compute yearly seasonality for a Jan 1 - Dec 31 sequence of dates.
    days = pd.date_range(start="2017-01-01", periods=365) + pd.Timedelta(days=yearly_start)
    df_y = pd.DataFrame({"ds": days})
    if quick:
        predicted = predict_season_from_dates(m, dates=df_y["ds"], name=comp_name)
    else:
        predicted = m.predict_seasonal_components(df_y)[comp_name]
    artists += ax.plot(df_y["ds"].dt.to_pydatetime(), predicted, ls="-", c="#0072B2")
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    months = MonthLocator(range(1, 13), bymonthday=1, interval=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos=None: "{dt:%B} {dt.day}".format(dt=num2date(x))))
    ax.xaxis.set_major_locator(months)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Seasonality: {}".format(comp_name))
    return artists


def plot_weekly(m, comp_name="weekly", weekly_start=0, quick=True, ax=None, figsize=(10, 6)):
    """Plot the yearly component of the forecast.

    Args:
        m (NeuralProphet): fitted model.
        ax (matplotlib axis): matplotlib Axes to plot on.
            One will be created if this is not provided.
        weekly_start (int): specifying the start day of the weekly seasonality plot.
            0 (default) starts the week on Sunday.
            1 shifts by 1 day to Monday, and so on.
        quick (bool): use quick low-evel call of model. might break in future.
        figsize (tuple): width, height in inches. Ignored if ax is not None.
             default: (10, 6)
        comp_name (str): Name of seasonality component if previously changed from default 'weekly'.

    Returns:
        a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
    # Compute weekly seasonality for a Sun-Sat sequence of dates.
    days_i = pd.date_range(start="2017-01-01", periods=7 * 24, freq="H") + pd.Timedelta(days=weekly_start)
    df_w = pd.DataFrame({"ds": days_i})
    if quick:
        predicted = predict_season_from_dates(m, dates=df_w["ds"], name=comp_name)
    else:
        predicted = m.predict_seasonal_components(df_w)[comp_name]
    days = pd.date_range(start="2017-01-01", periods=7) + pd.Timedelta(days=weekly_start)
    days = days.day_name()
    artists += ax.plot(range(len(days_i)), predicted, ls="-", c="#0072B2")
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    ax.set_xticks(24 * np.arange(len(days) + 1))
    ax.set_xticklabels(list(days) + [days[0]])
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Seasonality: {}".format(comp_name))
    return artists


def plot_daily(m, comp_name="daily", quick=True, ax=None, figsize=(10, 6)):
    """Plot the daily component of the forecast.

    Args:
        m (NeuralProphet): fitted model.
        ax (matplotlib axis): matplotlib Axes to plot on.
            One will be created if this is not provided.
        quick (bool): use quick low-evel call of model. might break in future.
        figsize (tuple): width, height in inches. Ignored if ax is not None.
             default: (10, 6)
        comp_name (str): Name of seasonality component if previously changed from default 'daily'.

    Returns:
        a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
    # Compute daily seasonality
    dates = pd.date_range(start="2017-01-01", periods=24 * 12, freq="5min")
    df = pd.DataFrame({"ds": dates})
    if quick:
        predicted = predict_season_from_dates(m, dates=df["ds"], name=comp_name)
    else:
        predicted = m.predict_seasonal_components(df)[comp_name]
    artists += ax.plot(range(len(dates)), predicted, ls="-", c="#0072B2")
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    ax.set_xticks(12 * np.arange(25))
    ax.set_xticklabels(np.arange(25))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Seasonality: {}".format(comp_name))
    return artists

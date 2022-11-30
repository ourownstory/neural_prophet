import datetime
import logging

# from tkinter.messagebox import NO
import numpy as np
import pandas as pd

from neuralprophet.plot_utils import predict_one_season, predict_season_from_dates, set_y_as_percent

log = logging.getLogger("NP.plotting")

try:
    from matplotlib import pyplot as plt
    from matplotlib.dates import AutoDateFormatter, AutoDateLocator, MonthLocator, num2date
    from matplotlib.ticker import FuncFormatter
    from pandas.plotting import deregister_matplotlib_converters

    deregister_matplotlib_converters()
except ImportError:
    log.error("Importing matplotlib failed. Plotting will not work.")


def plot_parameters(
    m,
    plot_configuration,
    quantile=0.5,
    weekly_start=0,
    yearly_start=0,
    figsize=None,
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
            Default value is set to ``None`` ->  automatic ``figsize = (10, 3 * npanel)``
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

    Returns
    -------
        matplotlib.pyplot.figure
            Figure showing the NeuralProphet parameters

    Examples
    --------
    Base usage of :meth:`plot_parameters`

    >>> from neuralprophet import NeuralProphet
    >>> m = NeuralProphet()
    >>> metrics = m.fit(df, freq="D")
    >>> future = m.make_future_dataframe(df=df, periods=365)
    >>> forecast = m.predict(df=future)
    >>> fig_param = m.plot_parameters()

    """
    components_to_plot = plot_configuration["components_list"]
    additive_future_regressors = plot_configuration["additive_future_regressors"]
    additive_events = plot_configuration["additive_events"]
    multiplicative_future_regressors = plot_configuration["multiplicative_future_regressors"]
    multiplicative_events = plot_configuration["multiplicative_events"]
    lagged_scalar_regressors = plot_configuration["lagged_scalar_regressors"]
    overwriting_unknown_data_normalization = plot_configuration["overwriting_unknown_data_normalization"]

    npanel = len(components_to_plot)
    figsize = figsize if figsize else (10, 3 * npanel)
    fig, axes = plt.subplots(npanel, 1, facecolor="w", figsize=figsize)
    if npanel == 1:
        axes = [axes]
    multiplicative_axes = []
    for ax, comp in zip(axes, components_to_plot):
        plot_name = comp["plot_name"].lower()
        if plot_name.startswith("trend"):
            if "change" in plot_name:
                plot_trend_change(m=m, quantile=quantile, ax=ax, plot_name=comp["plot_name"], df_name=df_name)
            else:
                plot_trend(m=m, quantile=quantile, ax=ax, plot_name=comp["plot_name"], df_name=df_name)
        elif plot_name.startswith("seasonality"):
            name = comp["comp_name"]
            if m.config_season.mode == "multiplicative":
                multiplicative_axes.append(ax)
            if name.lower() == "weekly" or m.config_season.periods[name].period == 7:
                plot_weekly(m=m, quantile=quantile, ax=ax, weekly_start=weekly_start, comp_name=name, df_name=df_name)
            elif name.lower() == "yearly" or m.config_season.periods[name].period == 365.25:
                plot_yearly(m=m, quantile=quantile, ax=ax, yearly_start=yearly_start, comp_name=name, df_name=df_name)
            elif name.lower() == "daily" or m.config_season.periods[name].period == 1:
                plot_daily(m=m, quantile=quantile, ax=ax, comp_name=name, df_name=df_name)
            else:
                plot_custom_season(m=m, quantile=quantile, ax=ax, comp_name=name, df_name=df_name)
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
    if overwriting_unknown_data_normalization:
        # if overwriting_unknown_data_normalization is True, we get back to the initial False state
        m.config_normalization.unknown_data_normalization = False

    return fig


def plot_trend_change(m, quantile, ax=None, plot_name="Trend Change", figsize=(10, 6), df_name="__df__"):
    """Make a barplot of the magnitudes of trend-changes.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        quantile : float
            The quantile for which the trend changes are plotted
        ax : matplotlib axis
            Matplotlib Axes to plot on
        plot_name : str
            Name of the plot Title
        figsize : tuple
            Width, height in inches, ignored if ax is not None.

            Note
            ----
            Default value is set to ``figsize = (10, 6)``

        df_name : str
            Name of dataframe to refer to data params from original keys of train dataframes

            Note
            ----
            Only used for local normalization in global modeling

    Returns
    -------
        matplotlib.artist.Artist
            List of Artist objects containing barplot
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
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


def plot_trend(m, quantile, ax=None, plot_name="Trend", figsize=(10, 6), df_name="__df__"):
    """Make a barplot of the magnitudes of trend-changes.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        quantile : float
            The quantile for which the trend changes are plotted
        ax : matplotlib axis
            Matplotlib Axes to plot on
        plot_name : str
            Name of the plot Title
        figsize : tuple
            Width, height in inches, ignored if ax is not None.

            Note
            ----
            Default value is set to ``figsize = (10, 6)``

        df_name : str
            Name of dataframe to refer to data params from original keys of train dataframes

            Note
            ----
            Only used for local normalization in global modeling

    Returns
    -------
        matplotlib.artist.Artist
            List of Artist objects containing barplot
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
    if m.config_trend.n_changepoints == 0:
        if isinstance(df_name, list):
            df_name = df_name[0]
        data_params = m.config_normalization.get_data_params(df_name)
        t_start = data_params["ds"].shift
        t_end = t_start + data_params["ds"].scale
        quantile_index = m.model.quantiles.index(quantile)

        fcst_t = pd.Series([t_start, t_end]).dt.to_pydatetime()
        trend_0 = m.model.bias[quantile_index].detach().numpy().squeeze()
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
        artists += ax.plot(fcst_t, [trend_0, trend_1], ls="-", c="#0072B2")
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

        artists += ax.plot(df_y["ds"], df_trend["trend"], ls="-", c="#0072B2", label="Mean" if mean_std else None)
        if mean_std:
            ax.fill_between(
                df_y["ds"].dt.to_pydatetime(),
                df_trend_q10["trend"],
                df_trend_q90["trend"],
                alpha=0.2,
                color="#0072B2",
                label="Quants 10-90%",
            )
            ax.legend()
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

    Parameters
    ----------
        weights : list
            tuples of (name, weights)
        plot_name : str
            Name of the plot Title
        focus : int
            Show weights for this forecast, if provided
        ax : matplotlib axis
            Matplotlib Axes to plot on
        figsize : tuple
            Width, height in inches, ignored if ax is not None.

            Note
            ----
            Default value is set to ``figsize = (10, 6)``

    Returns
    -------
        matplotlib.artist.Artist
            List of Artist objects containing barplot
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
            ax.set_ylabel(plot_name + f" weight ({focus})-ahead")
    else:
        ax.set_ylabel(plot_name + " weight")
    return artists


def plot_lagged_weights(weights, comp_name, focus=None, ax=None, figsize=(10, 6)):
    """Make a barplot of the importance of lagged inputs.

    Parameters
    ----------
        weights : list
            tuples of (name, weights)
        comp_name : str
            Name of lagged inputs
        focus : int
            Show weights for this forecast, if provided
        ax : matplotlib axis
            Matplotlib Axes to plot on
        figsize : tuple
            Width, height in inches, ignored if ax is not None.

            Note
            ----
            Default value is set to ``figsize = (10, 6)``

    Returns
    -------
        matplotlib.artist.Artist
            List of Artist objects containing barplot
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
    ax.set_xlabel(f"{comp_name} lag number")
    if focus is None:
        ax.set_ylabel(f"{comp_name} relevance")
        ax = set_y_as_percent(ax)
    else:
        ax.set_ylabel(f"{comp_name} weight ({focus})-ahead")
    return artists


def plot_custom_season(m, comp_name, quantile, ax=None, figsize=(10, 6), df_name="__df__"):
    """Plot any seasonal component of the forecast.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        comp_name : str
            Name of seasonality component
        quantile : float
            The quantile for which the custom season is plotted
        ax : matplotlib axis
            Matplotlib Axes to plot on
        focus : int
            Show weights for this forecast, if provided
        figsize : tuple
            Width, height in inches, ignored if ax is not None.

            Note
            ----
            Default value is set to ``figsize = (10, 6)``
        df_name : str
            Name of dataframe to refer to data params from original keys of train dataframes

            Note
            ----
            Only used for local normalization in global modeling

    Returns
    -------
        matplotlib.artist.Artist
            List of Artist objects containing seasonal forecast component

    """
    t_i, predicted = predict_one_season(m, name=comp_name, n_steps=300, quantile=quantile, df_name=df_name)
    artists = []
    if not ax:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
    artists += ax.plot(t_i, predicted, ls="-", c="#0072B2")
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    ax.set_xlabel(f"One period: {comp_name}")
    ax.set_ylabel(f"Seasonality: {comp_name}")
    return artists


def plot_yearly(
    m, quantile, comp_name="yearly", yearly_start=0, quick=True, ax=None, figsize=(10, 6), df_name="__df__"
):
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
        ax : matplotlib axis
            Matplotlib Axes to plot on
        figsize : tuple
            Width, height in inches, ignored if ax is not None.

            Note
            ----
            Default value is set to ``figsize = (10, 6)``
        df_name : str
            Name of dataframe to refer to data params from original keys of train dataframes

            Note
            ----
            Only used for local normalization in global modeling

    Returns
    -------
        matplotlib.artist.Artist
            List of Artist objects containing yearly forecast component

    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
    # Compute yearly seasonality for a Jan 1 - Dec 31 sequence of dates.
    days = pd.date_range(start="2017-01-01", periods=365) + pd.Timedelta(days=yearly_start)
    df_y = pd.DataFrame({"ds": days})
    if not isinstance(df_name, list):
        df_y["ID"] = df_name
    mean_std = False  # Indicates whether mean and std of global df shall be plotted
    if isinstance(df_name, list):
        df_y = pd.DataFrame()
        mean_std = True
        quick = False
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
        predicted_q90 = predicted.groupby("ds")[[comp_name]].apply(lambda x: x.quantile(0.9))
        predicted_q10 = predicted.groupby("ds")[[comp_name]].apply(lambda x: x.quantile(0.1))
        predicted = predicted.groupby("ds")[[comp_name]].apply(lambda x: x.mean())
        predicted["ID"] = m.id_list[0]
        df_y = df_y[df_y["ID"] == m.id_list[0]]

    artists += ax.plot(
        df_y["ds"].dt.to_pydatetime(),
        predicted[comp_name],
        ls="-",
        c="#0072B2",
        label=comp_name + " Mean" if mean_std else None,
    )
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    if mean_std:
        ax.fill_between(
            df_y["ds"],
            predicted_q10[comp_name],
            predicted_q90[comp_name],
            alpha=0.2,
            color="#0072B2",
            label="Quants 10-90%",
        )
        ax.legend()
    months = MonthLocator(range(1, 13), bymonthday=1, interval=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos=None: f"{num2date(x):%B} {num2date(x).day}"))
    ax.xaxis.set_major_locator(months)
    ax.set_xlabel("Day of year")
    ax.set_ylabel(f"Seasonality: {comp_name}")
    return artists


def plot_weekly(
    m, quantile, comp_name="weekly", weekly_start=0, quick=True, ax=None, figsize=(10, 6), df_name="__df__"
):
    """Plot the weekly component of the forecast.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        quantile : float
            The quantile for which the weekly seasonality is plotted
        comp_name : str
            Name of seasonality component
        weekly_start : int
            Specifying the start day of the weekly seasonality plot

            Options
                * (default) ``weekly_start = 0``: starts the week on Sunday
                * ``weekly_start = 1``: shifts by 1 day to Monday, and so on
        quick : bool
            Use quick low-level call of model
        ax : matplotlib axis
            Matplotlib Axes to plot on
        figsize : tuple
            Width, height in inches, ignored if ax is not None.

            Note
            ----
            Default value is set to ``figsize = (10, 6)``
        df_name : str
            Name of dataframe to refer to data params from original keys of train dataframes

            Note
            ----
            Only used for local normalization in global modeling

    Returns
    -------
        matplotlib.artist.Artist
            List of Artist objects containing weekly forecast component

    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
    week_days = 7
    if m.data_freq == "B":
        week_days = 5
        weekly_start = 1
    days_i = pd.date_range(start="2017-01-01", periods=week_days * 24, freq="H") + pd.Timedelta(days=weekly_start)
    df_w = pd.DataFrame({"ds": days_i})
    if not isinstance(df_name, list):
        df_w["ID"] = df_name
    mean_std = False  # Indicates whether mean and quant of global df shall be plotted
    if isinstance(df_name, list):
        df_w = pd.DataFrame()
        mean_std = True
        quick = False
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

    days = pd.date_range(start="2017-01-01", periods=7) + pd.Timedelta(days=weekly_start)
    days = days.day_name()
    artists += ax.plot(
        range(len(days_i)), predicted[comp_name], ls="-", c="#0072B2", label=comp_name + " Mean" if mean_std else None
    )
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    if mean_std:
        ax.fill_between(
            range(len(days_i)),
            predicted_q10[comp_name],
            predicted_q90[comp_name],
            alpha=0.2,
            label="Quants 10-90%",
            color="#0072B2",
        )
        ax.legend()
    ax.set_xticks(24 * np.arange(len(days) + 1 - weekly_start))
    ax.set_xticklabels(list(days) + [days[0]] if m.data_freq != "B" else list(days))
    ax.set_xlabel("Day of week")
    ax.set_ylabel(f"Seasonality: {comp_name}")
    return artists


def plot_daily(m, quantile, comp_name="daily", quick=True, ax=None, figsize=(10, 6), df_name="__df__"):
    """Plot the daily component of the forecast.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        quantile : float
            The quantile for which the daily seasonality is plotted
        comp_name : str
            Name of seasonality component if previously changed from default ``daily``
        quick : bool
            Use quick low-level call of model
        ax : matplotlib axis
            Matplotlib Axes to plot on
        figsize : tuple
            Width, height in inches, ignored if ax is not None.

            Note
            ----
            Default value is set to ``figsize = (10, 6)``
        df_name : str
            Name of dataframe to refer to data params from original keys of train dataframes

            Note
            ----
            Only used for local normalization in global modeling

    Returns
    -------
        matplotlib.artist.Artist
            List of Artist objects containing weekly forecast component
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
    # Compute daily seasonality
    days = pd.date_range(start="2017-01-01", periods=24 * 12, freq="5min")
    df_d = pd.DataFrame({"ds": days})
    if not isinstance(df_name, list):
        df_d["ID"] = df_name
    mean_std = False  # Indicates whether mean and std of global df shall be plotted
    if isinstance(df_name, list):
        df_d = pd.DataFrame()
        mean_std = True
        quick = False
        for i in range(m.id_list.__len__()):
            df_i = pd.DataFrame({"ds": days})
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

    artists += ax.plot(
        range(len(days)), predicted[comp_name], ls="-", c="#0072B2", label=comp_name + " Mean" if mean_std else None
    )
    if mean_std:
        ax.fill_between(
            range(len(days)),
            predicted_q10[comp_name],
            predicted_q90[comp_name],
            alpha=0.2,
            label="Quants 10-90%",
            color="#0072B2",
        )
        ax.legend()
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    ax.set_xticks(12 * np.arange(25))
    ax.set_xticklabels(np.arange(25))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(f"Seasonality: {comp_name}")
    return artists

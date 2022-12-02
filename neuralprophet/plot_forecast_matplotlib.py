import logging

import numpy as np

from neuralprophet.plot_model_parameters_matplotlib import plot_custom_season, plot_daily, plot_weekly, plot_yearly
from neuralprophet.plot_utils import set_y_as_percent

log = logging.getLogger("NP.plotting")

try:
    from matplotlib import pyplot as plt
    from matplotlib.dates import AutoDateFormatter, AutoDateLocator
    from pandas.plotting import deregister_matplotlib_converters

    deregister_matplotlib_converters()
except ImportError:
    log.error("Importing matplotlib failed. Plotting will not work.")


def plot(
    fcst,
    quantiles,
    ax=None,
    xlabel="ds",
    ylabel="y",
    highlight_forecast=None,
    line_per_origin=False,
    figsize=(10, 6),
):
    """Plot the NeuralProphet forecast

    Parameters
    ---------
        fcst : pd.DataFrame
            Output of m.predict
        quantiles: list
            Quantiles for which the forecasts are to be plotted
        ax : matplotlib axes
            Axes to plot on
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
        matplotlib.pyplot.figure
            Figure showing the NeuralProphet forecast

            Examples
            --------
            Base usage

            >>> from neuralprophet import NeuralProphet
            >>> m = NeuralProphet()
            >>> metrics = m.fit(df, freq="D")
            >>> future = m.make_future_dataframe(df=df, periods=365)
            >>> forecast = m.predict(df=future)
            >>> fig_forecast = m.plot(forecast)

            Additional plot specifications

            >>> fig_forecast = m.plot(forecast,
            >>>                       xlabel="ds",
            >>>                       ylabel="y",
            >>>                       highlight_forecast=None,
            >>>                       line_per_origin=False,
            >>>                       figsize=(10, 6)
            >>>                       )

    """
    fcst = fcst.fillna(value=np.nan)
    if ax is None:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
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

    if highlight_forecast is None or line_per_origin:
        for i, name in enumerate(yhat_col_names_no_qts):
            ax.plot(
                ds,
                fcst[f"{colname}{i if line_per_origin else i + 1}"],
                ls="-",
                c="#0072B2",
                alpha=0.2 + 2.0 / (i + 2.5),
                label=name,
            )

    if len(quantiles) > 1:
        for i in range(1, len(quantiles)):
            ax.fill_between(
                ds,
                fcst[f"{colname}{step}"],
                fcst[f"{colname}{step} {round(quantiles[i] * 100, 1)}%"],
                color="#0072B2",
                alpha=0.2,
            )

    if highlight_forecast is not None:
        if line_per_origin:
            num_forecast_steps = sum(fcst["origin-0"].notna())
            steps_from_last = num_forecast_steps - highlight_forecast
            for i in range(len(yhat_col_names_no_qts)):
                x = ds[-(1 + i + steps_from_last)]
                y = fcst[f"origin-{i}"].values[-(1 + i + steps_from_last)]
                ax.plot(x, y, "bx")
        else:
            ax.plot(ds, fcst[f"yhat{highlight_forecast}"], ls="-", c="b", label=f"yhat{highlight_forecast}")
            ax.plot(ds, fcst[f"yhat{highlight_forecast}"], "bx", label=f"yhat{highlight_forecast}")

            if len(quantiles) > 1:
                for i in range(1, len(quantiles)):
                    ax.fill_between(
                        ds,
                        fcst[f"yhat{highlight_forecast}"],
                        fcst[f"yhat{highlight_forecast} {round(quantiles[i] * 100, 1)}%"],
                        color="#0072B2",
                        alpha=0.2,
                    )

    # Plot any conformal prediction intervals
    if any("+ qhat" in col for col in yhat_col_names) and any("- qhat" in col for col in yhat_col_names):
        quantile_hi = str(max(quantiles) * 100)
        quantile_lo = str(min(quantiles) * 100)
        if f"yhat1 {quantile_hi}% + qhat1" in fcst.columns and f"yhat1 {quantile_hi}% - qhat1" in fcst.columns:
            ax.plot(ds, fcst[f"yhat1 {quantile_hi}% + qhat1"], c="r", label=f"yhat1 {quantile_hi}% + qhat1")
            ax.plot(ds, fcst[f"yhat1 {quantile_lo}% - qhat1"], c="r", label=f"yhat1 {quantile_lo}% - qhat1")
        else:
            ax.plot(ds, fcst["yhat1 + qhat1"], c="r", label="yhat1 + qhat1")
            ax.plot(ds, fcst["yhat1 - qhat1"], c="r", label="yhat1 - qhat1")

    ax.plot(ds, fcst["y"], "k.", label="actual y")

    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    handles, labels = ax.axes.get_legend_handles_labels()
    if len(labels) > 10:
        ax.legend(handles[:10] + [handles[-1]], labels[:10] + [labels[-1]])
        log.warning("Legend is available only for the ten first handles")
    else:
        ax.legend(handles, labels)
    fig.tight_layout()
    return fig


def plot_components(
    m,
    fcst,
    plot_configuration,
    df_name="__df__",
    quantile=0.5,
    one_period_per_season=False,
    figsize=None,
):
    """Plot the NeuralProphet forecast components.

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
        quantile : float
            Quantile for which the forecast components are to be plotted
        one_period_per_season : bool
            Plot one period per season, instead of the true seasonal components of the forecast.
        figsize : tuple
            Width, height in inches.

            Note
            ----
            Default value is set to ``None`` ->  automatic ``figsize = (10, 3 * npanel)``

    Returns
    -------
        matplotlib.pyplot.figure
            Figure showing the NeuralProphet forecast components
    """
    log.debug("Plotting forecast components")
    fcst = fcst.fillna(value=np.nan)
    components_to_plot = plot_configuration["components_list"]

    # set number of axes based on selected plot_names and sort them according to order in components
    panel_names = list(set(next(iter(dic.values())).lower() for dic in components_to_plot))
    panel_order = [x for dic in components_to_plot for x in panel_names if x in dic["plot_name"].lower()]
    npanel = len(panel_names)
    figsize = figsize if figsize else (10, 3 * npanel)
    fig, axes = plt.subplots(npanel, 1, facecolor="w", figsize=figsize)
    if npanel == 1:
        axes = [axes]
    multiplicative_axes = []
    ax = 0
    # for ax, comp in zip(axes, components):
    for comp in components_to_plot:
        name = comp["plot_name"].lower()
        ax = axes[panel_order.index(name)]
        if (
            name in ["trend"]
            or ("ar" in name and "ahead" in name)
            or ("lagged regressor" in name and "ahead" in name)
            or ("uncertainty" in name)
        ):
            plot_forecast_component(fcst=fcst, ax=ax, **comp)
        elif "event" in name or "future regressor" in name:
            if "multiplicative" in comp.keys() and comp["multiplicative"]:
                multiplicative_axes.append(ax)
            plot_forecast_component(fcst=fcst, ax=ax, **comp)
        elif "season" in name:
            if m.config_season.mode == "multiplicative":
                multiplicative_axes.append(ax)
            if one_period_per_season:
                comp_name = comp["comp_name"]
                if comp_name.lower() == "weekly" or m.config_season.periods[comp_name].period == 7:
                    plot_weekly(m=m, ax=ax, quantile=quantile, comp_name=comp_name, df_name=df_name)
                elif comp_name.lower() == "yearly" or m.config_season.periods[comp_name].period == 365.25:
                    plot_yearly(m=m, ax=ax, quantile=quantile, comp_name=comp_name, df_name=df_name)
                elif comp_name.lower() == "daily" or m.config_season.periods[comp_name].period == 1:
                    plot_daily(m=m, ax=ax, quantile=quantile, comp_name=comp_name, df_name=df_name)
                else:
                    plot_custom_season(m=m, ax=ax, quantile=quantile, comp_name=comp_name, df_name=df_name)
            else:
                comp_name = f"season_{comp['comp_name']}"
                plot_forecast_component(fcst=fcst, ax=ax, comp_name=comp_name, plot_name=comp["plot_name"])
        elif "auto-regression" in name or "lagged regressor" in name:
            plot_multiforecast_component(fcst=fcst, ax=ax, **comp)

    fig.tight_layout()
    # Reset multiplicative axes labels after tight_layout adjustment
    for ax in multiplicative_axes:
        ax = set_y_as_percent(ax)
    return fig


def plot_forecast_component(
    fcst,
    comp_name,
    plot_name=None,
    ax=None,
    figsize=(10, 6),
    multiplicative=False,
    bar=False,
    rolling=None,
    add_x=False,
    fill=False,
):
    """Plot a particular component of the forecast.

    Parameters
    ----------
        fcst : pd.DataFrame
            Output of m.predict
        comp_name : str
            Name of the component to plot
        plot_name : str
            Name of the plot Title
        ax : matplotlib axis
            Matplotlib Axes to plot on
        figsize : tuple
            Width, height in inches. Ignored if ax is not None

            Note
            ----
            Default value is set to ``figsize = (10, 6)``
        multiplicative : bool
            Set y axis as percentage
        bar : bool
            Make barplot
        rolling : int
            Rolling average underplot
        add_x : bool
            Add x symbols to plotted points
        fill: bool
            Add fill between signal and x(y=0) axis

    Returns
    -------
        matplotlib.artist.Artist
            List of Artist objects containing a particular forecast component
    """
    fcst = fcst.fillna(value=np.nan)
    artists = []
    if not ax:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
    fcst_t = fcst["ds"].dt.to_pydatetime()
    if rolling is not None:
        rolling_avg = fcst[comp_name].rolling(rolling, min_periods=1, center=True).mean()
        if bar:
            artists += ax.bar(fcst_t, rolling_avg, width=1.00, color="#0072B2", alpha=0.5)
        else:
            artists += ax.plot(fcst_t, rolling_avg, ls="-", color="#0072B2", alpha=0.5)
            if add_x:
                artists += ax.plot(fcst_t, fcst[comp_name], "bx")
    if "uncertainty" in plot_name.lower():
        y = fcst[comp_name].values - fcst["yhat1"].values
        label = comp_name
    else:
        y = fcst[comp_name].values
        label = None
    if bar:
        artists += ax.bar(fcst_t, y, width=1.00, color="#0072B2")
    elif "uncertainty" in plot_name.lower() and fill:
        ax.fill_between(fcst_t, 0, y, alpha=0.2, label=label, color="#0072B2")
    else:
        artists += ax.plot(fcst_t, y, ls="-", c="#0072B2")
        if add_x or sum(fcst[comp_name].notna()) == 1:
            artists += ax.plot(fcst_t, y, "bx")
    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    ax.set_xlabel("ds")
    if plot_name is None:
        plot_name = comp_name
    ax.set_ylabel(plot_name)
    if multiplicative:
        ax = set_y_as_percent(ax)
    handles, labels = ax.axes.get_legend_handles_labels()
    ax.legend(handles, labels)
    return ax


def plot_multiforecast_component(
    fcst,
    comp_name,
    plot_name=None,
    ax=None,
    figsize=(10, 6),
    multiplicative=False,
    bar=False,
    focus=1,
    num_overplot=None,
):
    """Plot a particular component of the forecast.

    Parameters
    ----------
        fcst : pd.DataFrame
            Output of m.predict.
        comp_name : str
            Name of the component to plot.
        plot_name : str
            Name of the plot Title.
        ax : matplotlib axis
            Matplotlib Axes to plot on.
        figsize : tuple
            Width, height in inches, ignored if ax is not None.

            Note
            ----
            Default value is set to ``figsize = (10, 6)``

        multiplicative : bool
            Set y axis as percentage
        bar : bool
            Make barplot
        focus : int
            Forecast number to portray in detail.
        num_overplot : int
            Overplot all forecasts up to num

            Note
            ----
            Default value is set to ``num_overplot = None`` -> only plot focus

    Returns
    -------
        matplotlib.artist.Artist
            List of Artist objects containing a particular forecast component
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)
    fcst_t = fcst["ds"].dt.to_pydatetime()
    col_names = [col_name for col_name in fcst.columns if col_name.startswith(comp_name)]
    if num_overplot is not None:
        assert num_overplot <= len(col_names)
        for i in list(range(num_overplot))[::-1]:
            y = fcst[f"{comp_name}{i + 1}"]
            y = y.values
            alpha_min = 0.2
            alpha_softness = 1.2
            alpha = alpha_min + alpha_softness * (1.0 - alpha_min) / (i + 1.0 * alpha_softness)
            y[-1] = 0
            if bar:
                artists += ax.bar(fcst_t, y, width=1.00, color="#0072B2", alpha=alpha)
            else:
                artists += ax.plot(fcst_t, y, ls="-", color="#0072B2", alpha=alpha)
    if num_overplot is None or focus > 1:
        y = fcst[f"{comp_name}{focus}"]
        y = y.values
        y[-1] = 0
        if bar:
            artists += ax.bar(fcst_t, y, width=1.00, color="b")
        else:
            artists += ax.plot(fcst_t, y, ls="-", color="b")
    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which="major", color="gray", ls="-", lw=1, alpha=0.2)
    ax.set_xlabel("ds")
    if plot_name is None:
        plot_name = comp_name
    ax.set_ylabel(plot_name)
    if multiplicative:
        ax = set_y_as_percent(ax)
    return artists


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
        matplotlib.pyplot.figure
            Figure showing the nonconformity score with horizontal line for q-value based on the significance level or alpha
    """
    confidence_levels = np.arange(len(scores)) / len(scores)
    fig, ax = plt.subplots()
    ax.plot(confidence_levels, scores, label="score")
    ax.axvline(x=1 - alpha, color="g", linestyle="-", label=f"(1-alpha) = {1-alpha}", linewidth=1)
    ax.axhline(y=q, color="r", linestyle="-", label=f"q1 = {round(q, 2)}", linewidth=1)
    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("One-Sided Interval Width")
    ax.set_title(f"{method} One-Sided Interval Width with q")
    ax.legend()
    return fig

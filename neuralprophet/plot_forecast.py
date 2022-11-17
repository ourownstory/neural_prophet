import logging

import numpy as np

from neuralprophet.plot_model_parameters import plot_custom_season, plot_daily, plot_weekly, plot_yearly
from neuralprophet.utils import set_y_as_percent

log = logging.getLogger("NP.plotting")

try:
    from matplotlib import pyplot as plt
    from matplotlib.dates import AutoDateFormatter, AutoDateLocator
    from pandas.plotting import deregister_matplotlib_converters

    deregister_matplotlib_converters()
except ImportError:
    log.error("Importing matplotlib failed. Plotting will not work.")


def plot(
    fcst, quantiles, ax=None, xlabel="ds", ylabel="y", highlight_forecast=None, line_per_origin=False, figsize=(10, 6)
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
        for i, name in enumerate(reversed(yhat_col_names_no_qts)):
            ax.plot(
                ds,
                fcst[name],
                ls="-",
                c="#0072B2",
                alpha=0.2 + 2.0 / (i + 2.5),
                label=f"{colname}{i if line_per_origin else i + 1}",
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
    df_name="__df__",
    quantile=0.5,
    forecast_in_focus=None,
    one_period_per_season=True,
    residuals=False,
    figsize=None,
):
    """Plot the NeuralProphet forecast components.

    Parameters
    ----------
        m : NeuralProphet
            Fitted model
        fcst : pd.DataFrame
            Output of m.predict
        df_name : str
            ID from time series that should be plotted
        quantile : float
            Quantile for which the forecast components are to be plotted
        forecast_in_focus : int
            n-th step ahead forecast AR-coefficients to plot
        one_period_per_season : bool
            Plot one period per season, instead of the true seasonal components of the forecast.
        residuals : bool
            Flag whether to plot the residuals or not.
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
            # 'add_x': True})

    # Add lagged regressors
    if m.model.config_lagged_regressors is not None:
        for name in m.model.config_lagged_regressors.keys():
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
    # Plot  quantiles as a separate component, if present
    if len(m.model.quantiles) > 1 and forecast_in_focus is None:
        for i in range(1, len(m.model.quantiles)):
            components.append(
                {
                    "plot_name": "Uncertainty",
                    "comp_name": f"yhat1 {round(m.model.quantiles[i] * 100, 1)}%",
                    "fill": True,
                }
            )
    elif len(m.model.quantiles) > 1 and forecast_in_focus is not None:
        for i in range(1, len(m.model.quantiles)):
            components.append(
                {
                    "plot_name": "Uncertainty",
                    "comp_name": f"yhat{forecast_in_focus} {round(m.model.quantiles[i] * 100, 1)}%",
                    "fill": True,
                }
            )

    # set number of axes based on selected plot_names and sort them according to order in components
    panel_names = list(set(next(iter(dic.values())).lower() for dic in components))
    panel_order = [x for dic in components for x in panel_names if x in dic["plot_name"].lower()]
    npanel = len(panel_names)
    figsize = figsize if figsize else (10, 3 * npanel)
    fig, axes = plt.subplots(npanel, 1, facecolor="w", figsize=figsize)
    if npanel == 1:
        axes = [axes]
    multiplicative_axes = []
    ax = 0
    # for ax, comp in zip(axes, components):
    for comp in components:
        name = comp["plot_name"].lower()
        ax = axes[panel_order.index(name)]
        if (
            name in ["trend"]
            or ("residuals" in name and "ahead" in name)
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
        elif "auto-regression" in name or "lagged regressor" in name or "residuals" in name:
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
    if "residual" in comp_name:
        y[-1] = 0
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
            notnull = y.notnull()
            y = y.values
            alpha_min = 0.2
            alpha_softness = 1.2
            alpha = alpha_min + alpha_softness * (1.0 - alpha_min) / (i + 1.0 * alpha_softness)
            if "residual" not in comp_name:
                pass
                # fcst_t=fcst_t[notnull]
                # y = y[notnull]
            else:
                y[-1] = 0
            if bar:
                artists += ax.bar(fcst_t, y, width=1.00, color="#0072B2", alpha=alpha)
            else:
                artists += ax.plot(fcst_t, y, ls="-", color="#0072B2", alpha=alpha)
    if num_overplot is None or focus > 1:
        y = fcst[f"{comp_name}{focus}"]
        notnull = y.notnull()
        y = y.values
        if "residual" not in comp_name:
            fcst_t = fcst_t[notnull]
            y = y[notnull]
        else:
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

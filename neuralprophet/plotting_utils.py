import numpy as np
import pandas as pd

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
    print('Importing matplotlib failed. Plotting will not work.')


def set_y_as_percent(ax):
    """Set y axis as percentage

    Args:
        ax (matplotlib axis):

    Returns:
        ax
    """
    yticks = 100 * ax.get_yticks()
    yticklabels = ['{0:.4g}%'.format(y) for y in yticks]
    ax.set_yticklabels(yticklabels)
    return ax


def plot(fcst, ax=None, xlabel='ds', ylabel='y', highlight_forecast=None, figsize=(10, 6)):
    """Plot the NeuralProphet forecast

    Args:
        fcst (pd.DataFrame):  output of m.predict.
        ax (matplotlib axes):  on which to plot.
        xlabel (str): label name on X-axis
        ylabel (str): label name on Y-axis
        highlight_forecast (int): i-th step ahead forecast to highlight.
        figsize (tuple): width, height in inches.

    Returns:
        A matplotlib figure.
    """
    if ax is None:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    ds = fcst['ds'].dt.to_pydatetime()

    yhat_col_names = [col_name for col_name in fcst.columns if 'yhat' in col_name]
    for i in range(len(yhat_col_names)):
        ax.plot(ds, fcst['yhat{}'.format(i + 1)], ls='-', c='#0072B2', alpha=0.2 + 2.0/(i+2.5))
        # Future Todo: use fill_between for all but highlight_forecast
        """
        col1 = 'yhat{}'.format(i+1)
        col2 = 'yhat{}'.format(i+2)
        no_na1 = fcst.copy()[col1].notnull().values
        no_na2 = fcst.copy()[col2].notnull().values
        no_na = [x1 and x2 for x1, x2 in zip(no_na1, no_na2)]
        fcst_na = fcst.copy()[no_na]
        fcst_na_t = fcst_na['ds'].dt.to_pydatetime()
        ax.fill_between(
            fcst_na_t,
            fcst_na[col1],
            fcst_na[col2],
            color='#0072B2', alpha=1.0/(i+1)
            )
        """
    if highlight_forecast is not None:
        ax.plot(ds, fcst['yhat{}'.format(highlight_forecast)], ls='-', c='b')
        ax.plot(ds, fcst['yhat{}'.format(highlight_forecast)], 'bx')

    ax.plot(ds, fcst['y'], 'k.')

    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_components(m, fcst, forecast_in_focus=None, figsize=None):
    """Plot the NeuralProphet forecast components.

    Args:
        m (NeuralProphet): fitted model.
        fcst (pd.DataFrame):  output of m.predict.
        forecast_in_focus (int): n-th step ahead forecast AR-coefficients to plot
        figsize (tuple): width, height in inches.

    Returns:
        A matplotlib figure.
    """
    # Identify components to be plotted
    # as dict, minimum: {plot_name, comp_name}
    components = [{'plot_name': 'Trend',
                   'comp_name': 'trend'}]

    # print(fcst.head().to_string())
    # Future TODO: Add Holidays
    # if m.train_holiday_names is not None and 'holidays' in fcst:
    #     components.append('holidays')

    ## Plot  seasonalities, if present
    if m.season_config is not None:
        for name in m.season_config.periods:
            if name in m.season_config.periods: # and name in fcst:
                components.append({'plot_name': '{} seasonality'.format(name),
                                   'comp_name': 'season_{}'.format(name)})

    if m.n_lags > 0:
        components.append({'plot_name': 'Auto-Regression',
                           'comp_name': 'ar',
                           'num_overplot': m.n_forecasts,
                           'bar': True})
        if forecast_in_focus is not None:
            components.append({'plot_name': 'AR Forecast {}'.format(forecast_in_focus),
                               'comp_name': 'ar{}'.format(forecast_in_focus)})

    # Add Covariates
    if m.covar_config is not None:
        for name in m.covar_config.keys():
            components.append({'plot_name': 'Covariate "{}"'.format(name),
                               'comp_name': 'covar_{}'.format(name),
                               'num_overplot': m.n_forecasts,
                               'bar': True})
            if forecast_in_focus is not None:
                components.append({'plot_name': 'COV "{}" Forecast {}'.format(name, forecast_in_focus),
                                   'comp_name': 'covar_{}{}'.format(name, forecast_in_focus)})
    if 'residuals' in fcst:
        components.append({'plot_name': 'Residuals',
                           'comp_name': 'residuals',
                           'rolling': 7,
                           'bar': True})
    npanel = len(components)
    figsize = figsize if figsize else (9, 3 * npanel)
    fig, axes = plt.subplots(npanel, 1, facecolor='w', figsize=figsize)
    if npanel == 1:
        axes = [axes]
    multiplicative_axes = []
    for ax, comp in zip(axes, components):
        name = comp['plot_name'].lower()
        if name in ['trend', 'residuals'] \
                or ('ar' in name and 'forecast' in name) \
                or ('cov' in name and 'forecast' in name):
            plot_forecast_component(fcst=fcst, ax=ax, **comp)
        elif 'season' in name:
            if m.season_config.mode == 'multiplicative':
                multiplicative_axes.append(ax)
            plot_forecast_component(fcst=fcst, ax=ax, **comp)
        elif 'auto-regression' in name or 'covariate' in name:
            plot_multiforecast_component(fcst=fcst, ax=ax, **comp)

    fig.tight_layout()
    # Reset multiplicative axes labels after tight_layout adjustment
    for ax in multiplicative_axes: ax = set_y_as_percent(ax)
    return fig


def plot_forecast_component(fcst, comp_name, plot_name=None, ax=None, figsize=(10, 6),
                            multiplicative=False, bar=False, rolling=None):
    """Plot a particular component of the forecast.

    Args:
        fcst (pd.DataFrame):  output of m.predict.
        comp_name (str): Name of the component to plot.
        plot_name (str): Name of the plot Title.
        ax (matplotlib axis): matplotlib Axes to plot on.
        figsize (tuple): width, height in inches.
        multiplicative (bool): set y axis as percentage
        bar (bool): make barplot
        rolling (int): rolling average underplot

    Returns:
        a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    fcst_t = fcst['ds'].dt.to_pydatetime()
    if rolling is not None:
        rolling_avg = fcst[comp_name].rolling(rolling, min_periods=1, center=True).mean()
        if bar: artists += ax.bar(fcst_t, rolling_avg, width=1.00, color='#0072B2', alpha=0.5)
        else: artists += ax.plot(fcst_t, rolling_avg, ls='-', color='#0072B2', alpha=0.5)
    if bar: artists += ax.bar(fcst_t, fcst[comp_name], width=1.00, color='#0072B2')
    else: artists += ax.plot(fcst_t, fcst[comp_name], ls='-', c='#0072B2')
    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel('ds')
    if plot_name is None: plot_name = comp_name
    ax.set_ylabel(plot_name)
    if multiplicative: ax = set_y_as_percent(ax)
    return artists


def plot_multiforecast_component(fcst, comp_name, plot_name=None, ax=None, figsize=(10, 6),
                                 multiplicative=False, bar=False, focus=1, num_overplot=None):
    """Plot a particular component of the forecast.

    Args:
        fcst (pd.DataFrame):  output of m.predict.
        comp_name (str): Name of the component to plot.
        plot_name (str): Name of the plot Title.
        ax (matplotlib axis): matplotlib Axes to plot on.
        figsize (tuple): width, height in inches.
        multiplicative (bool): set y axis as percentage
        bar (bool): make barplot
        focus (int): forecast number to portray in detail.
        num_overplot (int): overplot all forecasts up to num
            None (default): only plot focus

    Returns:
        a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    fcst_t = fcst['ds'].dt.to_pydatetime()
    col_names = [col_name for col_name in fcst.columns if col_name.startswith(comp_name)]
    assert num_overplot <= len(col_names)
    if num_overplot is not None:
        for i in list(range(num_overplot))[::-1]:
            y = fcst['{}{}'.format(comp_name, i+1)]
            notnull = y.notnull()
            alpha_min = 0.2
            alpha_softness = 1.2
            alpha = alpha_min + alpha_softness*(1.0-alpha_min) / (i + 1.0*alpha_softness)
            if bar:
                artists += ax.bar(fcst_t[notnull], y[notnull], width=1.00, color='#0072B2',  alpha=alpha)
            else:
                artists += ax.plot(fcst_t[notnull], y[notnull], ls='-', color='#0072B2',  alpha=alpha)
    if num_overplot is None or focus > 1:
        y = fcst['{}{}'.format(comp_name, focus)]
        notnull = y.notnull()
        if bar:
            artists += ax.bar(fcst_t[notnull], y[notnull], width=1.00, color='b')
        else:
            artists += ax.plot(fcst_t[notnull], y[notnull], ls='-', color='b')
    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', color='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel('ds')
    if plot_name is None: plot_name = comp_name
    ax.set_ylabel(plot_name)
    if multiplicative: ax = set_y_as_percent(ax)
    return artists


# def plot_parameters(m, weekly_start=0, yearly_start=0, forecast_in_focus=None, figsize=None,):
#     """Plot the parameters that the model is composed of, visually.
#
#     Args:
#         m (NeuralProphet): fitted model.
#         weekly_start (int):  specifying the start day of the weekly seasonality plot.
#             0 (default) starts the week on Sunday.
#             1 shifts by 1 day to Monday, and so on.
#         yearly_start (int): specifying the start day of the yearly seasonality plot.
#             0 (default) starts the year on Jan 1.
#             1 shifts by 1 day to Jan 2, and so on.
#         forecast_in_focus (int): n-th step ahead forecast AR-coefficients to plot
#         figsize (tuple): width, height in inches.
#
#     Returns:
#         A matplotlib figure.
#     """
#     # Identify components to be plotted
#     # as tuple: (plot_type, name, kwargs)
#     components = []
#     if m.n_changepoints > 0:
#         components.append(('trend', 'Trend-Changepoints', {}))
#
#     # Future TODO: Add Holidays
#
#     ## Plot  seasonalities, if present
#     if m.season_config is not None:
#         default_seasons = ['weekly', 'yearly']
#         for name in m.season_config.periods:
#             if name in default_seasons:
#                 components.append(('season', name, {}))
#             else:
#                 ## Other seasonalities: TODO: implement plotting
#                 pass
#
#     # Add Covariates
#     # if m.covar_config is not None:
#     #     for name in m.covar_config.keys():
#     #         components.append(('season', 'covar_{}'.format(name), {}))
#
#     if m.n_lags > 1:
#         components.append(('AR-Importance', 'AR', {}))
#     if m.n_lags > 0 and forecast_in_focus is not None:
#         components.append(('AR-Detail', 'AR', {}))
#
#     # Identify components to be plotted
#     # as dict, minimum: {plot_name}
#     components = [{'plot_name': 'Trend', 'comp_name': 'trend'}]
#
#     # print(fcst.head().to_string())
#     # Future TODO: Add Holidays
#     # if m.train_holiday_names is not None and 'holidays' in fcst:
#     #     components.append('holidays')
#
#     ## Plot  seasonalities, if present
#     if m.season_config is not None:
#         for name in m.season_config.periods:
#             if name in m.season_config.periods: # and name in fcst:
#                 components.append({'plot_name': 'Seasonality',
#                                    'comp_name': name})
#
#     if m.n_lags > 0:
#         components.append({'plot_name': 'Auto-Regression'})
#         if forecast_in_focus is not None:
#             components.append({'plot_name': 'Auto-Regression',
#                                'forecast_in_focus': forecast_in_focus})
#
#     # Add Covariates
#     if m.covar_config is not None:
#         for name in m.covar_config.keys():
#             components.append({'plot_name': 'Covariate',
#                                'comp_name': 'covar_{}'.format(name)})
#             if forecast_in_focus is not None:
#                 components.append({'plot_name': 'Covariate',
#                                    'comp_name': 'covar_{}'.format(name),
#                                    'forecast_in_focus': forecast_in_focus})
#     if 'residuals' in fcst:
#         components.append({'plot_name': 'Residuals',
#                            'comp_name': 'residuals',
#                            'rolling': 7,
#                            'bar': True})
#     npanel = len(components)
#     figsize = figsize if figsize else (9, 3 * npanel)
#     fig, axes = plt.subplots(npanel, 1, facecolor='w', figsize=figsize)
#     if npanel == 1:
#         axes = [axes]
#     multiplicative_axes = []
#     for ax, comp in zip(axes, components):
#         if comp['plot_name'] in ['Trend', 'Residuals']:
#             plot_forecast_component(fcst=fcst, ax=ax, **comp)
#         elif comp['plot_name'] == 'trend-changepoints':
#             plot_trend_change(m=m, ax=ax,)
#         elif m.season_config is not None and plot_name in m.season_config.periods:
#             if m.season_config.mode == 'multiplicative':
#                 multiplicative_axes.append(ax)
#             if comp['plot_name'] == 'weekly' or m.season_config.periods[plot_name]['period'] == 7:
#                 plot_weekly(m=m, name=plot_name, ax=ax, weekly_start=weekly_start, )
#             elif comp['plot_name'] == 'yearly' or m.season_config.periods[plot_name]['period'] == 365.25:
#                 plot_yearly(m=m, name=plot_name, ax=ax, yearly_start=yearly_start, )
#             # else:
#             #     plot_seasonality(name=plot_name, ax=ax, uncertainty=uncertainty,)
#         # elif plot_name in ['holidays', 'extra_regressors_additive', 'extra_regressors_multiplicative', ]:
#         #     plot_forecast_component(fcst=fcst, name=plot_name, ax=ax,)
#         elif comp['plot_name'] == 'AR':
#             plot_ar_weights_importance(m=m, ax=ax,)
#         elif comp['plot_name'] == 'AR-Detail':
#             plot_ar_weights_value(m=m, ax=ax, forecast_n=forecast_in_focus)
#
#         # if plot_name in m.component_modes['multiplicative']:
#         #     multiplicative_axes.append(ax)
#
#     fig.tight_layout()
#     # Reset multiplicative axes labels after tight_layout adjustment
#     for ax in multiplicative_axes:
#         ax = set_y_as_percent(ax)
#     return fig

def plot_trend_change(m, ax=None, figsize=(10, 6)):
    """Make a barplot of the magnitudes of trend-changes.

    Args:
        m (NeuralProphet): fitted model.
        ax (matplotlib axis): matplotlib Axes to plot on.
            One will be created if this is not provided.
        figsize (tuple): width, height in inches.

    Returns:
        a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)

    cp_range = range(0, 1 + m.n_changepoints)
    weights = m.model.get_trend_deltas.detach().numpy()
    artists += ax.bar(cp_range, weights, width=1.00, color='#0072B2')

    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel("Trend Segment")
    ax.set_ylabel('Trend Change')
    return artists


def plot_ar_weights_importance(m, ax=None, figsize=(10, 6)):
    """ Make a barplot of the relative importance of AR-lags.

    Args:
        m (NeuralProphet): fitted model.
        ax (matplotlib axis): matplotlib Axes to plot on.
            One will be created if this is not provided.
        figsize (tuple): width, height in inches.

    Returns:
        a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)

    # lags_range = [str(x) for x in range(1, 1 + m.n_lags)][::-1]
    lags_range = list(range(1, 1 + m.n_lags))[::-1]
    weights = m.model.ar_weights.detach().numpy()
    weights_imp = np.sum(np.abs(weights), axis=0)
    weights_imp = weights_imp / np.sum(weights_imp)
    artists += ax.bar(lags_range, weights_imp, width=1.00, color='#0072B2')

    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel("AR lag number")
    ax.set_ylabel('Relative importance')
    ax = set_y_as_percent(ax)
    return artists


def plot_ar_weights_value(m, forecast_n, ax=None, figsize=(10, 6)):
    """Make a barplot of the actual weights of AR-lags for a specific forecast-position.

    Args:
        m (NeuralProphet): fitted model.
        ax (matplotlib axis): matplotlib Axes to plot on.
            One will be created if this is not provided.
        figsize (tuple): width, height in inches.
        forecast_n (int): n-th step ahead forecast AR-coefficients to plot

    Returns:
        a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)

    # lags_range = [str(x) for x in range(1, 1 + m.n_lags)][::-1]
    lags_range = list(range(1, 1 + m.n_lags))[::-1]
    weights = m.model.ar_weights.detach().numpy()
    weights_detail = weights[forecast_n-1, :]
    artists += ax.bar(lags_range, weights_detail, width=0.80, color='#0072B2')

    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel("AR lag number")
    ax.set_ylabel('Weight for forecast {}'.format(forecast_n))
    return artists




def plot_yearly(m, ax=None, yearly_start=0, figsize=(10, 6), name='yearly'):
    """Plot the yearly component of the forecast.

    Args:
        m (NeuralProphet): fitted model.
        ax (matplotlib axis): matplotlib Axes to plot on.
            One will be created if this is not provided.
        yearly_start (int): specifying the start day of the yearly seasonality plot.
            0 (default) starts the year on Jan 1.
            1 shifts by 1 day to Jan 2, and so on.
        figsize (tuple): width, height in inches.
        name (str): Name of seasonality component if previously changed from default 'yearly'.

    Returns:
        a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    # Compute yearly seasonality for a Jan 1 - Dec 31 sequence of dates.
    days = (pd.date_range(start='2017-01-01', periods=365) +
            pd.Timedelta(days=yearly_start))
    df_y = pd.DataFrame({'ds': days, 'y': np.zeros_like(days)})
    seas = m.predict_seasonal_components(df_y)
    artists += ax.plot(
        df_y['ds'].dt.to_pydatetime(), seas[name], ls='-', c='#0072B2')
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    months = MonthLocator(range(1, 13), bymonthday=1, interval=2)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos=None: '{dt:%B} {dt.day}'.format(dt=num2date(x))))
    ax.xaxis.set_major_locator(months)
    ax.set_xlabel('Day of year')
    ax.set_ylabel(name)
    if m.season_config.mode == 'multiplicative':
        ax = set_y_as_percent(ax)
    return artists


def plot_weekly(m, ax=None, weekly_start=0, figsize=(10, 6), name='weekly'):
    """Plot the yearly component of the forecast.

    Args:
        m (NeuralProphet): fitted model.
        ax (matplotlib axis): matplotlib Axes to plot on.
            One will be created if this is not provided.
        weekly_start (int): specifying the start day of the weekly seasonality plot.
            0 (default) starts the week on Sunday.
            1 shifts by 1 day to Monday, and so on.
        figsize (tuple): width, height in inches.
        name (str): Name of seasonality component if previously changed from default 'weekly'.

    Returns:
        a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    # Compute weekly seasonality for a Sun-Sat sequence of dates.
    days = (pd.date_range(start='2017-01-01', periods=7) +
            pd.Timedelta(days=weekly_start))
    df_w = pd.DataFrame({'ds': days, 'y': np.zeros_like(days)})
    seas = m.predict_seasonal_components(df_w)
    days = days.day_name()
    artists += ax.plot(range(len(days)), seas[name], ls='-',
                    c='#0072B2')
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days)
    ax.set_xlabel('Day of week')
    ax.set_ylabel(name)
    if m.season_config.mode == 'multiplicative':
        ax = set_y_as_percent(ax)
    return artists

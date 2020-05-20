# import numpy as np

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
    yticks = 100 * ax.get_yticks()
    yticklabels = ['{0:.4g}%'.format(y) for y in yticks]
    ax.set_yticklabels(yticklabels)
    return ax


def plot(history, fcst, ax=None,
         # uncertainty=True, plot_cap=True,
         xlabel='ds', ylabel='y',
         multi_forecast=None,
         figsize=(10, 6),
         ):
    """Plot the Prophet forecast.

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    ax: Optional matplotlib axes on which to plot.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    xlabel: Optional label name on X-axis
    ylabel: Optional label name on Y-axis
    figsize: Optional tuple width, height in inches.

    Returns
    -------
    A matplotlib figure.
    """
    if ax is None:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    fcst_t = fcst['ds'].dt.to_pydatetime()
    ax.plot(history['ds'].dt.to_pydatetime(), history['y'], 'k.')


    if multi_forecast is not None:
        for i in range(multi_forecast):
            ax.plot(fcst_t, fcst['yhat{}'.format(i + 1)], ls='-', c='#0072B2', alpha=1.0/(i+1))
            # fill_between
            # col1 = 'yhat{}'.format(i+1)
            # col2 = 'yhat{}'.format(i+2)
            # no_na1 = fcst.copy()[col1].notnull().values
            # no_na2 = fcst.copy()[col2].notnull().values
            # no_na = [x1 and x2 for x1, x2 in zip(no_na1, no_na2)]
            # fcst_na = fcst.copy()[no_na]
            # fcst_na_t = fcst_na['ds'].dt.to_pydatetime()
            # ax.fill_between(
            #     fcst_na_t,
            #     fcst_na[col1],
            #     fcst_na[col2],
            #     color='#0072B2', alpha=1.0/(i+1)
            # )

    ax.plot(fcst_t, fcst['yhat'], ls='-', c='b')

    # just for debugging
    # ax.plot(fcst_t, fcst['actual'], ls='-', c='r')

    # Future TODO: logistic/limited growth?
    # if 'cap' in fcst and plot_cap:
    #     ax.plot(fcst_t, fcst['cap'], ls='--', c='k')
    # if m.logistic_floor and 'floor' in fcst and plot_cap:
    #     ax.plot(fcst_t, fcst['floor'], ls='--', c='k')
    # if uncertainty and m.uncertainty_samples:
    #     ax.fill_between(fcst_t, fcst['yhat_lower'], fcst['yhat_upper'],
    #                     color='#0072B2', alpha=0.2)

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


def plot_components(fcst,
                    # uncertainty=True, plot_cap=True,
                    # weekly_start=0, yearly_start=0,
                    figsize=None
                    ):
    """Plot the Prophet forecast components.

    Will plot whichever are available of: trend, holidays, weekly
    seasonality, yearly seasonality, and additive and multiplicative extra
    regressors.

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    weekly_start: Optional int specifying the start day of the weekly
        seasonality plot. 0 (default) starts the week on Sunday. 1 shifts
        by 1 day to Monday, and so on.
    yearly_start: Optional int specifying the start day of the yearly
        seasonality plot. 0 (default) starts the year on Jan 1. 1 shifts
        by 1 day to Jan 2, and so on.
    figsize: Optional tuple width, height in inches.

    Returns
    -------
    A matplotlib figure.
    """
    # Identify components to be plotted
    components = ['trend']
    # Future TODO: Add Holidays
    # if m.train_holiday_names is not None and 'holidays' in fcst:
    #     components.append('holidays')
    # Future TODO: Add Seasonalities
    # # Plot weekly seasonality, if present
    # if 'weekly' in m.seasonalities and 'weekly' in fcst:
    #     components.append('weekly')
    # # Yearly if present
    # if 'yearly' in m.seasonalities and 'yearly' in fcst:
    #     components.append('yearly')
    # # Other seasonalities
    # components.extend([
    #     name for name in sorted(m.seasonalities)
    #     if name in fcst and name not in ['weekly', 'yearly']
    # ])

    # Future TODO: Add Regressors
    # regressors = {'additive': False, 'multiplicative': False}
    # for name, props in m.extra_regressors.items():
    #     regressors[props['mode']] = True
    # for mode in ['additive', 'multiplicative']:
    #     if regressors[mode] and 'extra_regressors_{}'.format(mode) in fcst:
    #         components.append('extra_regressors_{}'.format(mode))
    npanel = len(components)

    figsize = figsize if figsize else (9, 3 * npanel)
    fig, axes = plt.subplots(npanel, 1, facecolor='w', figsize=figsize)

    if npanel == 1:
        axes = [axes]

    # multiplicative_axes = []

    for ax, plot_name in zip(axes, components):
        if plot_name == 'trend':
            plot_forecast_component(
                fcst=fcst, name='trend', ax=ax,
                # uncertainty=uncertainty, plot_cap=plot_cap,
            )
        # elif plot_name in m.seasonalities:
        #     if plot_name == 'weekly' or m.seasonalities[plot_name]['period'] == 7:
        #         plot_weekly(
        #             m=m, name=plot_name, ax=ax, uncertainty=uncertainty, weekly_start=weekly_start
        #         )
        #     elif plot_name == 'yearly' or m.seasonalities[plot_name]['period'] == 365.25:
        #         plot_yearly(
        #             m=m, name=plot_name, ax=ax, uncertainty=uncertainty, yearly_start=yearly_start
        #         )
        #     else:
        #         plot_seasonality(
        #             m=m, name=plot_name, ax=ax, uncertainty=uncertainty,
        #         )
        # elif plot_name in [
        #     'holidays',
        #     'extra_regressors_additive',
        #     'extra_regressors_multiplicative',
        # ]:
        #     plot_forecast_component(
        #         m=m, fcst=fcst, name=plot_name, ax=ax, uncertainty=uncertainty,
        #         plot_cap=False,
        #     )
        # if plot_name in m.component_modes['multiplicative']:
        #     multiplicative_axes.append(ax)

    fig.tight_layout()
    # Reset multiplicative axes labels after tight_layout adjustment
    # for ax in multiplicative_axes:
    #     ax = set_y_as_percent(ax)
    return fig


def plot_forecast_component(
        fcst, name, ax=None,
        # uncertainty=True, plot_cap=False,
        figsize=(10, 6)
):
    """Plot a particular component of the forecast.

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    name: Name of the component to plot.
    ax: Optional matplotlib Axes to plot on.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    figsize: Optional tuple width, height in inches.

    Returns
    -------
    a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    fcst_t = fcst['ds'].dt.to_pydatetime()
    artists += ax.plot(fcst_t, fcst[name], ls='-', c='#0072B2')
    # if 'cap' in fcst and plot_cap:
    #     artists += ax.plot(fcst_t, fcst['cap'], ls='--', c='k')
    # if m.logistic_floor and 'floor' in fcst and plot_cap:
    #     ax.plot(fcst_t, fcst['floor'], ls='--', c='k')
    # if uncertainty and m.uncertainty_samples:
    #     artists += [ax.fill_between(
    #         fcst_t, fcst[name + '_lower'], fcst[name + '_upper'],
    #         color='#0072B2', alpha=0.2)]
    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel('ds')
    ax.set_ylabel(name)
    # if name in m.component_modes['multiplicative']:
    #     ax = set_y_as_percent(ax)
    return artists

import numpy as np

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


def get_regularization_lambda(sparsity, lambda_delay_epochs=None, epoch=None):
    if sparsity is not None:
        lam = 0.02 * (1.0 / sparsity - 1.0)
        if lambda_delay_epochs is not None and epoch < lambda_delay_epochs:
            lam = lam * epoch / (1.0 * lambda_delay_epochs)
            # lam = lam * (epoch / (1.0 * lambda_delay_epochs))**2
    else:
        lam = None
    return lam


def symmetric_total_percentage_error(values, estimates):
    sum_abs_diff = np.sum(np.abs(estimates - values))
    sum_abs = np.sum(np.abs(estimates) + np.abs(values))
    return 100 * sum_abs_diff / (10e-9 + sum_abs)


def piecewise_linear(t, deltas, k, m, changepoint_ts):
    """Evaluate the piecewise linear function.

    Parameters
    ----------
    t: np.array of times on which the function is evaluated.
    deltas: np.array of rate changes at each changepoint.
    k: Float initial rate.
    m: Float initial offset.
    changepoint_ts: np.array of changepoint times.

    Returns
    -------
    Vector y(t).
    """
    # Intercept changes
    gammas = -changepoint_ts * deltas
    # Get cumulative slope and intercept at each t
    k_t = k * np.ones_like(t)
    m_t = m * np.ones_like(t)
    for s, t_s in enumerate(changepoint_ts):
        indx = t >= t_s
        k_t[indx] += deltas[s]
        m_t[indx] += gammas[s]
    return k_t * t + m_t




def plot(history, fcst, ax=None, uncertainty=True, plot_cap=True, xlabel='ds', ylabel='y', figsize=(10, 6),
         multi_forecast=None
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

    ax.plot(fcst_t, fcst['yhat'], ls='-', c='r')

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


def plot_multiple_forecasts(history, fcst, ax=None, uncertainty=True, plot_cap=True, xlabel='ds', ylabel='y', figsize=(10, 6)
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
    ax.plot(fcst_t, fcst['yhat'], ls='-', c='#0072B2')

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

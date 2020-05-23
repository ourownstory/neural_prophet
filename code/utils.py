import numpy as np
import pandas as pd
import torch


def get_regularization_lambda(sparsity, lambda_delay_epochs=None, epoch=None):
    if sparsity is not None:
        lam = 0.02 * (1.0 / sparsity - 1.0)
        if lambda_delay_epochs is not None and epoch < lambda_delay_epochs:
            lam = lam * epoch / (1.0 * lambda_delay_epochs)
            # lam = lam * (epoch / (1.0 * lambda_delay_epochs))**2
    else:
        lam = None
    return lam


def regulariziation_function(weights):
    abs_weights = torch.abs(weights)
    reg = torch.div(2.0, 1.0 + torch.exp(-3.0 * abs_weights.pow(1.0 / 3.0))) - 1.0
    return reg

def regulariziation_function_trend(weights):
    abs_weights = torch.abs(weights)
    # reg = torch.div(2.0, 1.0 + torch.exp(-3.0 * abs_weights.pow(1.0 / 3.0))) - 1.0
    reg = torch.sqrt(abs_weights)
    # reg = torch.log(1 + abs_weights)
    # reg = torch.abs(weights)
    return reg


def symmetric_total_percentage_error(values, estimates):
    sum_abs_diff = np.sum(np.abs(estimates - values))
    sum_abs = np.sum(np.abs(estimates) + np.abs(values))
    return 100 * sum_abs_diff / (10e-9 + sum_abs)


def piecewise_linear(t, k, m, changepoints_t):
    """Evaluate the piecewise linear function.

    Parameters
    ----------
    t: np.array of times on which the function is evaluated.
    k: np.array of rates after each changepoint.
    m: np.array of offset after each changepoint.
    changepoints_t: np.array of changepoint times, including zero.

    Returns
    -------
    Vector y(t).
    """
    print("WARNING: deprecated, might contain bug.")
    t = np.squeeze(t)
    past_changepoint = np.expand_dims(t, 1) >= np.expand_dims(changepoints_t, 0)
    segment_id = np.sum(past_changepoint, axis=1) - 1

    k_t = np.ones((len(t), 1)) * np.expand_dims(k, 0)
    m_t = np.ones((len(t), 1)) * np.expand_dims(m, 0)
    k_t = np.squeeze(k_t[np.arange(len(t)), segment_id])
    m_t = np.squeeze(m_t[np.arange(len(t)), segment_id])

    trend = k_t * t + m_t
    return trend


def piecewise_linear_prophet(t, k, m, deltas=None, changepoints_t=None):
    """Evaluate the piecewise linear function.

    Parameters
    ----------
    t: np.array of times on which the function is evaluated.
    deltas: np.array of rate changes at each changepoint.
    k: Float initial rate.
    m: Float initial offset.
    changepoints_t: np.array of changepoint times.

    Returns
    -------
    Vector y(t).
    """
    # Get cumulative slope and intercept at each t
    k_t = k * np.ones_like(t)
    m_t = m * np.ones_like(t)
    # Intercept changes

    if deltas is not None and changepoints_t is not None:
        gammas = -changepoints_t * deltas
        for s, t_s in enumerate(changepoints_t):
            indx = t >= t_s
            k_t[indx] += deltas[s]
            m_t[indx] += gammas[s]
    return k_t * t + m_t


def make_future_dataframe(history_dates, periods, freq='D', include_history=True):
    """Simulate the trend using the extrapolated generative model.

    Parameters
    ----------
    periods: Int number of periods to forecast forward.
    freq: Any valid frequency for pd.date_range, such as 'D' or 'M'.
    include_history: Boolean to include the historical dates in the data
        frame for predictions.

    Returns
    -------
    pd.Dataframe that extends forward from the end of self.history for the
    requested number of periods.
    """
    if history_dates is None:
        raise Exception('Model has not been fit.')
    last_date = history_dates.max()
    dates = pd.date_range(
        start=last_date,
        periods=periods + 1,  # An extra in case we include start
        freq=freq)
    dates = dates[dates > last_date]  # Drop start if equals last_date
    dates = dates[:periods]  # Return correct number of periods

    if include_history:
        dates = np.concatenate((np.array(history_dates), dates))

    return pd.DataFrame({'ds': dates})

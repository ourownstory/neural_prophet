import numpy as np
import pandas as pd
import torch


def get_regularization_lambda(sparsity, lambda_delay_epochs=None, epoch=None):
    if sparsity is not None and sparsity < 1:
        lam = 0.02 * (1.0 / sparsity - 1.0)
        if lambda_delay_epochs is not None and epoch < lambda_delay_epochs:
            lam = lam * epoch / (1.0 * lambda_delay_epochs)
            # lam = lam * (epoch / (1.0 * lambda_delay_epochs))**2
    else:
        lam = None
    return lam


def regulariziation_function_ar(weights):
    abs_weights = torch.abs(weights)
    reg = torch.div(2.0, 1.0 + torch.exp(-3.0 * abs_weights.pow(1.0 / 3.0))) - 1.0
    return reg

def regulariziation_function_trend(weights, threshold=None):
    abs_weights = torch.abs(weights)
    if threshold is not None:
        abs_weights = torch.clamp(abs_weights - threshold, min=0.0)
    # reg = 10*regulariziation_function_ar(abs_weights)
    reg = torch.abs(abs_weights)
    return reg


def symmetric_total_percentage_error(values, estimates):
    sum_abs_diff = np.sum(np.abs(estimates - values))
    sum_abs = np.sum(np.abs(estimates) + np.abs(values))
    return 100 * sum_abs_diff / (10e-9 + sum_abs)


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

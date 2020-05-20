import numpy as np


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


def piecewise_linear(t, k, m, deltas=None, changepoints_t=None):
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

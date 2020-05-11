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

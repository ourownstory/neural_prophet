import inspect
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch_lr_finder import LRFinder
from captum.attr import Saliency

from neuralprophet import utils

log = logging.getLogger("NP.utils_torch")


def penalize_nonzero(weights, eagerness=1.0, acceptance=1.0):
    cliff = 1.0 / (np.e * eagerness)
    return torch.log(cliff + acceptance * torch.abs(weights)) - np.log(cliff)


def lr_range_test(
    model,
    dataset,
    loss_func,
    optimizer="AdamW",
    batch_size=32,
    num_iter=None,
    skip_start=10,
    skip_end=5,
    start_lr=1e-7,
    end_lr=100,
    plot=False,
):
    if num_iter is None:
        num_iter = 50 + int(np.log10(100 + len(dataset)) * 25)
    n_train = min(len(dataset), num_iter * batch_size)
    n_val = min(int(0.3 * len(dataset)), 2 * num_iter)
    log.debug(f"num_iter: {num_iter}, n_val: {n_val}")
    split_idx = int(0.7 * len(dataset))
    idx_train = np.random.choice(split_idx, size=n_train)
    idx_val = np.random.choice(np.arange(split_idx, len(dataset)), size=n_val)
    train_data = Subset(dataset, idx_train)
    val_data = Subset(dataset, idx_val)
    lrtest_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    lrtest_loader_val = DataLoader(val_data, batch_size=1024, shuffle=True)
    lrtest_optimizer = create_optimizer_from_config(optimizer, model.parameters(), start_lr)
    with utils.HiddenPrints():
        lr_finder = LRFinder(model, lrtest_optimizer, loss_func)
        lr_finder.range_test(
            lrtest_loader,
            val_loader=lrtest_loader_val,
            end_lr=end_lr,
            num_iter=num_iter,
            smooth_f=0.05,  # re-consider if lr-rate varies a lot
            diverge_th=50,
            step_mode="exp",
        )
        lrs = lr_finder.history["lr"]
        losses = lr_finder.history["loss"]
    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]
    if plot:
        with utils.HiddenPrints():
            ax, steepest_lr = lr_finder.plot()  # to inspect the loss-learning rate graph
    lr = None
    try:
        steep_idx = (np.gradient(np.array(losses))).argmin()
        min_idx = (np.array(losses)).argmin()
        steep_lr = lrs[steep_idx]
        min_lr = lrs[min_idx]
        lr = steep_lr
        # lr = 10 ** ((np.log10(steep_lr) + np.log10(min_lr)) / 2.0)
        log.info(f"lr-range-test results: steep: {steep_lr:.2E}, min: {min_lr:.2E}")
    except ValueError:
        log.error("Failed to compute the gradients, there might not be enough points.")
    if lr is None:
        lr = 0.1
    with utils.HiddenPrints():
        lr_finder.reset()  # to reset the model and optimizer to their initial state
    return lr


def create_optimizer_from_config(optimizer_name, model_parameters, lr):
    if type(optimizer_name) == str:
        if optimizer_name.lower() == "adamw":
            # Tends to overfit, but reliable
            optimizer = torch.optim.AdamW(model_parameters, lr=lr, weight_decay=1e-3)
        elif optimizer_name.lower() == "sgd":
            # better validation performance, but diverges sometimes
            optimizer = torch.optim.SGD(model_parameters, lr=lr, momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError
    elif inspect.isclass(optimizer_name) and issubclass(optimizer_name, torch.optim.Optimizer):
        optimizer = optimizer_name(model_parameters, lr=lr)
    else:
        raise ValueError
    return optimizer


def interprete_model(target_model, net, forward_func):
    """
    Returns model input attributions for a given network and forward function.

    Parameters
    ----------
        net : str
            Name of the network for which input attributions are to be computed.

        forward_func : str
            Name of the forward function for which input attributions are to be computed.

    Returns
    -------
        torch.Tensor
            Input attributions for the given network and forward function.
    """
    # Load the respective forward function from the model and init model interpreter
    forward = getattr(target_model, forward_func)
    saliency = Saliency(forward_func=forward)

    # Number of quantiles
    num_quantiles = len(target_model.quantiles)
    # Number of input features to the net (aka n_lags)
    in_features = getattr(target_model, net)[0].in_features
    # Number of output features from the net (aka n_forecasts)
    out_features = getattr(target_model, net)[-1].out_features
    out_features = int(out_features / num_quantiles)

    # Create a tensor of ones as model input
    model_input = torch.ones(1, in_features, requires_grad=True)

    # Iterate through each output feature and compute the model attribution wrt. the input
    attributions = torch.empty((0, in_features))
    for output_feature in range(out_features):
        for quantile in range(num_quantiles):
            target_attribution = saliency.attribute(model_input, target=[(output_feature, quantile)], abs=False)
            attributions = torch.cat((attributions, target_attribution), 0)

    # Average the attributions over the input features
    # Idea: Average attribution of each lag on all forecasts (eg. the n'th lag has an attribution of xyz on the forecast)
    # TODO: support the visualization of 2d tensors in plot_parameters (aka the attribution of the n'th lag on the m'th forecast)
    # attributions = torch.mean(attributions, 0)

    return attributions

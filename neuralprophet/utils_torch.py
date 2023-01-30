import inspect
import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from captum.attr import Saliency

log = logging.getLogger("NP.utils_torch")


def init_parameter(dims):
    """
    Create and initialize a new torch Parameter.

    Parameters
    ----------
        dims : list or tuple
            Desired dimensions of parameter

    Returns
    -------
        nn.Parameter
            initialized Parameter
    """
    if len(dims) > 1:
        return nn.Parameter(nn.init.xavier_normal_(torch.randn(dims)), requires_grad=True)
    else:
        return nn.Parameter(torch.nn.init.xavier_normal_(torch.randn([1] + dims)).squeeze(0), requires_grad=True)


def penalize_nonzero(weights, eagerness=1.0, acceptance=1.0):
    cliff = 1.0 / (np.e * eagerness)
    return torch.log(cliff + acceptance * torch.abs(weights)) - np.log(cliff)


def create_optimizer_from_config(optimizer_name, optimizer_args):
    """
    Translate the optimizer name and arguments into a torch optimizer.
    If an optimizer object is provided, it is returned as is.
    The optimizer is not initialized yet since this is done by the trainer.

    Parameters
        ----------
            optimizer_name : int
                Object provided to NeuralProphet as optimizer.
            optimizer_args : dict
                Arguments for the optimizer.

        Returns
        -------
            optimizer : torch.optim.Optimizer
                The optimizer object.
            optimizer_args : dict
                The optimizer arguments.
    """
    if type(optimizer_name) == str:
        if optimizer_name.lower() == "adamw":
            # Tends to overfit, but reliable
            optimizer = torch.optim.AdamW
            optimizer_args["weight_decay"] = 1e-3
        elif optimizer_name.lower() == "sgd":
            # better validation performance, but diverges sometimes
            optimizer = torch.optim.SGD
            optimizer_args["momentum"] = 0.9
            optimizer_args["weight_decay"] = 1e-4
        else:
            raise ValueError(f"The optimizer name {optimizer_name} is not supported.")
    elif inspect.isclass(optimizer_name) and issubclass(optimizer_name, torch.optim.Optimizer):
        optimizer = optimizer_name
    else:
        raise ValueError("The provided optimizer is not supported.")
    return optimizer, optimizer_args


def interprete_model(target_model: pl.LightningModule, net: str, forward_func: str):
    """
    Returns model input attributions for a given network and forward function.

    Parameters
    ----------
        target_model : pl.LightningModule
            The model for which input attributions are to be computed.

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
    num_in_features = getattr(target_model, net)[0].in_features
    # Number of output features from the net (aka n_forecasts)
    num_out_features = getattr(target_model, net)[-1].out_features
    num_out_features_without_quantiles = int(num_out_features / num_quantiles)

    # Create a tensor of ones as model input
    model_input = torch.ones(1, num_in_features, requires_grad=True)

    # Iterate through each output feature and compute the model attribution wrt. the input
    attributions = torch.empty((0, num_in_features))
    for output_feature in range(num_out_features_without_quantiles):
        for quantile in range(num_quantiles):
            target_attribution = saliency.attribute(model_input, target=[(output_feature, quantile)], abs=False)
            attributions = torch.cat((attributions, target_attribution), 0)

    # Average the attributions over the input features
    # Idea: Average attribution of each lag on all forecasts (eg. the n'th lag has an attribution of xyz on the forecast)
    # TODO: support the visualization of 2d tensors in plot_parameters (aka the attribution of the n'th lag on the m'th forecast)

    return attributions

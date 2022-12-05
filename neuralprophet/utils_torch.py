import inspect
import logging

import numpy as np
import torch

log = logging.getLogger("NP.utils_torch")


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

import torch
import torch.nn as nn


class BaseComponent(nn.Module):
    def __init__(self, n_forecasts, quantiles, id_list, bias, device):
        super().__init__()
        self.n_forecasts = n_forecasts
        self.quantiles = quantiles
        self.id_list = id_list
        self.bias = bias
        self.device = device

    def forward(self, x):
        """
        Needs to be implemented by subclass.
        """
        pass

    def new_param(self, dims):
        """Create and initialize a new torch Parameter.

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

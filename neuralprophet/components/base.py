from abc import abstractmethod

import torch.nn as nn


class BaseComponent(nn.Module):
    def __init__(self, n_forecasts, quantiles, id_list, device):
        super().__init__()
        self.n_forecasts = n_forecasts
        self.quantiles = quantiles
        self.id_list = id_list
        self.device = device

    @abstractmethod
    def forward(self, x):
        """
        Needs to be implemented by subclass.

        Parameters
        ----------
            t : torch.Tensor float
                normalized time, dim: (batch, n_forecasts)
            meta: dict
                Metadata about the all the samples of the model input batch. Contains the following:
                    * ``df_name`` (list, str), time series ID corresponding to each sample of the input batch.
        Returns
        -------
            torch.Tensor
                Component forecast, same dimensions as input t
        """
        pass

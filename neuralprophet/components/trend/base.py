from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from neuralprophet.components import BaseComponent


class BaseTrend(ABC, BaseComponent):
    def __init__(self, config, id_list, quantiles, num_trends_modelled, n_forecasts, device):
        super().__init__(n_forecasts=n_forecasts, quantiles=quantiles, id_list=id_list, device=device)
        self.config_trend = config
        self.num_trends_modelled = num_trends_modelled

        # if only 1 time series, global strategy
        if len(self.id_list) == 1:
            self.config_trend.trend_global_local = "global"

        # dimensions  - [no. of quantiles, 1 bias shape]
        self.bias = self.new_param(
            dims=[
                len(self.quantiles),
            ]
        )

    @abstractmethod
    def forward(self, t, meta):
        """Computes trend based on model configuration.

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
                Trend component, same dimensions as input t

        """
        pass

    @property
    @abstractmethod
    def get_trend_deltas(self):
        """trend deltas for regularization.

        update if trend is modelled differently"""
        pass

    @abstractmethod
    def add_regularization(self):
        """add regularization to loss"""
        pass

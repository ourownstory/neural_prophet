import torch
import torch.nn as nn

from neuralprophet.components.trend import Trend
from neuralprophet.utils_torch import init_parameter


class LinearTrend(Trend):
    def __init__(self, config, id_list, quantiles, num_trends_modelled, n_forecasts, device):
        super().__init__(
            config=config,
            n_forecasts=n_forecasts,
            num_trends_modelled=num_trends_modelled,
            quantiles=quantiles,
            id_list=id_list,
            device=device,
        )
        # Trend_k0  parameter.
        # dimensions - [no. of quantiles,  num_trends_modelled, trend coeff shape]
        self.trend_k0 = init_parameter(dims=([len(self.quantiles)] + [self.num_trends_modelled] + [1]))

    @property
    def get_trend_deltas(self):
        """trend deltas for regularization.

        update if trend is modelled differently"""
        if self.config_trend is None:
            trend_delta = None
        else:
            trend_delta = self.trend_deltas

        return trend_delta

    def add_regularization(self):
        pass


class GlobalLinearTrend(LinearTrend):
    def __init__(self, config, id_list, quantiles, num_trends_modelled, n_forecasts, device):
        super().__init__(
            config=config,
            n_forecasts=n_forecasts,
            num_trends_modelled=num_trends_modelled,
            quantiles=quantiles,
            id_list=id_list,
            device=device,
        )

    def forward(self, t, meta):
        """
        Computes trend based on model configuration.

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
        # dimensions -  batch_size, n_forecasts, quantiles
        trend = self.trend_k0.permute(1, 2, 0) * t.unsqueeze(dim=2)
        return self.bias.unsqueeze(dim=0).unsqueeze(dim=0) + trend


class LocalLinearTrend(LinearTrend):
    def __init__(self, config, id_list, quantiles, num_trends_modelled, n_forecasts, device):
        super().__init__(
            config=config,
            n_forecasts=n_forecasts,
            num_trends_modelled=num_trends_modelled,
            quantiles=quantiles,
            id_list=id_list,
            device=device,
        )

    def forward(self, t, meta):
        """
        Computes trend based on model configuration.

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
        # From the dataloader meta data, we get the one-hot encoding of the df_name.
        meta_name_tensor_one_hot = nn.functional.one_hot(meta, num_classes=len(self.id_list))
        # trend_k_0 = trend_k_0(sample metadata)
        # dimensions - batch_size, segments(1), quantiles
        trend_k_0 = torch.sum(
            meta_name_tensor_one_hot.unsqueeze(dim=0).unsqueeze(dim=-1) * self.trend_k0.unsqueeze(dim=1), dim=2
        ).permute(1, 2, 0)
        # dimensions -  batch_size, n_forecasts, quantiles
        trend = trend_k_0 * t.unsqueeze(2)
        return self.bias.unsqueeze(dim=0).unsqueeze(dim=0) + trend

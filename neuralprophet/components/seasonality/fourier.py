from abc import abstractmethod

import torch
import torch.nn as nn

from neuralprophet import utils
from neuralprophet.components.seasonality import Seasonality
from neuralprophet.utils_torch import init_parameter


class FourierSeasonality(Seasonality):
    def __init__(self, config, id_list, quantiles, num_seasonalities_modelled, n_forecasts, device):
        super().__init__(
            config=config,
            n_forecasts=n_forecasts,
            num_seasonalities_modelled=num_seasonalities_modelled,
            quantiles=quantiles,
            id_list=id_list,
            device=device,
        )
        self.season_dims = utils.config_seasonality_to_model_dims(self.config_seasonality)
        if self.season_dims is not None:
            # Seasonality parameters for global or local modelling
            self.season_params = nn.ParameterDict(
                {
                    # dimensions - [no. of quantiles, num_seasonalities_modelled, no. of fourier terms for each seasonality]
                    name: init_parameter(dims=[len(self.quantiles)] + [self.num_seasonalities_modelled] + [dim])
                    for name, dim in self.season_dims.items()
                }
            )

    @abstractmethod
    def compute_fourier(self, features, name, meta=None):
        """Compute single seasonality component.

        Parameters
        ----------
            features : torch.Tensor, float
                Features related to seasonality component, dims: (batch, n_forecasts, n_features)
            name : str
                Name of seasonality. for attribution to corresponding model weights.
            meta: dict
                Metadata about the all the samples of the model input batch. Contains the following:
                    * ``df_name`` (list, str), time series ID corresponding to each sample of the input batch.

        Returns
        -------
            torch.Tensor
                Forecast component of dims (batch, n_forecasts)
        """
        pass

    def forward(self, s, meta):
        """Compute all seasonality components.

        Parameters
        ----------
            s : torch.Tensor, float
                dict of named seasonalities (keys) with their features (values)
                dims of each dict value (batch, n_forecasts, n_features)
            meta: dict
                Metadata about the all the samples of the model input batch. Contains the following:
                    * ``df_name`` (list, str), time series ID corresponding to each sample of the input batch.

        Returns
        -------
            torch.Tensor
                Forecast component of dims (batch, n_forecasts)
        """
        device = s[list(s.keys())[0]].device
        x = torch.zeros(
            size=(s[list(s.keys())[0]].shape[0], self.n_forecasts, len(self.quantiles)),
            device=device,
        )
        for name, features in s.items():
            x = x + self.compute_fourier(features, name, meta)
        return x


class GlobalFourierSeasonality(FourierSeasonality):
    def __init__(self, config, id_list, quantiles, num_seasonalities_modelled, n_forecasts, device):
        super().__init__(
            config=config,
            n_forecasts=n_forecasts,
            num_seasonalities_modelled=num_seasonalities_modelled,
            quantiles=quantiles,
            id_list=id_list,
            device=device,
        )

    def compute_fourier(self, features, name, meta=None):
        """Compute single seasonality component.

        Parameters
        ----------
            features : torch.Tensor, float
                Features related to seasonality component, dims: (batch, n_forecasts, n_features)
            name : str
                Name of seasonality. for attribution to corresponding model weights.
            meta: dict
                Metadata about the all the samples of the model input batch. Contains the following:
                    * ``df_name`` (list, str), time series ID corresponding to each sample of the input batch.

        Returns
        -------
            torch.Tensor
                Forecast component of dims (batch, n_forecasts)
        """
        # dimensions -  batch_size, n_forecasts, quantiles
        seasonality = torch.sum(
            features.unsqueeze(dim=2) * self.season_params[name].permute(1, 0, 2).unsqueeze(dim=0), dim=-1
        )
        return seasonality


class LocalFourierSeasonality(FourierSeasonality):
    def __init__(self, config, id_list, quantiles, num_seasonalities_modelled, n_forecasts, device):
        super().__init__(
            config=config,
            n_forecasts=n_forecasts,
            num_seasonalities_modelled=num_seasonalities_modelled,
            quantiles=quantiles,
            id_list=id_list,
            device=device,
        )

    def compute_fourier(self, features, name, meta=None):
        """Compute single seasonality component.

        Parameters
        ----------
            features : torch.Tensor, float
                Features related to seasonality component, dims: (batch, n_forecasts, n_features)
            name : str
                Name of seasonality. for attribution to corresponding model weights.
            meta: dict
                Metadata about the all the samples of the model input batch. Contains the following:
                    * ``df_name`` (list, str), time series ID corresponding to each sample of the input batch.

        Returns
        -------
            torch.Tensor
                Forecast component of dims (batch, n_forecasts)
        """
        # From the dataloader meta data, we get the one-hot encoding of the df_name.
        meta_name_tensor_one_hot = nn.functional.one_hot(meta, num_classes=len(self.id_list))
        # dimensions - quantiles, batch, parameters_fourier
        season_params_sample = torch.sum(
            meta_name_tensor_one_hot.unsqueeze(dim=0).unsqueeze(dim=-1) * self.season_params[name].unsqueeze(dim=1),
            dim=2,
        )
        # dimensions -  batch_size, n_forecasts, quantiles
        seasonality = torch.sum(features.unsqueeze(2) * season_params_sample.permute(1, 0, 2).unsqueeze(1), dim=-1)
        return seasonality

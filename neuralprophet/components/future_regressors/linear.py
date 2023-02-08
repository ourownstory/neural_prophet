import torch
import torch.nn as nn

from neuralprophet.components.future_regressors import FutureRegressors
from neuralprophet.utils_torch import init_parameter

# from neuralprophet.utils_torch import init_parameter


class LinearFutureRegressors(FutureRegressors):
    def __init__(self, config, id_list, quantiles, n_forecasts, device, config_trend_none_bool):
        super().__init__(
            config=config,
            n_forecasts=n_forecasts,
            quantiles=quantiles,
            id_list=id_list,
            device=device,
            config_trend_none_bool=config_trend_none_bool,
        )
        if self.regressors_dims is not None:
            # Regresors params
            self.regressor_params = nn.ParameterDict(
                {
                    # dimensions - [no. of quantiles, no. of additive regressors]
                    "additive": init_parameter(dims=[len(self.quantiles), self.n_additive_regressor_params]),
                    # dimensions - [no. of quantiles, no. of multiplicative regressors]
                    "multiplicative": init_parameter(
                        dims=[len(self.quantiles), self.n_multiplicative_regressor_params]
                    ),
                }
            )

    def scalar_features_effects(self, features, params, indices=None):
        """
        Computes events component of the model

        Parameters
        ----------
            features : torch.Tensor, float
                Features (either additive or multiplicative) related to event component dims (batch, n_forecasts, n_features)
            params : nn.Parameter
                Params (either additive or multiplicative) related to events
            indices : list of int
                Indices in the feature tensors related to a particular event
        Returns
        -------
            torch.Tensor
                Forecast component of dims (batch, n_forecasts)
        """
        if indices is not None:
            features = features[:, :, indices]
            params = params[:, indices]

        return torch.sum(features.unsqueeze(dim=2) * params.unsqueeze(dim=0).unsqueeze(dim=0), dim=-1)

    def get_reg_weights(self, name):
        """
        Retrieve the weights of regressor features given the name

        Parameters
        ----------
            name : string
                Regressor name

        Returns
        -------
            torch.tensor
                Weight corresponding to the given regressor
        """

        regressor_dims = self.regressors_dims[name]
        mode = regressor_dims["mode"]
        index = regressor_dims["regressor_index"]

        if mode == "additive":
            regressor_params = self.regressor_params["additive"]
        else:
            assert mode == "multiplicative"
            regressor_params = self.regressor_params["multiplicative"]

        return regressor_params[:, index : (index + 1)]

    def forward(self, inputs, mode, indeces=None):
        """Compute all seasonality components.
        Parameters
        ----------
            f_r : torch.Tensor, float
                future regressors inputs
            mode: string, either "additive" or "multiplicative"
                mode of the regressors
        Returns
        -------
            torch.Tensor
                Forecast component of dims (batch, n_forecasts, no_quantiles)
        """

        if "additive" == mode:
            f_r = self.scalar_features_effects(inputs, self.regressor_params["additive"], indeces)
        if "multiplicative" == mode:
            f_r = self.scalar_features_effects(inputs, self.regressor_params["multiplicative"], indeces)
        return f_r

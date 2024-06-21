from collections import Counter

import torch.nn as nn

from neuralprophet.components.future_regressors import FutureRegressors
from neuralprophet.utils_torch import interprete_model

# from neuralprophet.utils_torch import init_parameter


class SharedNeuralNetsFutureRegressors(FutureRegressors):
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
            self.regressor_nets = nn.ModuleDict({})
            # TO DO: if no hidden layers, then just a as legacy
            self.d_hidden_regressors = config.d_hidden
            self.num_hidden_layers_regressors = config.num_hidden_layers
            # Combined network
            for net_i, size_i in Counter([x["mode"] for x in self.regressors_dims.values()]).items():
                # Nets for both additive and multiplicative regressors
                regressor_net = nn.ModuleList()
                # This will be later size_i(1 + static covariates)
                d_inputs = size_i
                for i in range(self.num_hidden_layers_regressors):
                    regressor_net.append(nn.Linear(d_inputs, self.d_hidden_regressors, bias=True))
                    d_inputs = self.d_hidden_regressors
                # final layer has input size d_inputs and output size equal to  no. of quantiles
                regressor_net.append(nn.Linear(d_inputs, len(self.quantiles), bias=False))
                for lay in regressor_net:
                    nn.init.kaiming_normal_(lay.weight, mode="fan_in")
                self.regressor_nets[net_i] = regressor_net

    def get_reg_weights(self, name):
        """
        Get attributions of regressors component network w.r.t. the model input.

        Parameters
        ----------
            name : string
                Regressor name

        Returns
        -------
            torch.tensor
                Weight corresponding to the given regressor
        """

        mode = self.config_regressors.regressors[name].mode
        reg_attributions = interprete_model(
            self,
            net="regressor_nets",
            forward_func="regressors_net",
            _num_in_features=self.regressor_nets[mode][0].in_features,
            _num_out_features=self.regressor_nets[mode][-1].out_features,
            additional_forward_args=mode,
        )

        regressor_index = self.regressors_dims[name]["regressor_index"]
        return reg_attributions[:, regressor_index].unsqueeze(-1)

    def regressors(self, regressor_inputs, mode):
        """Compute single regressor component.
        Parameters
        ----------
            regressor_input : torch.Tensor, float
                regressor values at corresponding, dims: (batch, n_forecasts, 1)
            nam : str
                Name of regressor, for attribution to corresponding model weights
        Returns
        -------
            torch.Tensor
                Forecast component of dims (batch, n_forecasts, num_quantiles)
        """
        x = regressor_inputs
        for i in range(self.num_hidden_layers_regressors + 1):
            if i > 0:
                x = nn.functional.relu(x)
            x = self.regressor_nets[mode][i](x)

        # segment the last dimension to match the quantiles
        # x = x.reshape(x.shape[0], self.n_forecasts, len(self.quantiles)) # causes error in case of multiple forecast targes, possibly unneeded/wrong
        return x

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
        return self.regressors(inputs, mode)

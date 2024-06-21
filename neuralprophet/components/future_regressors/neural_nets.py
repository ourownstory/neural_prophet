from collections import OrderedDict

import torch.nn as nn

from neuralprophet.components.future_regressors import FutureRegressors
from neuralprophet.utils_torch import interprete_model

# from neuralprophet.utils_torch import init_parameter


class NeuralNetsFutureRegressors(FutureRegressors):
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
            # one net per regressor. to be adapted to combined network
            for regressor in self.regressors_dims.keys():
                # Nets for both additive and multiplicative regressors
                regressor_net = nn.ModuleList()
                # This will be later 1 + static covariates
                d_inputs = 1
                for i in range(self.num_hidden_layers_regressors):
                    regressor_net.append(nn.Linear(d_inputs, self.d_hidden_regressors, bias=True))
                    d_inputs = self.d_hidden_regressors
                # final layer has input size d_inputs and output size equal to no. of quantiles
                regressor_net.append(nn.Linear(d_inputs, len(self.quantiles), bias=False))
                for lay in regressor_net:
                    nn.init.kaiming_normal_(lay.weight, mode="fan_in")
                self.regressor_nets[regressor] = regressor_net

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

        reg_attributions = interprete_model(
            self,
            net="regressor_nets",
            forward_func="regressor",
            _num_in_features=self.regressor_nets[name][0].in_features,
            _num_out_features=self.regressor_nets[name][-1].out_features,
            additional_forward_args=name,
        )

        return reg_attributions

    def regressor(self, regressor_input, name):
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
        x = regressor_input
        for i in range(self.num_hidden_layers_regressors + 1):
            if i > 0:
                x = nn.functional.relu(x)
            x = self.regressor_nets[name][i](x)

        return x

    def all_regressors(self, regressor_inputs, mode):
        """Compute all regressors components.
        Parameters
        ----------
            regressor_inputs : torch.Tensor, float
                regressor values at corresponding, dims: (batch, n_forecasts, num_regressors)
        Returns
        -------
            torch.Tensor
                Forecast component of dims (batch, n_forecasts, num_quantiles)
        """
        # Select only elements from OrderedDict that have the value mode == 'mode_of_interest'
        regressors_dims_filtered = OrderedDict((k, v) for k, v in self.regressors_dims.items() if v["mode"] == mode)
        for i, name in enumerate(regressors_dims_filtered.keys()):
            regressor_index = regressors_dims_filtered[name]["regressor_index"]
            regressor_input = regressor_inputs[:, :, regressor_index].unsqueeze(dim=2)
            if i == 0:
                x = self.regressor(regressor_input, name=name)
            if i > 0:
                x = x + self.regressor(regressor_input, name=name)
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
        return self.all_regressors(inputs, mode)

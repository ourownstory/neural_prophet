from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import logging
from neuralprophet.utils import (
    season_config_to_model_dims,
    regressors_config_to_model_dims,
    events_config_to_model_dims,
)

log = logging.getLogger("NP.time_net")


def new_param(dims):
    """Create and initialize a new torch Parameter.

    Args:
        dims (list or tuple): desired dimensions of parameter

    Returns:
        initialized Parameter
    """
    if len(dims) > 1:
        return nn.Parameter(nn.init.xavier_normal_(torch.randn(dims)), requires_grad=True)
    else:
        return nn.Parameter(torch.nn.init.xavier_normal_(torch.randn([1] + dims)).squeeze(0), requires_grad=True)


class TimeNet(nn.Module):
    """Linear time regression fun and some not so linear fun.

    A modular model that models classic time-series components
    - trend
    - seasonality
    - auto-regression (as AR-Net)
    - covariates (as AR-Net)
    - apriori regressors
    - events and holidays
    by using Neural Network components.
    The Auto-regression and covariate components can be configured as a deeper network (AR-Net).
    """

    def __init__(
        self,
        config_trend=None,
        config_season=None,
        config_covar=None,
        config_regressors=None,
        config_events=None,
        config_holidays=None,
        n_forecasts=1,
        n_lags=0,
        num_hidden_layers=0,
        d_hidden=None,
        n_quantiles=None,
        quantiles=None,
    ):
        """
        Args:
            config_trend (configure.Trend):
            config_season (configure.Season):
            config_covar (OrderedDict):
            config_regressors (OrderedDict): Configs of regressors with mode and index.
            config_events (OrderedDict):
            config_holidays (OrderedDict):
            n_forecasts (int): number of steps to forecast. Aka number of model outputs.
            n_lags (int): number of previous steps of time series used as input. Aka AR-order.
                0 (default): no auto-regression
            num_hidden_layers (int): number of hidden layers (for AR-Net)
                0 (default): no hidden layers, corresponds to classic Auto-Regression
            d_hidden (int): dimensionality of hidden layers  (for AR-Net). ignored if no hidden layers.
                None (default): sets to n_lags + n_forecasts
            n_quantiles (int): the number of quantiles estimated
            quantiles (list): the set of quantiles estimated
        """
        super(TimeNet, self).__init__()
        # General
        self.n_forecasts = n_forecasts

        # Quantiles
        self.n_quantiles = n_quantiles
        # TODO: For quantile forecasts
        # self.quantile_params = new_param(dims=[n_quantiles - 1])
        self.quantiles = quantiles

        # Bias
        self.bias = new_param(dims=[self.n_quantiles, 1])

        # Trend
        self.config_trend = config_trend
        if self.config_trend.growth in ["linear", "discontinuous"]:
            self.segmentwise_trend = self.config_trend.trend_reg == 0
            self.trend_k0 = new_param(dims=[self.n_quantiles, 1])
            if self.config_trend.n_changepoints > 0:
                if self.config_trend.changepoints is None:
                    # create equidistant changepoint times, including zero.
                    linear_t = np.arange(self.config_trend.n_changepoints + 1).astype(float)
                    linear_t = linear_t / (self.config_trend.n_changepoints + 1)
                    self.config_trend.changepoints = self.config_trend.changepoints_range * linear_t
                else:
                    self.config_trend.changepoints = np.insert(self.config_trend.changepoints, 0, 0.0)
                self.trend_changepoints_t = torch.tensor(
                    self.config_trend.changepoints, requires_grad=False, dtype=torch.float
                )
                self.trend_deltas = new_param(
                    dims=[self.n_quantiles, (self.config_trend.n_changepoints + 1)]
                )  # including first segment
                if self.config_trend.growth == "discontinuous":
                    self.trend_m = new_param(
                        dims=[self.n_quantiles, (self.config_trend.n_changepoints + 1)]
                    )  # including first segment

        # Seasonalities
        self.config_season = config_season
        self.season_dims = season_config_to_model_dims(self.config_season)
        if self.season_dims is not None:
            if self.config_season.mode == "multiplicative" and self.config_trend is None:
                log.error("Multiplicative seasonality requires trend.")
                raise ValueError
            if self.config_season.mode not in ["additive", "multiplicative"]:
                log.error(
                    "Seasonality Mode {} not implemented. Defaulting to 'additive'.".format(self.config_season.mode)
                )
                self.config_season.mode = "additive"
            self.season_params = nn.ParameterDict(
                {name: new_param(dims=[self.n_quantiles, dim]) for name, dim in self.season_dims.items()}
            )
            # self.season_params_vec = torch.cat([self.season_params[name] for name in self.season_params.keys()])

        # Events
        self.config_events = config_events
        self.events_dims = events_config_to_model_dims(config_events, config_holidays)
        if self.events_dims is not None:
            n_additive_event_params = 0
            n_multiplicative_event_params = 0
            for event, configs in self.events_dims.items():
                if configs["mode"] not in ["additive", "multiplicative"]:
                    log.error("Event Mode {} not implemented. Defaulting to 'additive'.".format(configs["mode"]))
                    self.events_dims[event]["mode"] = "additive"
                if configs["mode"] == "additive":
                    n_additive_event_params += len(configs["event_indices"])
                elif configs["mode"] == "multiplicative":
                    if self.config_trend is None:
                        log.error("Multiplicative events require trend.")
                        raise ValueError
                    n_multiplicative_event_params += len(configs["event_indices"])
            self.event_params = nn.ParameterDict(
                {
                    "additive": new_param(dims=[self.n_quantiles, n_additive_event_params]),
                    "multiplicative": new_param(dims=[self.n_quantiles, n_multiplicative_event_params]),
                }
            )
        else:
            self.config_events = None

        # Autoregression
        self.n_lags = n_lags
        self.num_hidden_layers = num_hidden_layers
        self.d_hidden = n_lags + n_forecasts if d_hidden is None else d_hidden
        if self.n_lags > 0:
            self.ar_net = nn.ModuleList()
            d_inputs = self.n_lags
            for i in range(self.num_hidden_layers):
                self.ar_net.append(nn.Linear(d_inputs, self.d_hidden, bias=True))
                d_inputs = self.d_hidden
            self.ar_net.append(nn.Linear(d_inputs, self.n_quantiles * self.n_forecasts, bias=False))
            for lay in self.ar_net:
                nn.init.xavier_normal_(lay.weight)

        # Covariates
        self.config_covar = config_covar
        if self.config_covar is not None:
            assert self.n_lags > 0
            self.covar_nets = nn.ModuleDict({})
            for covar in self.config_covar.keys():
                covar_net = nn.ModuleList()
                d_inputs = self.n_lags
                if self.config_covar[covar].as_scalar:
                    d_inputs = 1
                for i in range(self.num_hidden_layers):
                    covar_net.append(nn.Linear(d_inputs, self.d_hidden, bias=True))
                    d_inputs = self.d_hidden
                covar_net.append(nn.Linear(d_inputs, self.n_quantiles * self.n_forecasts, bias=False))
                for lay in covar_net:
                    nn.init.kaiming_normal_(lay.weight, mode="fan_in")
                self.covar_nets[covar] = covar_net

        ## Regressors
        self.config_regressors = config_regressors
        self.regressors_dims = regressors_config_to_model_dims(config_regressors)
        if self.regressors_dims is not None:
            n_additive_regressor_params = 0
            n_multiplicative_regressor_params = 0
            for name, configs in self.regressors_dims.items():
                if configs["mode"] not in ["additive", "multiplicative"]:
                    log.error("Regressors mode {} not implemented. Defaulting to 'additive'.".format(configs["mode"]))
                    self.regressors_dims[name]["mode"] = "additive"
                if configs["mode"] == "additive":
                    n_additive_regressor_params += 1
                elif configs["mode"] == "multiplicative":
                    if self.config_trend is None:
                        log.error("Multiplicative regressors require trend.")
                        raise ValueError
                    n_multiplicative_regressor_params += 1

            self.regressor_params = nn.ParameterDict(
                {
                    "additive": new_param(dims=[self.n_quantiles, n_additive_regressor_params]),
                    "multiplicative": new_param(dims=[self.n_quantiles, n_multiplicative_regressor_params]),
                }
            )
        else:
            self.config_regressors = None

    def get_trend_deltas(self, quantile=None):
        """trend deltas for regularization.

        update if trend is modelled differently"""
        if self.config_trend is None or self.config_trend.n_changepoints < 1:
            trend_delta = None
        elif self.segmentwise_trend:
            trend_delta = self.trend_deltas - torch.cat((self.trend_k0, self.trend_deltas[:, :-1]), dim=1)
        else:
            trend_delta = self.trend_deltas

        if quantile is not None and trend_delta is not None:
            quantile_index = self.quantiles.index(quantile)
            return trend_delta[quantile_index, :]
        else:
            return trend_delta

    @property
    def ar_weights(self):
        """sets property auto-regression weights for regularization. Update if AR is modelled differently"""
        return self.ar_net[0].weight

    def get_covar_weights(self, name):
        """sets property auto-regression weights for regularization. Update if AR is modelled differently"""
        return self.covar_nets[name][0].weight

    def get_event_weights(self, name, quantile=None):
        """
        Retrieve the weights of event features given the name

        Args:
            name (string): Event name
            quantile (float): the quantile for which the event weights are requested

        Returns:
            event_param_dict (OrderedDict): Dict of the weights of all offsets corresponding
            to a particular event.
        """

        event_dims = self.events_dims[name]
        mode = event_dims["mode"]

        if mode == "multiplicative":
            event_params = self.event_params["multiplicative"]
        else:
            assert mode == "additive"
            event_params = self.event_params["additive"]

        event_param_dict = OrderedDict({})
        for event_delim, indices in zip(event_dims["event_delim"], event_dims["event_indices"]):
            if quantile is None:
                event_param_dict[event_delim] = event_params[:, indices : (indices + 1)]
            else:
                quantile_index = self.quantiles.index(quantile)
                event_param_dict[event_delim] = event_params[quantile_index, indices : (indices + 1)]
        return event_param_dict

    def get_reg_weights(self, name, quantile=None):
        """
        Retrieve the weights of regressor features given the name

        Args:
            name (string): Regressor name
            quantile (float): the quantile for which the reg weights are requested

        Returns:
            weight (torch.tensor): Weight corresponding to the given regressor
        """

        regressor_dims = self.regressors_dims[name]
        mode = regressor_dims["mode"]
        index = regressor_dims["regressor_index"]

        if mode == "additive":
            regressor_params = self.regressor_params["additive"]
        else:
            assert mode == "multiplicative"
            regressor_params = self.regressor_params["multiplicative"]

        if quantile is not None:
            quantile_index = self.quantiles.index(quantile)
            return regressor_params[quantile_index, index : (index + 1)]
        else:
            return regressor_params[:, index : (index + 1)]

    def _compute_quantile_forecasts_from_diffs(self, diffs):
        """
        Computes the actual quantile forecasts from quantile differences estimated from the model
        Consrtaints the differences to be positive by using absolute value as an activation fn

        Args:
            diffs (torch.tensor): tensor of dims (batch, n_quantiles, n_forecasts) which
                contains the median quantile forecasts as well as the diffs of other quantiles
                from the median quantile

        Returns:
            final forecasts of dim (batch, n_quantiles, n_forecasts)
        """

        # generate the actual quantile forecasts from predicted differences
        median_quantile_index = self.quantiles.index(0.5)
        upper_quantiles = self.quantiles[median_quantile_index + 1 :]
        lower_quantiles = self.quantiles[:median_quantile_index]

        epsilon = 1e-10
        upper_quantiles_forecasts = list()
        last_upper_quantile_forecast = torch.squeeze(diffs[:, median_quantile_index, :])
        for i, _ in enumerate(upper_quantiles):
            quantile_forecast = last_upper_quantile_forecast + (
                torch.abs(diffs[:, (median_quantile_index + i + 1), :]) + epsilon
            )
            upper_quantiles_forecasts.append(quantile_forecast)
            last_upper_quantile_forecast = quantile_forecast

        lower_quantiles_forecasts = list()
        last_lower_quantile_forecast = torch.squeeze(diffs[:, median_quantile_index, :])
        for i, _ in enumerate(lower_quantiles):
            quantile_forecast = last_lower_quantile_forecast - (
                torch.abs(diffs[:, (median_quantile_index - i - 1), :]) + epsilon
            )
            lower_quantiles_forecasts.append(quantile_forecast)
            last_lower_quantile_forecast = quantile_forecast

        lower_quantiles_forecasts.reverse()  # reverse the lower quantiles to get the sorted order
        lower_quantiles_forecasts.append(
            torch.squeeze(diffs[:, median_quantile_index, :], dim=1)
        )  # add back the median quantile

        # merge lower and upper quantiles
        out = torch.stack((lower_quantiles_forecasts + upper_quantiles_forecasts), dim=1)
        return out

    def _piecewise_linear_trend(self, t):
        """Piecewise linear trend, computed segmentwise or with deltas.

        Args:
            t (torch tensor, float): normalized time of
                dimensions (batch, n_forecasts)

        Returns:
            Trend component, same dimensions as input t
        """
        past_next_changepoint = t.unsqueeze(2) >= torch.unsqueeze(self.trend_changepoints_t[1:], dim=0)
        segment_id = torch.sum(past_next_changepoint, dim=2)
        current_segment = nn.functional.one_hot(segment_id, num_classes=self.config_trend.n_changepoints + 1)

        k_t = torch.sum(torch.unsqueeze(current_segment, dim=1) * torch.unsqueeze(self.trend_deltas, dim=1), dim=3)

        if not self.segmentwise_trend:
            previous_deltas_t = torch.sum(
                torch.unsqueeze(past_next_changepoint, dim=1)
                * self.trend_deltas[:, -1].reshape(self.n_quantiles, 1, 1),
                dim=3,
            )
            k_t = k_t + previous_deltas_t

        if self.config_trend.growth != "discontinuous":
            if self.segmentwise_trend:
                deltas = self.trend_deltas - torch.cat((self.trend_k0, self.trend_deltas[:, 0:-1]), dim=1)
            else:
                deltas = self.trend_deltas
            gammas = -self.trend_changepoints_t[1:] * deltas[:, 1:]
            m_t = torch.sum(torch.unsqueeze(past_next_changepoint, dim=1) * torch.unsqueeze(gammas, dim=1), dim=3)
            if not self.segmentwise_trend:
                m_t = m_t.detach()
        else:
            m_t = torch.sum(torch.unsqueeze(current_segment, dim=1) * torch.unsqueeze(self.trend_m, dim=1), dim=3)

        return (self.trend_k0 + k_t) * torch.unsqueeze(t, dim=1) + m_t

    def trend(self, t, quantile=None):
        """Computes trend based on model configuration.

        Args:
            t (torch tensor float): normalized time
                dimensions (batch, n_forecasts)
            quantile(float): the quantile the trend is requested for

        Returns:
            Trend component, same dimensions as input t

        """
        if self.config_trend.growth == "off":
            trend = torch.zeros(size=(t.shape[0], self.n_quantiles, self.n_forecasts))
        elif int(self.config_trend.n_changepoints) == 0:
            trend = self.trend_k0 * torch.unsqueeze(t, dim=1)
        else:
            trend = self._piecewise_linear_trend(t)
        if quantile is None:
            return self.bias + trend
        else:
            quantile_index = self.quantiles.index(quantile)
            return self.bias[quantile_index, :] + trend[:, quantile_index, :]

    def seasonality(self, features, name, quantile=None):
        """Compute single seasonality component.

        Args:
            features (torch tensor, float): features related to seasonality component
                dims: (batch, n_forecasts, n_features)
            name (str): name of seasonality. for attributiun to corresponding model weights.
            quantile(float): the quantile the seasonality is requested for
        Returns:
            forecast component of dims (batch, n_forecasts)
        """
        seasonality = torch.sum(
            torch.unsqueeze(features, dim=1) * torch.unsqueeze(self.season_params[name], dim=1), dim=3
        )
        if quantile is None:
            return seasonality
        else:
            index = self.quantiles.index(quantile)
            return seasonality[:, index, :]

    def all_seasonalities(self, s):
        """Compute all seasonality components.

        Args:
            s (dict(torch tensor, float)): dict of named seasonalities (keys) with their features (values)
                dims of each dict value: (batch, n_forecasts, n_features)

        Returns:
            forecast component of dims (batch, n_forecasts)
        """
        x = torch.zeros(size=(s[list(s.keys())[0]].shape[0], self.n_quantiles, self.n_forecasts))
        for name, features in s.items():
            x = x + self.seasonality(features, name)
        return x

    def scalar_features_effects(self, features, params, indices=None):
        """
        Computes events component of the model
        Args:
            features (torch tensor, float): features (either additive or multiplicative) related to event component
                dims: (batch, n_forecasts, n_features)
            params (nn.Parameter): params (either additive or multiplicative) related to events
            indices (list of int): indices in the feature tensors related to a particular event
        Returns:
            forecast component of dims (batch, n_forecasts)
        """
        if indices is not None:
            features = features[:, :, indices]
            params = params[:, indices]

        return torch.sum(torch.unsqueeze(features, dim=1) * torch.unsqueeze(params, dim=1), dim=3)

    def auto_regression(self, lags):
        """Computes auto-regessive model component AR-Net.

        Args:
            lags (torch tensor, float): previous times series values.
                dims: (batch, n_lags)

        Returns:
            forecast component of dims: (batch, n_forecasts)
        """
        x = lags
        for i in range(self.num_hidden_layers + 1):
            if i > 0:
                x = nn.functional.relu(x)
            x = self.ar_net[i](x)

        # segment the last dimension to match the quantiles
        x = x.reshape(x.shape[0], self.n_quantiles, self.n_forecasts)
        return x

    def covariate(self, lags, name):
        """Compute single covariate component.

        Args:
            lags (torch tensor, float): lagged values of covariate
                dims: (batch, n_lags)
            name (str): name of covariate. for attributiun to corresponding model weights.

        Returns:
            forecast component of dims (batch, n_forecasts)
        """
        x = lags
        for i in range(self.num_hidden_layers + 1):
            if i > 0:
                x = nn.functional.relu(x)
            x = self.covar_nets[name][i](x)

        # segment the last dimension to match the quantiles
        x = x.reshape(x.shape[0], self.n_quantiles, self.n_forecasts)
        return x

    def all_covariates(self, covariates):
        """Compute all covariate components.

        Args:
            covariates (dict(torch tensor, float)): dict of named covariates (keys) with their features (values)
                dims of each dict value: (batch, n_lags)

        Returns:
            forecast component of dims (batch, n_forecasts)
        """
        for i, name in enumerate(covariates.keys()):
            if i == 0:
                x = self.covariate(lags=covariates[name], name=name)
            if i > 0:
                x = x + self.covariate(lags=covariates[name], name=name)
        return x

    def forward(self, inputs):
        """This method defines the model forward pass.

        Time input is required. Minimum model setup is a linear trend.
        Args:
            inputs (dict):
                time (torch tensor float): normalized time
                    dims: (batch, n_forecasts)
                lags (torch tensor, float): previous times series values.
                    dims: (batch, n_lags)
                seasonalities (dict(torch tensor, float)): dict of named seasonalities (keys) with their features (values)
                    dims of each dict value: (batch, n_forecasts, n_features)
                covariates (dict(torch tensor, float)): dict of named covariates (keys) with their features (values)
                    dims of each dict value: (batch, n_lags)
                events (torch tensor, float): all event features
                    dims: (batch, n_forecasts, n_features)
                regressors (torch tensor, float): all regressor features
                    dims: (batch, n_forecasts, n_features)
        Returns:
            forecast of dims (batch, n_quantiles, n_forecasts)
        """
        additive_components = torch.zeros(size=(inputs["time"].shape[0], self.n_quantiles, self.n_forecasts))
        multiplicative_components = torch.zeros(size=(inputs["time"].shape[0], self.n_quantiles, self.n_forecasts))

        if "lags" in inputs:
            additive_components += self.auto_regression(lags=inputs["lags"])
        # else: assert self.n_lags == 0

        if "covariates" in inputs:
            additive_components += self.all_covariates(covariates=inputs["covariates"])

        if "seasonalities" in inputs:
            s = self.all_seasonalities(s=inputs["seasonalities"])
            if self.config_season.mode == "additive":
                additive_components += s
            elif self.config_season.mode == "multiplicative":
                multiplicative_components += s

        if "events" in inputs:
            if "additive" in inputs["events"].keys():
                additive_components += self.scalar_features_effects(
                    inputs["events"]["additive"], self.event_params["additive"]
                )
            if "multiplicative" in inputs["events"].keys():
                multiplicative_components += self.scalar_features_effects(
                    inputs["events"]["multiplicative"], self.event_params["multiplicative"]
                )

        if "regressors" in inputs:
            if "additive" in inputs["regressors"].keys():
                additive_components += self.scalar_features_effects(
                    inputs["regressors"]["additive"], self.regressor_params["additive"]
                )
            if "multiplicative" in inputs["regressors"].keys():
                multiplicative_components += self.scalar_features_effects(
                    inputs["regressors"]["multiplicative"], self.regressor_params["multiplicative"]
                )

        trend = self.trend(t=inputs["time"])
        out = trend + additive_components + trend * multiplicative_components

        if self.quantiles is not None:
            out = self._compute_quantile_forecasts_from_diffs(out)
        else:
            out = torch.squeeze(out, dim=1)
        return out

    def compute_components(self, inputs):
        """This method returns the values of each model component.

        Time input is required. Minimum model setup is a linear trend.
        Args:
            inputs (dict):
                time (torch tensor float): normalized time
                    dims: (batch, n_forecasts)
                lags (torch tensor, float): previous times series values.
                    dims: (batch, n_lags)
                seasonalities (dict(torch tensor, float)): dict of named seasonalities (keys) with their features (values)
                    dims of each dict value: (batch, n_forecasts, n_features)
                covariates (dict(torch tensor, float)): dict of named covariates (keys) with their features (values)
                    dims of each dict value: (batch, n_lags)
                events (torch tensor, float): all event features
                    dims: (batch, n_forecasts, n_features)
        Returns:
            dict of forecast_component: value
                with elements of dims (batch, n_forecasts)
        """
        components = {}
        components["trend"] = self.trend(t=inputs["time"])
        if self.config_trend is not None and "seasonalities" in inputs:
            for name, features in inputs["seasonalities"].items():
                components["season_{}".format(name)] = self.seasonality(features=features, name=name)
        if self.n_lags > 0 and "lags" in inputs:
            components["ar"] = self.auto_regression(lags=inputs["lags"])
        if self.config_covar is not None and "covariates" in inputs:
            for name, lags in inputs["covariates"].items():
                components["lagged_regressor_{}".format(name)] = self.covariate(lags=lags, name=name)
        if self.config_events is not None and "events" in inputs:
            if "additive" in inputs["events"].keys():
                components["events_additive"] = self.scalar_features_effects(
                    features=inputs["events"]["additive"], params=self.event_params["additive"]
                )
            if "multiplicative" in inputs["events"].keys():
                components["events_multiplicative"] = self.scalar_features_effects(
                    features=inputs["events"]["multiplicative"], params=self.event_params["multiplicative"]
                )
            for event, configs in self.events_dims.items():
                mode = configs["mode"]
                indices = configs["event_indices"]
                if mode == "additive":
                    features = inputs["events"]["additive"]
                    params = self.event_params["additive"]
                else:
                    features = inputs["events"]["multiplicative"]
                    params = self.event_params["multiplicative"]
                components["event_{}".format(event)] = self.scalar_features_effects(
                    features=features, params=params, indices=indices
                )
        if self.config_regressors is not None and "regressors" in inputs:
            if "additive" in inputs["regressors"].keys():
                components["future_regressors_additive"] = self.scalar_features_effects(
                    features=inputs["regressors"]["additive"], params=self.regressor_params["additive"]
                )
            if "multiplicative" in inputs["regressors"].keys():
                components["future_regressors_multiplicative"] = self.scalar_features_effects(
                    features=inputs["regressors"]["multiplicative"], params=self.regressor_params["multiplicative"]
                )
            for regressor, configs in self.regressors_dims.items():
                mode = configs["mode"]
                index = []
                index.append(configs["regressor_index"])
                if mode == "additive":
                    features = inputs["regressors"]["additive"]
                    params = self.regressor_params["additive"]
                else:
                    features = inputs["regressors"]["multiplicative"]
                    params = self.regressor_params["multiplicative"]
                components["future_regressor_{}".format(regressor)] = self.scalar_features_effects(
                    features=features, params=params, indices=index
                )
        return components


class FlatNet(nn.Module):
    """
    Linear regression fun
    """

    def __init__(self, d_inputs, d_outputs):
        # Perform initialization of the pytorch superclass
        super(FlatNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_inputs, d_outputs),
        )
        nn.init.kaiming_normal_(self.layers[0].weight, mode="fan_in")

    def forward(self, x):
        return self.layers(x)

    @property
    def ar_weights(self):
        return self.model.layers[0].weight


class DeepNet(nn.Module):
    """
    A simple, general purpose, fully connected network
    """

    def __init__(self, d_inputs, d_outputs, d_hidden=32, num_hidden_layers=0):
        # Perform initialization of the pytorch superclass
        super(DeepNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(d_inputs, d_hidden, bias=True))
            d_inputs = d_hidden
        self.layers.append(nn.Linear(d_inputs, d_outputs, bias=True))
        for lay in self.layers:
            nn.init.kaiming_normal_(lay.weight, mode="fan_in")

    def forward(self, x):
        """
        This method defines the network layering and activation functions
        """
        activation = nn.functional.relu
        for i in range(len(self.layers)):
            if i > 0:
                x = activation(x)
            x = self.layers[i](x)
        return x

    @property
    def ar_weights(self):
        return self.layers[0].weight

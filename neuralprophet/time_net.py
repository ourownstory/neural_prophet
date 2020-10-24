from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import logging

log = logging.getLogger("nprophet.time_net")


def new_param(dims):
    """Create and initialize a new torch Parameter.

    Args:
        dims (list or tuple): desired dimensions of parameter

    Returns:
        initialized Parameter
    """
    if len(dims) > 1:
        return nn.Parameter(nn.init.xavier_normal_(
            torch.randn(dims)),
            requires_grad=True)
    else:
        return nn.Parameter(torch.nn.init.xavier_normal_(
            torch.randn([1]+dims)).squeeze(0),
            requires_grad=True)


class TimeNet(nn.Module):
    """Linear regression fun and some more fun.

    A modular model that models classic time-series components
    - trend
    - auto-regression (AR-Net)
    - seasonality
    by using Neural Network components.
    The Auto-regression component can be configured to to be a deeper network (AR-Net).
    """
    def __init__(self,
                 n_forecasts,
                 n_lags=0,
                 n_changepoints=0,
                 trend_smoothness=0,
                 num_hidden_layers=0,
                 d_hidden=None,
                 season_dims=None,
                 season_mode='additive',
                 covar_config=None,
                 events_dims=None,
                 regressors_dims=None,
                 ):
        """
        Args:
            n_forecasts (int): number of steps to forecast. Aka number of model outputs.
            n_lags (int): number of previous steps of time series used as input. Aka AR-order.
                0 (default): no auto-regression
            n_changepoints (int): number of trend changepoints.
                0 (default): no changepoints
            trend_smoothness (int/float): how much to regularize the trend changepoints
                0 (default): segmentwise trend with continuity (individual k for each segment)
                -1: discontinuous segmentwise trend (individual k, m for each segment)
            num_hidden_layers (int): number of hidden layers (for AR-Net)
                0 (default): no hidden layers, corresponds to classic Auto-Regression
            d_hidden (int): dimensionality of hidden layers  (for AR-Net). ignored if no hidden layers.
                None (default): sets to n_lags + n_forecasts
            season_dims (OrderedDict(int)): ordered Dict with entries: <seasonality name>: vector dimension
                None (default): No seasonality
            season_mode (str): 'additive', 'multiplicative', how seasonality term is accounted for in forecast.
                'additive' (default): add seasonality component to outputs of other model components
            covar_config (OrderedDict): Names of covariate variables.
            events_dims (pd.DataFrame): Dataframe with columns 'event' and 'event_delim'
            regressors_dims (OrderedDict): Configs of regressors with mode and index.
        """
        super(TimeNet, self).__init__()
        ## General
        self.n_forecasts = n_forecasts

        ## Bias
        self.forecast_bias = new_param(dims=[self.n_forecasts])

        ## Trend
        self.n_changepoints = n_changepoints
        self.continuous_trend = True
        self.segmentwise_trend = True
        if trend_smoothness < 0:
            self.continuous_trend = False
        elif trend_smoothness > 0:
            # compute trend delta-wise to allow for stable regularization.
            # has issues with gradient bleedover to past.
            self.segmentwise_trend = False
        # changepoint times, including zero.
        linear_t = np.arange(self.n_changepoints + 1).astype(float) / (self.n_changepoints + 1)
        self.trend_changepoints_t = torch.tensor(linear_t, requires_grad=False, dtype=torch.float)
        self.trend_k0 = new_param(dims=[1])
        self.trend_m0 = new_param(dims=[1])
        if self.n_changepoints > 0:
            self.trend_deltas = new_param(dims=[self.n_changepoints + 1]) # including first segment
            if not self.continuous_trend:
                self.trend_m = new_param(dims=[self.n_changepoints + 1]) # including first segment

        ## Seasonalities
        self.season_dims = season_dims
        self.season_mode = season_mode
        if self.season_dims is not None:
            if self.season_mode not in ['additive', 'multiplicative']:
                raise NotImplementedError("Seasonality Mode {} not implemented".format(self.season_mode))
            self.season_params = nn.ParameterDict({
                name: new_param(dims=[dim]) for name, dim in self.season_dims.items()
            })
            # self.season_params_vec = torch.cat([self.season_params[name] for name in self.season_params.keys()])

        ## Events
        self.events_dims = events_dims

        if self.events_dims is not None:
            self.event_params = nn.ParameterDict({})
            n_additive_event_params = 0
            n_multiplicative_event_params = 0
            for event, configs in self.events_dims.items():
                if configs["mode"] == "additive":
                    n_additive_event_params += len(configs['event_indices'])
                else:
                    n_multiplicative_event_params += len(configs['event_indices'])

            self.event_params["additive"] = new_param(dims=[n_additive_event_params])
            self.event_params["multiplicative"] = new_param(dims=[n_multiplicative_event_params])
        else:
            self.event_params = None

        ## Autoregression
        self.n_lags = n_lags
        self.num_hidden_layers = num_hidden_layers
        self.d_hidden = n_lags + n_forecasts if d_hidden is None else d_hidden
        if self.n_lags > 0:
            self.ar_net = nn.ModuleList()
            d_inputs = self.n_lags
            for i in range(self.num_hidden_layers):
                self.ar_net.append(nn.Linear(d_inputs, self.d_hidden, bias=True))
                d_inputs = self.d_hidden
            self.ar_net.append(nn.Linear(d_inputs, self.n_forecasts, bias=False))
            for lay in self.ar_net:
                nn.init.kaiming_normal_(lay.weight, mode='fan_in')

        ## Covariates
        if covar_config is not None:
            assert self.n_lags > 0
            self.covar_nets = nn.ModuleDict({})
            for covar in covar_config.keys():
                covar_net = nn.ModuleList()
                d_inputs = self.n_lags
                if covar_config[covar].as_scalar:
                    d_inputs = 1
                for i in range(self.num_hidden_layers):
                    covar_net.append(nn.Linear(d_inputs, self.d_hidden, bias=True))
                    d_inputs = self.d_hidden
                covar_net.append(nn.Linear(d_inputs, self.n_forecasts, bias=False))
                for lay in covar_net:
                    nn.init.kaiming_normal_(lay.weight, mode='fan_in')
                self.covar_nets[covar] = covar_net

        ## Regressors
        self.regressors_dims = regressors_dims
        if self.regressors_dims is not None:
            self.regressor_params = nn.ParameterDict({})
            n_additive_regressor_params = 0
            n_multiplicative_regressor_params = 0
            for configs in self.regressors_dims.values():
                if configs["mode"] == "additive":
                    n_additive_regressor_params += 1
                else:
                    n_multiplicative_regressor_params += 1

            self.regressor_params["additive"] = new_param(dims=[n_additive_regressor_params])
            self.regressor_params["multiplicative"] = new_param(dims=[n_multiplicative_regressor_params])
        else:
            self.regressor_params = None


    @property
    def get_trend_deltas(self):
        """trend deltas for regularization.

        update if trend is modelled differently"""
        if self.n_changepoints < 1:
            return None
        elif self.segmentwise_trend:
            return torch.cat((self.trend_k0, self.trend_deltas[:-1])) - self.trend_deltas
        else:
            return self.trend_deltas

    @property
    def ar_weights(self):
        """sets property auto-regression weights for regularization. Update if AR is modelled differently"""
        return self.ar_net[0].weight

    def get_covar_weights(self, name):
        """sets property auto-regression weights for regularization. Update if AR is modelled differently"""
        return self.covar_nets[name][0].weight

    def get_event_weights(self, name):
        """
        Retrieve the weights of event features given the name

        Args:
            name (string): Event name

        Returns:
            event_param_dict (OrderedDict): Dict of the weights of all offsets corresponding
            to a particular event.
        """

        event_dims = self.events_dims[name]
        mode = event_dims["mode"]

        if mode == "multiplicative":
            event_params = self.event_params["multiplicative"]
        else:
            assert (mode == "additive")
            event_params = self.event_params["additive"]

        event_param_dict = OrderedDict({})
        for event_delim, indices in zip(event_dims["event_delim"], event_dims["event_indices"]):
            event_param_dict[event_delim] = event_params[indices]
        return event_param_dict

    def get_reg_weights(self, name):
        """
        Retrieve the weights of regressor features given the name

        Args:
            name (string): Regressor name

        Returns:
            weight (torch.tensor): Weight corresponding to the given regressor
        """

        regressor_dims = self.regressors_dims[name]
        mode = regressor_dims["mode"]
        index = regressor_dims["regressor_index"]

        if mode == "additive":
            regressor_params = self.regressor_params["additive"]
        if mode == "multiplicative":
            regressor_params = self.regressor_params["multiplicative"]

        return regressor_params[index]

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
        current_segment = nn.functional.one_hot(segment_id, num_classes=self.n_changepoints + 1)

        k_t = torch.sum(current_segment * torch.unsqueeze(self.trend_deltas, dim=0), dim=2)

        if not self.segmentwise_trend:
            previous_deltas_t = torch.sum(past_next_changepoint * torch.unsqueeze(self.trend_deltas[:-1], dim=0), dim=2)
            k_t = k_t + previous_deltas_t

        if self.continuous_trend:
            if self.segmentwise_trend:
                deltas = self.trend_deltas[:] - torch.cat((self.trend_k0, self.trend_deltas[0:-1]))
            else:
                deltas = self.trend_deltas
            gammas = -self.trend_changepoints_t[1:] * deltas[1:]
            m_t = torch.sum(past_next_changepoint * gammas, dim=2)
            if not self.segmentwise_trend:
                m_t = m_t.detach()
        else:
            m_t = torch.sum(current_segment * torch.unsqueeze(self.trend_m, dim=0), dim=2)

        return (self.trend_k0 + k_t) * t + (self.trend_m0 + m_t)

    def trend(self, t):
        """Computes trend based on model configuration.

        Args:
            t (torch tensor float): normalized time
                dimensions (batch, n_forecasts)

        Returns:
            Trend component, same dimensions as input t

        """
        if int(self.n_changepoints) == 0:
            return self.trend_k0 * t + self.trend_m0
        else:
            return self._piecewise_linear_trend(t)

    def seasonality(self, features, name):
        """Compute single seasonality component.

        Args:
            features (torch tensor, float): features related to seasonality component
                dims: (batch, n_forecasts, n_features)
            name (str): name of seasonality. for attributiun to corresponding model weights.

        Returns:
            forecast component of dims (batch, n_forecasts)
        """
        return torch.sum(features * torch.unsqueeze(self.season_params[name], dim=0), dim=2)

    def all_seasonalities(self, s):
        """Compute all seasonality components.

        Args:
            s (dict(torch tensor, float)): dict of named seasonalities (keys) with their features (values)
                dims of each dict value: (batch, n_forecasts, n_features)

        Returns:
            forecast component of dims (batch, n_forecasts)
        """
        x = torch.zeros(s[list(s.keys())[0]].shape[:2])
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
            features = features[:,:,indices]
            params = params[indices]

        return torch.sum(features * torch.unsqueeze(params, dim=0), dim=2)

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
            if i > 0: x = nn.functional.relu(x)
            x = self.ar_net[i](x)
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
            if i > 0: x = nn.functional.relu(x)
            x = self.covar_nets[name][i](x)
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
            forecast of dims (batch, n_forecasts)
        """
        trend = self.trend(t=inputs['time'])

        additive_components = torch.zeros_like(trend)
        multiplicative_components = torch.zeros_like(trend)

        if "lags" in inputs:
            additive_components += self.auto_regression(lags=inputs['lags'])
        # else: assert self.n_lags == 0

        if 'covariates' in inputs:
            additive_components += self.all_covariates(covariates=inputs['covariates'])

        if 'seasonalities' in inputs:
            s = self.all_seasonalities(s=inputs['seasonalities'])
            if self.season_mode == 'additive':
                additive_components += s
            elif self.season_mode == 'multiplicative':
                multiplicative_components += s

        if 'events' in inputs:
            if "additive" in inputs["events"].keys():
                additive_components += self.scalar_features_effects(
                    inputs["events"]["additive"], self.event_params["additive"])
            if "multiplicative" in inputs["events"].keys():
                multiplicative_components += self.scalar_features_effects(
                    inputs["events"]["multiplicative"], self.event_params["multiplicative"])

        if 'regressors' in inputs:
            if "additive" in inputs["regressors"].keys():
                additive_components += self.scalar_features_effects(
                    inputs["regressors"]["additive"], self.regressor_params["additive"])
            if "multiplicative" in inputs["regressors"].keys():
                multiplicative_components += self.scalar_features_effects(
                    inputs["regressors"]["multiplicative"], self.regressor_params["multiplicative"])

        out = trend + trend * multiplicative_components + additive_components

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
        components = {
            'trend': self.trend(t=inputs['time']),
        }
        if 'seasonalities' in inputs:
            for name, features in inputs['seasonalities'].items():
                components['season_{}'.format(name)] = self.seasonality(features=features, name=name)
        if "lags" in inputs:
            assert self.n_lags >= 1
            components['ar'] = self.auto_regression(lags=inputs['lags'])
        if "covariates" in inputs:
            for name, lags in inputs['covariates'].items():
                components['lagged_regressor_{}'.format(name)] = self.covariate(lags=lags, name=name)
        if "events" in inputs:
            if 'additive' in inputs["events"].keys():
                components['events_additive'] = self.scalar_features_effects(features=inputs["events"]["additive"],
                                                               params=self.event_params["additive"])
            if 'multiplicative' in inputs["events"].keys():
                components['events_multiplicative'] = self.scalar_features_effects(features=inputs["events"]["multiplicative"],
                                                                     params=self.event_params["multiplicative"])
            for event, configs in self.events_dims.items():
                mode = configs["mode"]
                indices = configs["event_indices"]
                if mode == "additive":
                    features = inputs["events"]["additive"]
                    params = self.event_params["additive"]
                else:
                    features = inputs["events"]["multiplicative"]
                    params = self.event_params["multiplicative"]
                components['event_{}'.format(event)] = self.scalar_features_effects(features=features, params=params, indices=indices)
        if "regressors" in inputs:
            if 'additive' in inputs["regressors"].keys():
                components['future_regressors_additive'] = self.scalar_features_effects(
                    features=inputs["regressors"]["additive"],
                    params=self.regressor_params["additive"])
            if 'multiplicative' in inputs["regressors"].keys():
                components['future_regressors_multiplicative'] = self.scalar_features_effects(
                    features=inputs["regressors"]["multiplicative"],
                    params=self.regressor_params["multiplicative"])
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
                components['future_regressor_{}'.format(regressor)] = self.scalar_features_effects(
                    features=features, params=params, indices=index
                )
        return components


class FlatNet(nn.Module):
    '''
    Linear regression fun
    '''

    def __init__(self, d_inputs, d_outputs):
        # Perform initialization of the pytorch superclass
        super(FlatNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_inputs, d_outputs),
        )
        nn.init.kaiming_normal_(self.layers[0].weight, mode='fan_in')

    def forward(self, x):
        return self.layers(x)

    @property
    def ar_weights(self):
        return self.model.layers[0].weight


class DeepNet(nn.Module):
    '''
    A simple, general purpose, fully connected network
    '''
    def __init__(self, d_inputs, d_outputs, d_hidden=32, num_hidden_layers=0):
        # Perform initialization of the pytorch superclass
        super(DeepNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(d_inputs, d_hidden, bias=True))
            d_inputs = d_hidden
        self.layers.append(nn.Linear(d_inputs, d_outputs, bias=True))
        for lay in self.layers:
            nn.init.kaiming_normal_(lay.weight, mode='fan_in')

    def forward(self, x):
        '''
        This method defines the network layering and activation functions
        '''
        activation = nn.functional.relu
        for i in range(len(self.layers)):
            if i > 0:
                x = activation(x)
            x = self.layers[i](x)
        return x

    @property
    def ar_weights(self):
        return self.layers[0].weight

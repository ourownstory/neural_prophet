from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn


def new_param(dims):
    """Create and initialize a new torch Parameter.

    Args:
        dims (list or tuple): desired dimensions of parameter

    Returns:
        initialized Parameter
    """
    # return nn.Parameter(nn.init.kaiming_normal_(torch.randn(dims), mode='fan_out'), requires_grad=True)
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
        """
        super(TimeNet, self).__init__()
        ## General
        self.n_forecasts = n_forecasts

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
            self.season_params = nn.ParameterDict({})
            for name, dim in self.season_dims.items():
                self.season_params[name] = new_param(dims=[dim])

        ## Autoregression
        self.n_lags = n_lags
        self.num_hidden_layers = num_hidden_layers
        self.d_hidden = n_lags + n_forecasts if d_hidden is None else d_hidden
        if self.n_lags > 0:
            self.ar_net = nn.ModuleList()
            d_inputs = self.n_lags
            for i in range(self.num_hidden_layers):
                self.ar_net.append(nn.Linear(d_inputs, self.d_hidden, bias=True))
                d_inputs = d_hidden
            self.ar_net.append(nn.Linear(d_inputs, self.n_forecasts, bias=True))
            for lay in self.ar_net:
                nn.init.kaiming_normal_(lay.weight, mode='fan_in')

    @property
    def get_trend_deltas(self):
        """sets property trend deltas for regularization. update if trend is modelled differently"""
        if self.segmentwise_trend:
            return torch.cat((self.trend_k0, self.trend_deltas[:-1])) - self.trend_deltas
        else:
            return self.trend_deltas

    @property
    def ar_weights(self):
        """sets property auto-regression weights for regularization. Update if AR is modelled differently"""
        return self.ar_net[0].weight

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
            ## TODO: Why do the deltas explode when we stop the gradient?
            ## Why needed: if we do not, the gradient is shared to past deltas, fails to learn past deltas well
            # previous_deltas_t = previous_deltas_t.data # explodes
            # previous_deltas_t = previous_deltas_t.detach() # explodes
            # previous_deltas_t = previous_deltas_t.detach().requires_grad_(False)  # explodes
            # previous_deltas_t = previous_deltas_t.clone().detach()  # explodes
            # previous_deltas_t = previous_deltas_t.clone().detach().requires_grad_(False)  # explodes
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

    def auto_regression(self, lags):
        """Computes auto-regessive model component AR-Net.

        Args:
            lags (torch tensor, float): previous times series values.
                dims: (batch, n_lags)

        Returns:
            forecast component of dims: (batch, n_forecasts)
        """
        # if self.num_hidden_layers == 0:
        #     return self.ar_net(lags)
        # else:
        activation = nn.functional.relu
        x = lags
        for i in range(len(self.ar_net)):
            if i > 0: x = activation(x)
            x = self.ar_net[i](x)
        return x

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

    def forward(self, time, lags=None, seasonalities=None):
        """This method defines the model forward pass.

        Time input is required. Minimum model setup is a linear trend.
        Lags and
        Args:
            time (torch tensor float): normalized time
                dims: (batch, n_forecasts)
            lags (torch tensor, float): previous times series values.
                dims: (batch, n_lags)
            seasonalities (dict(torch tensor, float)): dict of named seasonalities (keys) with their features (values)
                dims of each dict value: (batch, n_forecasts, n_features)

        Returns:
            forecast of dims (batch, n_forecasts)
        """
        out = self.trend(t=time)

        if lags is not None:
            # assert self.n_lags >= 1
            out += self.auto_regression(lags=lags)
        # else: assert self.n_lags == 0

        if seasonalities is not None:
            # assert self.season_dims is not None
            s = self.all_seasonalities(s=seasonalities)
            if self.season_mode == 'additive': out = out + s
            elif self.season_mode == 'multiplicative': out = out * s
        # else: assert self.season_dims is None
        return out


class FlatNet(nn.Module):
    '''
    Linear regression fun
    '''

    def __init__(self, d_inputs, d_outputs):
        # Perform initialization of the pytorch superclass
        super(FlatNet, self).__init__()
        # if self.num_hidden_layers == 0:
        #     self.ar_net = nn.Linear(n_lags, n_forecasts, bias=False)
        #     nn.init.kaiming_normal_(self.ar_net.weight, mode='fan_in')
        # else:
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
            if i > 0: x = activation(x)
            x = self.layers[i](x)
        return x

    @property
    def ar_weights(self):
        return self.layers[0].weight

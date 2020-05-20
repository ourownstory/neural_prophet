import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def new_param_1d(dims):
    return nn.Parameter(torch.nn.init.xavier_normal_(
        torch.randn([1]+dims)).squeeze(0),
        requires_grad=True)

def new_param(dims):
    # return nn.Parameter(nn.init.kaiming_normal_(torch.randn(dims), mode='fan_out'), requires_grad=True)
    return nn.Parameter(nn.init.xavier_normal_(
        torch.randn(dims)),
        requires_grad=True)


class TimeNet(nn.Module):
    '''
    Linear regression
    '''

    def __init__(self, n_forecasts, n_lags=0, n_changepoints=0):
        # Perform initialization of the pytorch superclass
        super(TimeNet, self).__init__()
        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self.n_changepoints = n_changepoints

        ## model definition
        # trend
        self.trend_k = new_param_1d(dims=[1])
        self.trend_m = new_param_1d(dims=[1])
        if self.n_changepoints > 0:
            self.trend_deltas = new_param(dims=[1, self.n_changepoints])
            linear_t = np.arange(self.n_changepoints+1).astype(float) / (self.n_changepoints+1)
            self.trend_changepoints = torch.tensor(linear_t[1:], requires_grad=False, dtype=torch.float).unsqueeze(0)
        # autoregression
        if self.n_lags > 0:
            self.ar = nn.Linear(n_lags, n_forecasts, bias=False)
            nn.init.kaiming_normal_(self.ar.weight, mode='fan_in')

    def piecewise_linear_trend(self, t):
        # note: t is referring to the time at forecast-target.
        # broadcast trend rate and offset
        out = self.trend_k * t + self.trend_m
        if self.n_changepoints > 0:
            past_changepoint = t.unsqueeze(2) > self.trend_changepoints
            k_t = torch.sum(past_changepoint * self.trend_deltas, dim=2)
            # # Intercept changes
            gammas = -self.trend_changepoints * self.trend_deltas
            m_t = torch.sum(past_changepoint * gammas, dim=2)
            # add delta changes to trend impact
            out = out + k_t * t + m_t
        return out

    def auto_regression(self, lags):
        return self.ar(lags)

    @property
    def ar_weights(self):
        return self.ar.weight

    def forward(self, time, lags=None):
        out = self.piecewise_linear_trend(t=time)
        if self.n_lags >= 1:
            out += self.auto_regression(lags=lags)
        return out


class FlatNet(nn.Module):
    '''
    Linear regression
    '''

    def __init__(self, d_inputs, d_outputs):
        # Perform initialization of the pytorch superclass
        super(FlatNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_inputs, d_outputs),
        )

    def forward(self, x):
        return self.layers(x)


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

    def forward(self, x):
        '''
        This method defines the network layering and activation functions
        '''
        activation = F.relu
        for i in range(len(self.layers)):
            if i > 0: x = activation(x)
            x = self.layers[i](x)
        return x

    @property
    def ar_weights(self):
        return self.model.layers[0].weight

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from attrdict import AttrDict


def new_param(dims):
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
    '''
    Linear regression
    '''

    def __init__(self, n_forecasts, n_lags=0, n_changepoints=0, continuous_trend=True):
        # Perform initialization of the pytorch superclass
        super(TimeNet, self).__init__()
        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self.n_changepoints = n_changepoints
        self.continuous_trend = continuous_trend

        ## model definition
        # trend
        # if self.prophet_trend:
        #     self.trend_k = new_param(dims=[1])
        #     self.trend_m = new_param(dims=[1])
        #     if self.n_changepoints > 0:
        #         self.trend_deltas = new_param(dims=[1, self.n_changepoints])
        #         linear_t = np.arange(self.n_changepoints+1).astype(float) / (self.n_changepoints+1)
        #         self.trend_changepoints = torch.tensor(linear_t[1:], requires_grad=False, dtype=torch.float).unsqueeze(0)
        # else:
        self.trend_k = new_param(dims=[self.n_changepoints + 1])
        if self.continuous_trend:
            self.trend_m = new_param(dims=[1])
        else:
            self.trend_m = new_param(dims=[self.n_changepoints + 1])
        linear_t = np.arange(self.n_changepoints + 1).astype(float) / (self.n_changepoints + 1)
        self.trend_changepoints_t = torch.tensor(linear_t, requires_grad=False, dtype=torch.float).unsqueeze(0)

        # autoregression
        if self.n_lags > 0:
            self.ar = nn.Linear(n_lags, n_forecasts, bias=False)
            nn.init.kaiming_normal_(self.ar.weight, mode='fan_in')


    def _prophet_trend(self, t):
        # note: t is referring to the time at forecast-target.
        # broadcast trend rate and offset
        # this has issues, as gradients from more recent segments bleed over to old trend parameters.
        # better use _piecewise_linear_trend
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

    def _piecewise_linear_trend(self, t):
        # This solves the issue of gradients being over-shared with past trend segments.
        # However, it does not ensure continuity of the trend.
        # TODO: add option to make continuous again.
        # note: t is referring to the time at forecast-target.

        past_changepoint = t.unsqueeze(2) >= torch.unsqueeze(self.trend_changepoints_t, dim=0)
        segment_id = torch.sum(past_changepoint, dim=2) - 1
        current_segment = F.one_hot(segment_id, num_classes=self.n_changepoints+1)

        k_t = torch.sum(current_segment * torch.unsqueeze(self.trend_k, dim=0), dim=2)
        if self.continuous_trend:
            ks = self.trend_k.clone().detach().requires_grad_(False)
            deltas = ks[1:] - ks[0:-1]
            deltas = torch.cat((torch.zeros(1, requires_grad=False), deltas))
            gammas = -self.trend_changepoints_t * deltas
            m_t = torch.sum(past_changepoint * gammas, dim=2)
            m_t = self.trend_m + m_t
        else:
            m_t = torch.sum(current_segment * torch.unsqueeze(self.trend_m, dim=0), dim=2)

        out = k_t * t + m_t
        return out

    def trend(self, t):
        # if self.prophet_trend:
        #     return self._prophet_trend(t)
        # else:
        return self._piecewise_linear_trend(t)

    def auto_regression(self, lags):
        return self.ar(lags)

    @property
    def trend_params(self):
        print("WARNING: deprecated, might contain bug.")
        changepoints_t = self.trend_changepoints_t.detach().numpy()
        k = self.trend_k.detach().numpy()
        m = self.trend_m.detach().numpy()
        if self.continuous_trend and self.n_changepoints > 0:
            past_changepoint = np.tril(np.ones((self.n_changepoints+1,self.n_changepoints+1)))
            deltas = k[1:] - k[0:-1]
            deltas = np.append(np.zeros(1), deltas)
            gammas = - changepoints_t * deltas
            m_t = np.sum(past_changepoint * np.expand_dims(gammas, 0), axis=1)
            m = m + np.squeeze(m_t)
        changepoints_t = np.squeeze(changepoints_t)
        k = np.squeeze(k)
        m = np.squeeze(m)
        return AttrDict({"k": k, "m": m, "changepoints_t": changepoints_t})

    @property
    def trend_deltas(self):
        deltas = self.trend_k[1:] - self.trend_k[0:-1]
        return deltas

    @property
    def ar_weights(self):
        return self.ar.weight

    def forward(self, time, lags=None):
        out = self.trend(t=time)
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

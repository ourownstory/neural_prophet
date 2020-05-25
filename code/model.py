import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


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

    def __init__(self, n_forecasts, n_lags=0, n_changepoints=0, trend_smoothness=0):
        # Perform initialization of the pytorch superclass
        super(TimeNet, self).__init__()
        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self.n_changepoints = n_changepoints

        self.continuous_trend = True
        self.segmentwise_trend = True
        if trend_smoothness < 0:
            self.continuous_trend = False
        elif trend_smoothness > 0:
            # compute trend delta-wise to allow for stable regularization.
            # has issues with gradient bleedover to past.
            self.segmentwise_trend = False

        ## model definition
        ## trend
        self.trend_k0 = new_param(dims=[1])
        self.trend_m0 = new_param(dims=[1])
        if self.n_changepoints > 0:
            # including first segment
            self.trend_deltas = new_param(dims=[self.n_changepoints + 1])
            if not self.continuous_trend:
                # including first segment
                self.trend_m = new_param(dims=[self.n_changepoints + 1])

        linear_t = np.arange(self.n_changepoints + 1).astype(float) / (self.n_changepoints + 1)
        # changepoint times, including zero.
        self.trend_changepoints_t = torch.tensor(linear_t, requires_grad=False, dtype=torch.float)

        # autoregression
        if self.n_lags > 0:
            self.ar = nn.Linear(n_lags, n_forecasts, bias=False)
            nn.init.kaiming_normal_(self.ar.weight, mode='fan_in')


    def _deltawise_trend_prophet(self, t):
        # note: t is referring to the time at forecast-target.
        # broadcast trend rate and offset
        # this has issues, as gradients from more recent segments bleed over to old trend parameters.
        # better use _segmentwise_linear_trend
        out = self.trend_k0 * t + self.trend_m0
        if self.n_changepoints > 0:
            past_changepoint = t.unsqueeze(2) >= torch.unsqueeze(self.trend_changepoints_t, dim=0)

            k_t = torch.sum(past_changepoint * self.trend_deltas, dim=2)
            # # Intercept changes
            gammas = -self.trend_changepoints_t * self.trend_deltas
            m_t = torch.sum(past_changepoint * gammas, dim=2)
            # add delta changes to trend impact
            out = out + k_t * t + m_t
        return out

    def _piecewise_linear_trend(self, t):
        past_next_changepoint = t.unsqueeze(2) >= torch.unsqueeze(self.trend_changepoints_t[1:], dim=0)
        segment_id = torch.sum(past_next_changepoint, dim=2)
        current_segment = F.one_hot(segment_id, num_classes=self.n_changepoints+1)

        k_t = torch.sum(current_segment * torch.unsqueeze(self.trend_deltas, dim=0), dim=2)

        if not self.segmentwise_trend:
            previous_deltas_t = torch.sum(past_next_changepoint * torch.unsqueeze(self.trend_deltas[:-1], dim=0), dim=2)
            # TODO: Why do the deltas explode when we stop the gradient?
            ## Why needed: if we do not, the gradient is shared to past deltas,
            # fails to learn past deltas well
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

    @property
    def get_trend_deltas(self):
        if self.segmentwise_trend:
            return self.trend_deltas
        else:
            return self.trend_deltas

    @property
    def ar_weights(self):
        return self.ar.weight

    def trend(self, t):
        if int(self.n_changepoints) == 0:
            return self.trend_k0 * t + self.trend_m0
        else:
            return self._piecewise_linear_trend(t)

    def auto_regression(self, lags):
        return self.ar(lags)

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

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
            self.trend_deltas = new_param(dims=[self.n_changepoints + 1])
            if not self.continuous_trend:
                self.trend_m = new_param(dims=[self.n_changepoints + 1])

        linear_t = np.arange(self.n_changepoints + 1).astype(float) / (self.n_changepoints + 1)
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


    def _deltawise_trend(self, t):
        # note: t is referring to the time at forecast-target.
        # broadcast trend rate and offset
        # this has issues, as gradients from more recent segments bleed over to old trend parameters.
        # better use _segmentwise_linear_trend

        # Note: assumes that there is a delta for each segment
        # there are n_changepoints +1 deltas, the first trend_changepoints_t is 0.0

        segment_id = torch.sum(past_next_changepoint, dim=2)
        current_segment = F.one_hot(segment_id, num_classes=self.n_changepoints + 1)

        delta_t = torch.sum(current_segment * torch.unsqueeze(self.trend_deltas, dim=0), dim=2)
        previous_deltas_t =  torch.sum(past_next_changepoint * torch.unsqueeze(self.trend_deltas[:-1], dim=0), dim=2)
        # TODO: Why do the deltas explode when we stop the gradient?
        ## Why needed: if we do not, the gradient is shared to past deltas,
        # fails to learn past deltas well
        # previous_deltas_t = previous_deltas_t.data # explodes
        # previous_deltas_t = previous_deltas_t.detach() # explodes
        # previous_deltas_t = previous_deltas_t.detach().requires_grad_(False)  # explodes
        # previous_deltas_t = previous_deltas_t.clone().detach()  # explodes
        # previous_deltas_t = previous_deltas_t.clone().detach().requires_grad_(False)  # explodes
        k_t = previous_deltas_t + delta_t

        if self.continuous_trend:
            # # Intercept changes
            gammas = -self.trend_changepoints_t[1:] * self.trend_deltas[1:]
            m_t = torch.sum(past_next_changepoint * gammas, dim=2)
            m_t = m_t.detach()
            # add delta changes to trend impact
        else:
            m_t = torch.sum(current_segment * torch.unsqueeze(self.trend_m, dim=0), dim=2)

        return (self.trend_k0 + k_t) * t + (self.trend_m0 + m_t)


    def _segmentwise_trend(self, t):
        # This solves the issue of gradients being over-shared with past trend segments.
        # note: t is referring to the time at forecast-target.
        past_changepoint = t.unsqueeze(2) >= torch.unsqueeze(self.trend_changepoints_t, dim=0)
        segment_id = torch.sum(past_changepoint, dim=2) - 1
        current_segment = F.one_hot(segment_id, num_classes=self.n_changepoints+1)

        k_t = torch.sum(current_segment * torch.unsqueeze(self.trend_deltas, dim=0), dim=2)

        if self.continuous_trend:
            # ks = self.trend_k.detach()
            # deltas = ks[1:] - ks[0:-1]
            # deltas = torch.cat((torch.zeros(1, requires_grad=False), deltas))
            deltas = self.trend_deltas[:] - torch.cat((self.trend_k0, self.trend_deltas[0:-1]))
            deltas = deltas.detach()
            gammas = -self.trend_changepoints_t * deltas
            m_t = torch.sum(past_changepoint * gammas, dim=2)
        else:
            m_t = torch.sum(current_segment * torch.unsqueeze(self.trend_m, dim=0), dim=2)

        return (self.trend_k0 + k_t) * t + (self.trend_m0 + m_t)

    # def _piecewise_linear_trend_with_k0(self, t): # if using implementation with k0
    #     # This solves the issue of gradients being over-shared with past trend segments.
    #     # note: t is referring to the time at forecast-target.
    #
    #     past_changepoint = t.unsqueeze(2) >= torch.unsqueeze(self.trend_changepoints_t, dim=0)
    #     segment_id = torch.sum(past_changepoint, dim=2) - 1
    #     current_segment = F.one_hot(segment_id, num_classes=self.n_changepoints+1)
    #
    #     trend_deltas = self.trend_deltas
    #     # print("trend_deltas", self.trend_deltas.shape)
    #     # print(self.trend_deltas)
    #     previous_deltas = torch.cumsum(trend_deltas, dim=0)
    #     previous_deltas = torch.cat((torch.zeros(1, requires_grad=False), previous_deltas[:-1]))
    #     # print("previous_deltas", previous_deltas.shape)
    #     # print(previous_deltas)
    #
    #     trend_k = self.trend_k0 + previous_deltas + self.trend_deltas
    #     # print("trend_k0", trend_k.shape)
    #     # print(trend_k)
    #
    #     k_t = torch.sum(current_segment * torch.unsqueeze(trend_k, dim=0), dim=2)
    #     # print(t[0], segment_id[0])
    #     # print(self.trend_changepoints_t)
    #     # print(past_changepoint[0])
    #     # print(current_segment[0])
    #     # print(k_t[0])
    #     if self.continuous_trend:
    #         deltas = self.trend_deltas.clone().detach().requires_grad_(False)
    #         # deltas = torch.cat((torch.zeros(1, requires_grad=False), deltas))
    #         gammas = -self.trend_changepoints_t * deltas
    #         m_t = torch.sum(past_changepoint * gammas, dim=2)
    #         m_t = self.trend_m0 + m_t
    #     else:
    #         m_t = torch.sum(current_segment * torch.unsqueeze(self.trend_m, dim=0), dim=2)
    #
    #     out = k_t * t + m_t
    #     return out

    # @property
    # def trend_k(self):
    #     """combines base trend with trend deltas to the segmentwise k values"""
    #     if self.use_k0 and not self.prophet_trend:
    #         def previous_deltas_no_gradient(trend_deltas):
    #             trend_deltas = trend_deltas.detach()
    #             # print("trend_deltas", self.trend_deltas.shape)
    #             # print(self.trend_deltas)
    #             previous_deltas = torch.cumsum(trend_deltas, dim=0)
    #             previous_deltas = torch.cat((torch.zeros(1, requires_grad=False), previous_deltas[:-1]))
    #             # print("previous_deltas", previous_deltas.shape)
    #             # print(previous_deltas)
    #             return previous_deltas
    #         trend_k = torch.cat((self.trend_k0, self.trend_k0 + previous_deltas_no_gradient(self.trend_deltas) + self.trend_deltas), dim=0)
    #         print("trend_k0", trend_k.shape)
    #         print(trend_k)
    #     else:
    #         raise NotImplementedError
    #     return trend_k

    # @property
    # def trend_params(self):
    #     print("WARNING: deprecated, might contain bug.")
    #     changepoints_t = self.trend_changepoints_t.detach().numpy()
    #     k = self.trend_k0.detach().numpy()
    #     if self.continuous_trend:
    #         m = self.trend_m.detach().numpy()
    #         if self.n_changepoints > 0:
    #             past_changepoint = np.tril(np.ones((self.n_changepoints + 1, self.n_changepoints + 1)))
    #             deltas = k[1:] - k[0:-1]
    #             deltas = np.append(np.zeros(1), deltas)
    #             gammas = - changepoints_t * deltas
    #             m_t = np.sum(past_changepoint * np.expand_dims(gammas, 0), axis=1)
    #             m = m + np.squeeze(m_t)
    #     else:
    #         m = self.trend_m0.detach().numpy()
    #     changepoints_t = np.squeeze(changepoints_t)
    #     k = np.squeeze(k)
    #     m = np.squeeze(m)
    #     return AttrDict({"k": k, "m": m, "changepoints_t": changepoints_t})

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

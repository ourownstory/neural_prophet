import numpy as np
import torch
import torch.nn as nn

from neuralprophet.components.trend import Trend
from neuralprophet.utils_torch import init_parameter


class PiecewiseLinearTrend(Trend):
    def __init__(self, config, id_list, quantiles, num_trends_modelled, n_forecasts, device):
        super().__init__(
            config=config,
            n_forecasts=n_forecasts,
            num_trends_modelled=num_trends_modelled,
            quantiles=quantiles,
            id_list=id_list,
            device=device,
        )

        self.segmentwise_trend = self.config_trend.trend_reg == 0

        # Trend_k0  parameter.
        # dimensions - [no. of quantiles,  num_trends_modelled, trend coeff shape]
        self.trend_k0 = init_parameter(dims=([len(self.quantiles)] + [self.num_trends_modelled] + [1]))

        # Configure the changepoints
        if self.config_trend.changepoints is None:
            # create equidistant changepoint times, including zero.
            linear_t = np.arange(self.config_trend.n_changepoints + 1).astype(float)
            linear_t = linear_t / (self.config_trend.n_changepoints + 1)
            self.config_trend.changepoints = self.config_trend.changepoints_range * linear_t
        else:
            self.config_trend.changepoints = np.insert(self.config_trend.changepoints, 0, 0.0)
        # Register in buffer so the tensor is moved to the correct device once initialized,
        # https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html#remove-any-cuda-or-to-device-calls
        self.register_buffer(
            "trend_changepoints_t",
            torch.tensor(self.config_trend.changepoints, requires_grad=False, dtype=torch.float),
        )

        # Trend Deltas parameters
        self.trend_deltas = init_parameter(
            dims=([len(self.quantiles)] + [self.num_trends_modelled] + [self.config_trend.n_changepoints + 1])
        )  # including first segment

        # When discontinuous, the start of the segment is not defined by the previous segments.
        # This brings a new set of parameters to optimize.
        if self.config_trend.growth == "discontinuous":
            self.trend_m = init_parameter(
                dims=([len(self.quantiles)] + [self.num_trends_modelled] + [self.config_trend.n_changepoints + 1])
            )  # including first segment

    def forward(self, t, meta):
        """Computes trend based on model configuration.

        Parameters
        ----------
            t : torch.Tensor float
                normalized time, dim: (batch, n_forecasts)
            meta: dict
                Metadata about the all the samples of the model input batch. Contains the following:
                    * ``df_name`` (list, str), time series ID corresponding to each sample of the input batch.
        Returns
        -------
            torch.Tensor
                Trend component, same dimensions as input t

        """
        # From the dataloader meta data, we get the one-hot encoding of the df_name.
        if self.config_trend.trend_global_local == "local":
            # dimensions - batch , num_time_series
            meta_name_tensor_one_hot = nn.functional.one_hot(meta, num_classes=len(self.id_list))
        else:
            meta_name_tensor_one_hot = None

        # Variables identifying, for t, the corresponding trend segment (for each sample of the batch).
        past_next_changepoint = t.unsqueeze(dim=2) >= self.trend_changepoints_t[1:].unsqueeze(dim=0)
        segment_id = past_next_changepoint.sum(dim=2)
        # = dimensions - batch_size, n_forecasts, segments (+ 1)
        current_segment = nn.functional.one_hot(segment_id, num_classes=self.config_trend.n_changepoints + 1)

        # Computing k_t.
        # dimensions - batch_size, n_forecasts, quantiles_size
        k_t = self.compute_k_t(current_segment, past_next_changepoint, meta_name_tensor_one_hot)

        # Computing m_t.
        # dimensions - batch_size, n_forecasts, quantiles_size
        m_t = self.compute_m_t(current_segment, past_next_changepoint, meta_name_tensor_one_hot)

        # Computing trend value at time(t) for each batch sample.
        # dimensions - batch_size, n_forecasts, quantiles_size
        trend = self.compute_trend(t, k_t, m_t, meta_name_tensor_one_hot)

        return self.bias.unsqueeze(dim=0).unsqueeze(dim=0) + trend

    @property
    def get_trend_deltas(self):
        """trend deltas for regularization.

        update if trend is modelled differently"""
        if self.config_trend is None or self.config_trend.n_changepoints < 1:
            trend_delta = None
        elif self.segmentwise_trend:
            trend_delta = self.trend_deltas[:, :, :] - torch.cat((self.trend_k0, self.trend_deltas[:, :, 0:-1]), dim=2)
        else:
            trend_delta = self.trend_deltas

        return trend_delta

    def add_regularization(self):
        pass

    def compute_k_t(self, current_segment, past_next_changepoint, meta_name_tensor_one_hot):
        """For segmentwise, k_t is the model parameter representing the trend slope(actually, trend slope-k_0) in the current_segment at time t (for each sample of the batch).

        For not segmentwise, k_t is the model parameter representing the difference between trend slope in the current_segment at time t and the trend slope in the previous segment (for each sample of the batch).

        Parameters
        ----------
            current_segment : torch.Tensor, int
                segment corresponding to time t (batch_size, n_forecasts, segments (+ 1))

            past_next_changepoint : torch.Tensor, bool
                whether the a changepoint >= time t (batch_size, n_forecasts, segments (+ 1))

            meta_name_tensor_one_hot : torch.Tensor, float
                Metadata about the all the samples of the model input batch.

                Contains the following:
                    * ``df_name`` (list, str), time series name ID corresponding to each sample of the input batch.
        Returns
        -------
            torch.Tensor
                k_t,  ( batch_size, n_forecasts, quantiles_size)
        """

        pass

    def compute_m_t(self, current_segment, past_next_changepoint, meta_name_tensor_one_hot):
        """m_t represents the value at the origin(t=0) that we would need to have so that if we use (k_t + k_0) as slope,

        we reach the same value at time = chagepoint_start_of_segment_i

        as we would reach by following the segmented slope (having in each segment the slope trend_deltas(i) + k_0)


        Parameters
        ----------
            current_segment : torch.Tensor, int
                segment corresponding to time t (batch_size, n_forecasts, segments (+ 1))

            past_next_changepoint : torch.Tensor, bool
                whether the a changepoint >= time t (batch_size, n_forecasts, segments (+ 1))

            meta_name_tensor_one_hot : torch.Tensor, float
                Metadata about the all the samples of the model input batch.

                Contains the following:
                    * ``df_name`` (list, str), time series name ID corresponding to each sample of the input batch.
        Returns
        -------
            torch.Tensor
                m_t,  ( batch_size, n_forecasts, quantiles_size)
        """

        pass

    def compute_trend(self, t, k_t, m_t, meta_name_tensor_one_hot=None):
        """This method computes the trend component of the model.


        Parameters
        ----------
            t : torch.Tensor, float
                time

            k_t : torch.Tensor, int
                see compute_k_t

            m_t : torch.Tensor, bool
                see compute_m_t

            meta_name_tensor_one_hot : torch.Tensor, float
                Metadata about the all the samples of the model input batch.

                Contains the following:
                    * ``df_name`` (list, str), time series name ID corresponding to each sample of the input batch.
        Returns
        -------
            torch.Tensor
                trend_component,  ( batch_size, n_forecasts, quantiles_size)
        """
        pass


class GlobalPiecewiseLinearTrend(PiecewiseLinearTrend):
    def __init__(self, config, id_list, quantiles, num_trends_modelled, n_forecasts, device):
        super().__init__(
            config=config,
            n_forecasts=n_forecasts,
            num_trends_modelled=num_trends_modelled,
            quantiles=quantiles,
            id_list=id_list,
            device=device,
        )

    def compute_k_t(self, current_segment, past_next_changepoint, meta_name_tensor_one_hot=None):
        """This method overrides the method from the PiecewiseLinear class."""
        # For segmentwise k_t, is the model parameter representing the trend slope(actually, trend slope-k_0) in the current_segment at time t (for each sample of the batch).
        # k_t = k_t(current_segment).
        # dimensions - batch_size, n_forecasts, quantiles_size
        k_t = torch.sum(
            current_segment.unsqueeze(dim=2) * self.trend_deltas.permute(1, 0, 2).unsqueeze(1),
            dim=-1,
        )

        # For not segmentwise k_t is the model parameter representing the difference between trend slope in the current_segment at time t
        # and the trend slope in the previous segment (for each sample of the batch).
        if not self.segmentwise_trend:
            # k_t = k_t(current_segment, previous_segment)
            # dimensions - batch_size, n_forecasts, quantiles_size
            previous_deltas_t = torch.sum(
                past_next_changepoint.unsqueeze(dim=2) * self.trend_deltas.permute(1, 0, 2)[:, :, :-1].unsqueeze(dim=0),
                dim=-1,
            )
            k_t = k_t + previous_deltas_t

        return k_t

    def compute_m_t(self, current_segment, past_next_changepoint, meta_name_tensor_one_hot=None):
        """This method overrides the method from the PiecewiseLinear class."""
        # Computing m_t.
        # m_t represents the value at the origin(t=0) that we would need to have so that if we use (k_t + k_0) as slope,
        # we reach the same value at time = chagepoint_start_of_segment_i
        # that the segmented slope (having in each segment the slope trend_deltas(i) + k_0)
        if self.config_trend.growth != "discontinuous":
            # Intermediate computation: deltas.
            # `deltas`` is representing the difference between trend slope in the current_segment at time t
            #  and the trend slope in the previous segment.
            if self.segmentwise_trend:
                # dimensions - quantiles, num_trends_modelled, segments
                deltas = self.trend_deltas[:, :, :] - torch.cat((self.trend_k0, self.trend_deltas[:, :, 0:-1]), dim=2)

            else:
                deltas = self.trend_deltas

            # dimensions - quantiles, 1, segments
            gammas = -self.trend_changepoints_t[1:] * deltas[:, :, 1:]
            # dimensions - batch_size, n_forecasts, quantiles
            m_t = torch.sum(past_next_changepoint.unsqueeze(dim=2) * gammas.permute(1, 0, 2).unsqueeze(1), dim=-1)

            if not self.segmentwise_trend:
                m_t = m_t.detach()
        else:
            # For discontinuous, trend_m is a parameter to optimize, as it is not defined just by trend_deltas & trend_k0
            # m_t = m_t(current_segment)
            # dimensions - batch_size, n_forecasts, quantiles
            m_t = torch.sum(current_segment.unsqueeze(dim=2) * self.trend_m.permute(1, 0, 2).unsqueeze(dim=0), dim=-1)

        return m_t

    def compute_trend(self, t, k_t, m_t, meta_name_tensor_one_hot=None):
        """This method overrides the method from the PiecewiseLinear class."""
        # Computing trend value at time(t) for each batch sample.
        # dimensions - batch_size, n_forecasts, quantiles
        return (self.trend_k0.permute(1, 2, 0) + k_t) * torch.unsqueeze(t, dim=2) + m_t


class LocalPiecewiseLinearTrend(PiecewiseLinearTrend):
    def __init__(self, config, id_list, quantiles, num_trends_modelled, n_forecasts, device):
        super().__init__(
            config=config,
            n_forecasts=n_forecasts,
            num_trends_modelled=num_trends_modelled,
            quantiles=quantiles,
            id_list=id_list,
            device=device,
        )

    def compute_k_t(self, current_segment, past_next_changepoint, meta_name_tensor_one_hot):
        """This method overrides the method from the PiecewiseLinear class."""
        # For segmentwise k_t, is the model parameter representing the trend slope(actually, trend slope-k_0) in the current_segment at time t (for each sample of the batch).
        # k_t = k_t(current_segment, sample metadata)
        # dimensions - quantiles, batch_size, segments (+ 1)
        trend_deltas_by_sample = torch.sum(
            meta_name_tensor_one_hot.unsqueeze(dim=0).unsqueeze(dim=-1) * self.trend_deltas.unsqueeze(dim=1), dim=2
        )
        # dimensions - batch_size, n_forecasts, quantiles_size
        k_t = torch.sum(current_segment.unsqueeze(dim=2) * trend_deltas_by_sample.permute(1, 0, 2).unsqueeze(1), dim=-1)

        # For not segmentwise k_t is the model parameter representing the difference between trend slope in the current_segment at time t
        # and the trend slope in the previous segment (for each sample of the batch).
        if not self.segmentwise_trend:
            # k_t = k_t(current_segment, previous_segment, sample metadata)
            previous_deltas_t = torch.sum(
                past_next_changepoint.unsqueeze(dim=2)
                * trend_deltas_by_sample.permute(1, 0, 2)[:, :, :-1].unsqueeze(dim=1),
                dim=-1,
            )
            # dimensions - batch_size, n_forecasts, quantiles_size
            k_t = k_t + previous_deltas_t

        return k_t

    def compute_m_t(self, current_segment, past_next_changepoint, meta_name_tensor_one_hot=None):
        """This method overrides the method from the PiecewiseLinear class."""
        # Computing m_t.
        # m_t represents the value at the origin(t=0) that we would need to have so that if we use (k_t + k_0) as slope,
        # we reach the same value at time = chagepoint_start_of_segment_i
        # that the segmented slope (having in each segment the slope trend_deltas(i) + k_0)
        if self.config_trend.growth != "discontinuous":
            # Intermediate computation: deltas.
            # `deltas`` is representing the difference between trend slope in the current_segment at time t
            #  and the trend slope in the previous segment.
            if self.segmentwise_trend:
                # dimensions - quantiles, num_trends_modelled, segments
                deltas = self.trend_deltas[:, :, :] - torch.cat((self.trend_k0, self.trend_deltas[:, :, 0:-1]), dim=2)

            else:
                deltas = self.trend_deltas

            # We create a dict of gammas based on the df_name
            # m_t = m_t(current_segment, sample metadata)
            # dimensions - quantiles, num_time_series, segments
            gammas_0 = -self.trend_changepoints_t[1:] * deltas[:, :, 1:]
            # dimensions - quantiles, segments, batch_size
            gammas = torch.sum(
                torch.transpose(meta_name_tensor_one_hot, 1, 0).unsqueeze(dim=-2).unsqueeze(dim=0)
                * torch.unsqueeze(gammas_0, dim=-1),
                dim=1,
            )
            # dimensions - batch_size, n_forecasts, quantiles
            m_t = torch.sum(past_next_changepoint.unsqueeze(2) * gammas.permute(2, 0, 1).unsqueeze(1), dim=-1)

            if not self.segmentwise_trend:
                m_t = m_t.detach()
        else:
            # For discontinuous, trend_m is a parameter to optimize, as it is not defined just by trend_deltas & trend_k0
            # m_t = m_t(current_segment, sample metadata)
            # dimensions - quantiles, batch_size, segments
            m_t_0 = torch.sum(
                meta_name_tensor_one_hot.unsqueeze(dim=0).unsqueeze(dim=-1) * self.trend_m.unsqueeze(dim=1), dim=2
            )
            # dimensions - batch_size, n_forecasts, quantiles
            m_t = torch.sum(
                current_segment.unsqueeze(dim=2) * m_t_0.permute(1, 0, 2).unsqueeze(dim=1),
                dim=-1,
            )

        return m_t

    def compute_trend(self, t, k_t, m_t, meta_name_tensor_one_hot=None):
        """This method overrides the method from the PiecewiseLinear class."""
        # Computing trend value at time(t) for each batch sample.
        # trend_k_0 = trend_k_0(current_segment, sample metadata)
        trend_k_0 = torch.sum(
            meta_name_tensor_one_hot.unsqueeze(dim=0).unsqueeze(dim=-1) * self.trend_k0.unsqueeze(dim=1), dim=2
        ).permute(1, 2, 0)
        # dimensions - batch_size, n_forecasts, quantiles
        return (trend_k_0 + k_t) * t.unsqueeze(dim=2) + m_t

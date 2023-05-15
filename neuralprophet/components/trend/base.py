from abc import abstractmethod

from neuralprophet.components import BaseComponent
from neuralprophet.utils_torch import init_parameter


class Trend(BaseComponent):
    def __init__(self, config, id_list, quantiles, num_trends_modelled, n_forecasts, device):
        super().__init__(n_forecasts=n_forecasts, quantiles=quantiles, id_list=id_list, device=device)
        self.config_trend = config
        self.num_trends_modelled = num_trends_modelled

        # if only 1 time series, global strategy
        if len(self.id_list) == 1:
            self.config_trend.trend_global_local = "global"

        # dimensions  - [no. of quantiles, 1 bias shape]
        self.bias = init_parameter(
            dims=[
                len(self.quantiles),
            ]
        )

    @property
    @abstractmethod
    def get_trend_deltas(self):
        """trend deltas for regularization.

        update if trend is modelled differently"""
        pass

    @abstractmethod
    def add_regularization(self):
        """add regularization to loss"""
        pass

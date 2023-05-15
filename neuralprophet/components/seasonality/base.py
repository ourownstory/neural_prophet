from neuralprophet.components import BaseComponent


class Seasonality(BaseComponent):
    def __init__(self, config, id_list, quantiles, num_seasonalities_modelled, n_forecasts, device):
        super().__init__(n_forecasts=n_forecasts, quantiles=quantiles, id_list=id_list, device=device)
        self.config_seasonality = config
        self.num_seasonalities_modelled = num_seasonalities_modelled

        # if only 1 time series, global strategy
        if len(self.id_list) == 1:
            self.config_seasonality.global_local = "global"

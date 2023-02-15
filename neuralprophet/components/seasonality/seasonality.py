from neuralprophet.components import BaseComponent


class Seasonality(BaseComponent):
    def __init__(
        self,
        config,
        id_list,
        quantiles,
        num_seasonalities_modelled,
        num_seasonalities_modelled_dict,
        n_forecasts,
        device,
    ):
        super().__init__(n_forecasts=n_forecasts, quantiles=quantiles, id_list=id_list, device=device)
        self.config_seasonality = config
        self.num_seasonalities_modelled = num_seasonalities_modelled
        self.num_seasonalities_modelled_dict = num_seasonalities_modelled_dict

        # if only 1 time series, global strategy
        if len(self.id_list) == 1:
            self.config_seasonality.global_local = "global"
            for season_i in self.config_seasonality.periods:
                self.config_seasonality.periods[season_i].global_local = "global"

        # if all self.config_seasonality.periods[season_i] are the same(x), then self.config_seasonality.global_local = x
        if (
            len(
                set(
                    [
                        self.config_seasonality.periods[season_i].global_local
                        for season_i in self.config_seasonality.periods
                    ]
                )
            )
            == 1
        ):
            self.config_seasonality.global_local = list(
                set(
                    [
                        self.config_seasonality.periods[season_i].global_local
                        for season_i in self.config_seasonality.periods
                    ]
                )
            )[0]

        # if all self.config_seasonality.periods[season_i] are different, then self.config_seasonality.global_local = 'glocal'
        if (
            len(
                set(
                    [
                        self.config_seasonality.periods[season_i].global_local
                        for season_i in self.config_seasonality.periods
                    ]
                )
            )
            > 1
        ):
            self.config_seasonality.global_local = "glocal"

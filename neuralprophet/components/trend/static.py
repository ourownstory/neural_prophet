from neuralprophet.components.trend import Trend


class StaticTrend(Trend):
    def __init__(self, config, id_list, quantiles, num_trends_modelled, n_forecasts, device):
        super().__init__(
            config=config,
            n_forecasts=n_forecasts,
            num_trends_modelled=num_trends_modelled,
            quantiles=quantiles,
            id_list=id_list,
            device=device,
        )

    def forward(self, t, meta):
        """
        Computes trend based on model configuration.

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
        return self.bias.unsqueeze(dim=0).repeat(t.shape[0], self.n_forecasts, 1)

    @property
    def get_trend_deltas(self):
        pass

    def add_regularization(self):
        pass

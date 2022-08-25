from neuralprophet.forecaster import NeuralProphet
import pandas as pd
import logging

log = logging.getLogger("NP.forecaster")


class Prophet(NeuralProphet):
    def __init__(
        self,
        growth="linear",
        changepoints=None,
        n_changepoints=25,
        changepoint_range=0.8,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        holidays=None,
        seasonality_mode="additive",
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        mcmc_samples=0,
        interval_width=0.80,
        uncertainty_samples=1000,
        stan_backend=None,
    ):
        # holidays, seasonality_prior_scale, holidays_prior_scale,
        # changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, stan_backend
        # Set Prophet-like default args
        kwargs["quantiles"] = [0.9, 0.1]
        # Run the NeuralProphet function
        super(Prophet, self).__init__(
            growth=growth,
            changepoints=changepoints,
            n_changepoints=n_changepoints,
            changepoint_range=changepoint_range,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode=seasonality_mode,
            quantiles=[0.9, 0.1],
        )
        # Overwrite NeuralProphet properties
        self.name = "Prophet"
        self.history = None

        # Warnings
        if holidays:
            log.warning("Passing holidays directly to NeuralProphet does not work, please use add_country_holidays()")

    def fit(self, df, **kwargs):
        # Run the NeuralProphet function
        metrics_df = super(Prophet, self).fit(df=df)
        # Store the df for future use like in Prophet
        self.history = df
        return metrics_df

    def make_future_dataframe(self, periods, freq="D", include_history=True):
        # Convert all frequencies to daily
        if freq == "M":
            periods = periods * 30
        # Run the NeuralProphet function
        df_future = super(Prophet, self).make_future_dataframe(
            df=self.history, periods=periods, n_historic_predictions=True
        )
        return df_future


class ProphetAuto(NeuralProphet):
    def __init__(self, *args, **kwargs):
        # Set Prophet-like default args
        kwargs["quantiles"] = [0.9, 0.1]
        # Run the NeuralProphet function
        super(Prophet, self).__init__(*args, **kwargs)
        # Overwrite NeuralProphet properties
        self.name = "Prophet"
        self.history = None

    def fit(self, *args, **kwargs):
        # Run the NeuralProphet function
        metrics_df = super(Prophet, self).fit(*args, **kwargs)
        # Store the df for future use like in Prophet
        self.history = kwargs.get("df", args[0])
        return metrics_df

    def make_future_dataframe(self, *args, **kwargs):
        # Set Prophet-like default args
        kwargs["n_historic_predictions"] = kwargs.get("n_historic_predictions", True)
        # Use the provided df or fallback to the stored df during fit()
        try:
            df = kwargs.get("df", args[0])
        except:
            df = self.history
        kwargs["df"] = df
        # Run the NeuralProphet function
        df_future = super(Prophet, self).make_future_dataframe(**kwargs)
        return df_future

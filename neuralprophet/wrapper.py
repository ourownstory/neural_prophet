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
        seasonality_prior_scale=None,  # 10.0,
        holidays_prior_scale=None,  # 10.0,
        changepoint_prior_scale=None,  # 0.05,
        mcmc_samples=None,  # 0,
        interval_width=0.80,
        uncertainty_samples=None,  # 1000,
        stan_backend=None,
        **kwargs,
    ):
        # Check for unsupported features
        if seasonality_prior_scale or holidays_prior_scale or changepoint_prior_scale:
            log.info(
                "seasonality_prior_scale, holidays_prior_scale and changepoint_prior_scale are not used in NeuralProphet."
            )
        if mcmc_samples or uncertainty_samples:
            log.info("mcmc_samples and uncertainty_samples are not used in NeuralProphet.")
        if stan_backend:
            log.info("stan_backend is not used in NeuralProphet.")
        if holidays:
            raise NotImplementedError(
                "Passing holidays directly to NeuralProphet does not work, please use add_country_holidays()"
            )
        # Run the NeuralProphet function
        super(Prophet, self).__init__(
            growth=growth,
            changepoints=changepoints,
            n_changepoints=n_changepoints,
            changepoints_range=changepoint_range,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode=seasonality_mode,
            prediction_interval=interval_width,
            **kwargs,
        )
        # Overwrite NeuralProphet properties
        self.name = "Prophet"
        self.history = None

    def fit(self, df, **kwargs):
        # Check for unsupported features
        if "cap" in df.columns:
            raise NotImplementedError("Saturating forecasts using cap is not supported in NeuralProphet.")
        if "show_progress" in kwargs:
            del kwargs["show_progress"]
        # Run the NeuralProphet function
        metrics_df = super(Prophet, self).fit(df=df, **kwargs)
        # Store the df for future use like in Prophet
        self.history = df
        return metrics_df

    def predict(self, df=None, **kwargs):
        if df is None:
            df = self.history.copy()
        df = super(Prophet, self).predict(df=df, **kwargs)
        return df

    def make_future_dataframe(self, periods, freq="D", include_history=True, **kwargs):
        # Convert all frequencies to daily
        if freq == "M":
            periods = periods * 30
        # Run the NeuralProphet function
        df_future = super(Prophet, self).make_future_dataframe(
            df=self.history, periods=periods, n_historic_predictions=include_history, **kwargs
        )
        return df_future

    def add_seasonality(self, name, period, fourier_order, prior_scale=None, mode=None, condition_name=None):
        # Check for unsupported features
        if condition_name:
            log.warn("Conditioning on seasonality is not supported in NeuralProphet.")
        # Set attributes in NeuralProphet config
        self.season_config.mode = mode
        self.season_config.seasonality_reg = prior_scale
        # Run the NeuralProphet function
        return super().add_seasonality(name, period, fourier_order)

    def add_regressor(self, name, prior_scale=None, standardize="auto", mode="additive"):
        # Run the NeuralProphet function
        super(Prophet, self).add_future_regressor(name, regularization=prior_scale, normalize=standardize, mode=mode)
        return self


def plot_plotly(m, forecast, **kwargs):
    # Run the NeuralProphet plotting function
    fig = m.plot(forecast, plotting_backend="plotly", **kwargs)
    return fig


def plot_components_plotly(m, forecast, **kwargs):
    # Run the NeuralProphet plotting function
    fig = m.plot_components(forecast, plotting_backend="plotly", **kwargs)
    return fig

import logging

import numpy as np

from neuralprophet.forecaster import NeuralProphet

log = logging.getLogger("NP.forecaster")


class TorchProphet(NeuralProphet):
    """
    Prophet wrapper for the NeuralProphet forecaster.

    Parameters
    ----------
    growth: String 'linear' or 'flat' to specify a linear or
        flat trend. Note: 'flat' is equivalent to 'off' in NeuralProphet.
    changepoints: List of dates at which to include potential changepoints. If
        not specified, potential changepoints are selected automatically.
    n_changepoints: Number of potential changepoints to include. Not used
        if input `changepoints` is supplied. If `changepoints` is not supplied,
        then n_changepoints potential changepoints are selected uniformly from
        the first `changepoint_range` proportion of the history.
    changepoint_range: Proportion of history in which trend changepoints will
        be estimated. Defaults to 0.8 for the first 80%. Not used if
        `changepoints` is specified.
    yearly_seasonality: Fit yearly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    weekly_seasonality: Fit weekly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    daily_seasonality: Fit daily seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    holidays: pd.DataFrame with columns holiday (string) and ds (date type)
        and optionally columns lower_window and upper_window which specify a
        range of days around the date to be included as holidays.
        lower_window=-2 will include 2 days prior to the date as holidays. Also
        optionally can have a column prior_scale specifying the prior scale for
        that holiday.
    seasonality_mode: 'additive' (default) or 'multiplicative'.
    seasonality_prior_scale: Not supported for regularisation in NeuralProphet,
        please use the `seasonality_reg` arg instead.
    holidays_prior_scale: Not supported for regularisation in NeuralProphet.
    changepoint_prior_scale: Not supported for regularisation in NeuralProphet,
        please use the `trend_reg` arg instead.
    mcmc_samples: Not required for NeuralProphet
    interval_width: Float, width of the uncertainty intervals provided
        for the forecast. Converted to list of quantiles for NeuralProphet. Use
        the quantiles arg to pass quantiles directly to NeuralProphet.
    uncertainty_samples: Not required for NeuralProphet.
    stan_backend: Not supported by NeuralProphet.
    """

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
        seasonality_prior_scale=None,
        holidays_prior_scale=None,
        changepoint_prior_scale=None,
        mcmc_samples=None,
        interval_width=0.80,
        uncertainty_samples=None,
        stan_backend=None,
        **kwargs,
    ):
        # Check for unsupported features
        if seasonality_prior_scale or holidays_prior_scale or changepoint_prior_scale:
            log.error(
                "Using `_prior_scale` is unsupported for regularisation in NeuralProphet, please use the corresponding `_reg` arg instead."
            )
        if mcmc_samples or uncertainty_samples:
            log.warning(
                "Providing the number of samples for Bayesian inference or Uncertainty estimation is not required in NeuralProphet."
            )
        if stan_backend:
            log.warning("A stan_backend is not used in NeuralProphet. Please remove the parameter")

        # Handle growth
        if growth == "flat":
            log.warning("Using 'flat' growth is equivalent to 'off' in NeuralProphet.")
            growth = "off"

        # Handle quantiles
        if "quantiles" not in kwargs:
            alpha = 1 - interval_width
            quantiles = [np.round(alpha / 2, 4), np.round(1 - (alpha / 2), 4)]

        # Run the NeuralProphet function
        super(TorchProphet, self).__init__(
            growth=growth,
            changepoints=changepoints,
            n_changepoints=n_changepoints,
            changepoints_range=changepoint_range,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode=seasonality_mode,
            quantiles=quantiles,
            **kwargs,
        )
        # Handle holidays as events
        if holidays is not None:
            self.add_events(
                events=list(holidays["holiday"].unique()),
                lower_window=holidays["lower_window"].max(),
                upper_window=holidays["upper_window"].max(),
            )
            self.events_df = holidays.copy()
            self.events_df.rename(columns={"holiday": "event"}, inplace=True)
            self.events_df.drop(["lower_window", "upper_window"], axis=1, errors="ignore", inplace=True)

        # Overwrite NeuralProphet properties
        self.name = "TorchProphet"
        self.history = None

        # Unused properties
        self.train_holiday_names = None

    def validate_inputs(self):
        """
        Validates the inputs to NeuralProphet.
        """
        log.error("Not required in NeuralProphet as all inputs are automatically checked.")

    def validate_column_name(self, name, check_holidays=True, check_seasonalities=True, check_regressors=True):
        """Validates the name of a seasonality, holiday, or regressor.

        Parameters
        ----------
        name: string
        check_holidays: bool check if name already used for holiday
        check_seasonalities: bool check if name already used for seasonality
        check_regressors: bool check if name already used for regressor
        """
        super(TorchProphet, self)._validate_column_name(
            name=name,
            events=check_holidays,
            seasons=check_seasonalities,
            regressors=check_regressors,
            covariates=check_regressors,
        )

    def setup_dataframe(self, df, initialize_scales=False):
        """
        Dummy function that raises an error.

        This function is not supported in NeuralProphet.
        """
        log.error(
            "Not required in NeuralProphet as the dataframe is automatically prepared using the private `_normalize` function."
        )

    def fit(self, df, **kwargs):
        """Fit the NeuralProphet model.

        This sets self.params to contain the fitted model parameters. It is a
        dictionary parameter names as keys and the following items:
            k (Mx1 array): M posterior samples of the initial slope.
            m (Mx1 array): The initial intercept.
            delta (MxN array): The slope change at each of N changepoints.
            beta (MxK matrix): Coefficients for K seasonality features.
            sigma_obs (Mx1 array): Noise level.
        Note that M=1 if MAP estimation.

        Parameters
        ----------
        df: pd.DataFrame containing the history. Must have columns ds (date
            type) and y, the time series. If self.growth is 'logistic', then
            df must also have a column cap that specifies the capacity at
            each ds.
        kwargs: Additional arguments passed to the optimizing or sampling
            functions in Stan.

        Returns
        -------
        The fitted NeuralProphet object.
        """
        # Check for unsupported features
        if "cap" in df.columns:
            raise NotImplementedError("Saturating forecasts using cap is not supported in NeuralProphet.")
        if "show_progress" in kwargs:
            del kwargs["show_progress"]
        # Handle holidays as events
        if hasattr(self, "events_df"):
            df = self.create_df_with_events(df, self.events_df)
        # Run the NeuralProphet function
        metrics_df = super(TorchProphet, self).fit(df=df, **kwargs)
        # Store the df for future use like in Prophet
        self.history = df
        return metrics_df

    def predict(self, df=None, **kwargs):
        """Predict using the NeuralProphet model.

        Parameters
        ----------
        df: pd.DataFrame with dates for predictions (column ds), and capacity
            (column cap) if logistic growth. If not provided, predictions are
            made on the history.

        Returns
        -------
        A pd.DataFrame with the forecast components.
        """
        if df is None:
            df = self.history.copy()
        df = super(TorchProphet, self).predict(df=df, **kwargs)
        for column in df.columns:
            # Copy column according to Prophet naming convention
            if "event_" in column:
                df[column.replace("event_", "")] = df[column]
        return df

    def predict_trend(self, df):
        """Predict trend using the NeuralProphet model.

        Parameters
        ----------
        df: Prediction dataframe.

        Returns
        -------
        Vector with trend on prediction dates.
        """
        df = super(TorchProphet, self).predict_trend(self, df, quantile=0.5)
        return df["trend"].to_numpy()

    def make_future_dataframe(self, periods, freq="D", include_history=True, **kwargs):
        """Simulate the trend using the extrapolated generative model.

        Parameters
        ----------
        periods: Int number of periods to forecast forward.
        freq: Any valid frequency for pd.date_range, such as 'D' or 'M'.
        include_history: Boolean to include the historical dates in the data
            frame for predictions.

        Returns
        -------
        pd.Dataframe that extends forward from the end of self.history for the
        requested number of periods.
        """
        # Convert all frequencies to daily
        if freq == "M":
            periods = periods * 30
        # Run the NeuralProphet function
        if hasattr(self, "events_df"):
            # Pass holidays as events
            df_future = super(TorchProphet, self).make_future_dataframe(
                df=self.history,
                events_df=self.events_df,
                periods=periods,
                n_historic_predictions=include_history,
                **kwargs,
            )
        else:
            df_future = super(TorchProphet, self).make_future_dataframe(
                df=self.history, periods=periods, n_historic_predictions=include_history, **kwargs
            )
        return df_future

    def add_seasonality(self, name, period, fourier_order, prior_scale=None, mode=None, condition_name=None, **kwargs):
        """Add a seasonal component with specified period, number of Fourier
        components, and prior scale.

        Increasing the number of Fourier components allows the seasonality to
        change more quickly (at risk of overfitting). Default values for yearly
        and weekly seasonalities are 10 and 3 respectively.

        Increasing prior scale will allow this seasonality component more
        flexibility, decreasing will dampen it. If not provided, will use the
        seasonality_prior_scale provided on initialization (defaults
        to 10).

        Mode can be specified as either 'additive' or 'multiplicative'. If not
        specified, self.seasonality_mode will be used (defaults to additive).
        Additive means the seasonality will be added to the trend,
        multiplicative means it will multiply the trend.

        If condition_name is provided, the dataframe passed to `fit` and
        `predict` should have a column with the specified condition_name
        containing booleans which decides when to apply seasonality.

        Parameters
        ----------
        name: string name of the seasonality component.
        period: float number of days in one period.
        fourier_order: int number of Fourier components to use.
        prior_scale: Not supported in NeuralProphet.
        mode: optional 'additive' or 'multiplicative'
        condition_name: Not supported in NeuralProphet.

        Returns
        -------
        The NeuralProphet object.
        """
        # Check for unsupported features
        if condition_name:
            raise NotImplementedError("Conditioning on seasonality is not supported in NeuralProphet.")
        if prior_scale:
            log.warning(
                "Prior scale is not supported in NeuralProphet. Use the `regularisation` parameter for regularisation."
            )
        # Set attributes in NeuralProphet config
        try:
            self.season_config.mode = mode
        except AttributeError:
            log.warning("Cannot set the seasonality mode attribute in NeuralProphet. Pleas inspect manually.")
        # Run the NeuralProphet function
        return super(TorchProphet, self).add_seasonality(name, period, fourier_order, **kwargs)

    def add_regressor(self, name, prior_scale=None, standardize="auto", mode="additive", **kwargs):
        """Add an additional (future) regressor to be used for fitting and predicting.

        Parameters
        ----------
        name: string name of the regressor.
        prior_scale: Not supported in NeuralProphet.
        standardize: optional, specify whether this regressor will be
            standardized prior to fitting. Can be 'auto' (standardize if not
            binary), True, or False.
        mode: optional, 'additive' or 'multiplicative'. Defaults to
            self.seasonality_mode. Not supported in NeuralProphet.

        Returns
        -------
        The NeuralProphet object.
        """
        # Check for unsupported features
        if prior_scale:
            log.warning(
                "Prior scale is not supported in NeuralProphet. Use the `regularisation` parameter for regularisation."
            )
        # Run the NeuralProphet function
        super(TorchProphet, self).add_future_regressor(name, normalize=standardize, **kwargs)
        return self

    def add_country_holidays(self, country_name, **kwargs):
        """Add in built-in holidays for the specified country.

        These holidays will be included in addition to any specified on model
        initialization.

        Holidays will be calculated for arbitrary date ranges in the history
        and future. See the online documentation for the list of countries with
        built-in holidays.

        Built-in country holidays can only be set for a single country.

        Parameters
        ----------
        country_name: Name of the country, like 'UnitedStates' or 'US'

        Returns
        -------
        The NeuralProphet object.
        """
        super(TorchProphet, self).add_country_holidays(country_name=country_name, **kwargs)

    def plot(
        self,
        fcst,
        ax=None,
        uncertainty=True,
        plot_cap=True,
        xlabel="ds",
        ylabel="y",
        figsize=(10, 6),
        include_legend=False,
        **kwargs,
    ):
        """Plot the NeuralProphet forecast.

        Parameters
        ----------
        fcst: pd.DataFrame output of self.predict.
        ax: Optional matplotlib axes on which to plot.
        uncertainty: Not supported in NeuralProphet.
        plot_cap: Not supported in NeuralProphet.
        xlabel: Optional label name on X-axis
        ylabel: Optional label name on Y-axis
        figsize: Optional tuple width, height in inches.
        include_legend: Not supported in NeuralProphet.

        Returns
        -------
        A matplotlib figure.
        """
        log.warning("The attributes `uncertainty`, `plot_cap` and `include_legend` are not supported by NeuralProphet")
        fig = super(TorchProphet, self).plot(fcst=fcst, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize, **kwargs)
        return fig

    def plot_components(
        self, fcst, uncertainty=True, plot_cap=True, weekly_start=0, yearly_start=0, figsize=None, **kwargs
    ):
        """Plot the NeuralProphet forecast components.

        Will plot whichever are available of: trend, holidays, weekly
        seasonality, and yearly seasonality.

        Parameters
        ----------
        fcst: pd.DataFrame output of self.predict.
        uncertainty: Not supported in NeuralProphet.
        plot_cap: Not supported in NeuralProphet.
        weekly_start: Not supported in NeuralProphet.
        yearly_start: Not supported in NeuralProphet.
        figsize: Optional tuple width, height in inches.

        Returns
        -------
        A matplotlib figure.
        """
        log.warning(
            "The attributes `uncertainty`, `plot_cap`, `weekly_start` and `yearly_start` are not supported by NeuralProphet"
        )
        fig = super(TorchProphet, self).plot_components(fcst=fcst, figsize=figsize, **kwargs)
        return fig


def plot(
    self,
    fcst,
    ax=None,
    uncertainty=True,
    plot_cap=True,
    xlabel="ds",
    ylabel="y",
    figsize=(10, 6),
    include_legend=False,
    **kwargs,
):
    """Plot the NeuralProphet forecast.

    Parameters
    ----------
    fcst: pd.DataFrame output of self.predict.
    ax: Optional matplotlib axes on which to plot.
    uncertainty: Not supported in NeuralProphet.
    plot_cap: Not supported in NeuralProphet.
    xlabel: Optional label name on X-axis
    ylabel: Optional label name on Y-axis
    figsize: Optional tuple width, height in inches.
    include_legend: Not supported in NeuralProphet.

    Returns
    -------
    A matplotlib figure.
    """
    log.warning("The attributes `uncertainty`, `plot_cap` and `include_legend` are not supported by NeuralProphet")
    fig = super(TorchProphet, self).plot(fcst=fcst, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize, **kwargs)
    return fig


def plot_plotly(
    self,
    fcst,
    ax=None,
    uncertainty=True,
    plot_cap=True,
    xlabel="ds",
    ylabel="y",
    figsize=(10, 6),
    include_legend=False,
    **kwargs,
):
    """Plot the NeuralProphet forecast.

    Parameters
    ----------
    fcst: pd.DataFrame output of self.predict.
    ax: Optional matplotlib axes on which to plot.
    uncertainty: Not supported in NeuralProphet.
    plot_cap: Not supported in NeuralProphet.
    xlabel: Optional label name on X-axis
    ylabel: Optional label name on Y-axis
    figsize: Optional tuple width, height in inches.
    include_legend: Not supported in NeuralProphet.

    Returns
    -------
    A matplotlib figure.
    """
    log.warning("The attributes `uncertainty`, `plot_cap` and `include_legend` are not supported by NeuralProphet")
    fig = super(TorchProphet, self).plot(
        fcst=fcst, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize, plotting_backend="plotly", **kwargs
    )
    return fig


def plot_components(m, fcst, uncertainty=True, plot_cap=True, weekly_start=0, yearly_start=0, figsize=None, **kwargs):
    """
    Plot the NeuralProphet forecast components.

    Will plot whichever are available of: trend, holidays, weekly
    seasonality, yearly seasonality, and additive and multiplicative extra
    regressors.

    Parameters
    ----------
    m: NeuralProphet model.
    fcst: pd.DataFrame output of m.predict.
    uncertainty: Not supported in NeuralProphet.
    plot_cap: Not supported in NeuralProphet.
    weekly_start: Not supported in NeuralProphet.
    yearly_start: Not supported in NeuralProphet.
    figsize: Optional tuple width, height in inches.

    Returns
    -------
    A matplotlib figure.
    """
    log.warning(
        "The attributes `uncertainty`, `plot_cap`, `weekly_start` and `yearly_start` are not supported by NeuralProphet"
    )
    # Run the NeuralProphet plotting function
    fig = m.plot_components(fcst, **kwargs)
    return fig


def plot_components_plotly(m, fcst, uncertainty=True, plot_cap=True, figsize=(900, 200), **kwargs):
    """
    Plot the NeuralProphet forecast components using Plotly.
    See plot_plotly() for Plotly setup instructions

    Will plot whichever are available of: trend, holidays, weekly
    seasonality, yearly seasonality, and additive and multiplicative extra
    regressors.

    Parameters
    ----------
    m: NeuralProphet model.
    fcst: pd.DataFrame output of m.predict.
    uncertainty: Not supported in NeuralProphet.
    plot_cap: Not supported in NeuralProphet.
    figsize: Not supported in NeuralProphet.
    Returns
    -------
    A Plotly Figure.
    """
    log.warning(
        "The attributes `uncertainty`, `plot_cap`, `weekly_start` and `yearly_start` are not supported by NeuralProphet"
    )
    # Run the NeuralProphet plotting function
    fig = m.plot_components(fcst, figsize=None, plotting_backend="plotly", **kwargs)
    return fig

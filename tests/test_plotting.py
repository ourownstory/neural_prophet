#!/usr/bin/env python3

import logging
import os
import pathlib

import pandas as pd
import pytest

from neuralprophet import NeuralProphet

log = logging.getLogger("NP.test")
log.setLevel("DEBUG")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")
NROWS = 256
EPOCHS = 1
BATCH_SIZE = 128
LR = 1.0

PLOT = False

# plot tests cover both plotting backends
decorator_input = ["plotting_backend", [("plotly"), ("matplotlib")]]


@pytest.mark.parametrize(*decorator_input)
def test_plot(plotting_backend):
    log.info(f"testing: Basic plotting with forecast in focus with {plotting_backend}")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_forecasts=7,
        n_lags=14,
    )
    metrics_df = m.fit(df, freq="D")

    m.highlight_nth_step_ahead_of_each_forecast(7)
    future = m.make_future_dataframe(df, n_historic_predictions=10)
    forecast = m.predict(future)
    fig1 = m.plot(forecast, plotting_backend=plotting_backend)
    fig2 = m.plot_latest_forecast(forecast, plotting_backend=plotting_backend)
    fig3 = m.plot_components(forecast, plotting_backend=plotting_backend)
    fig4 = m.plot_parameters(plotting_backend=plotting_backend)

    log.info(f"testing: Basic plotting without forecast in focus with {plotting_backend}")
    m.highlight_nth_step_ahead_of_each_forecast(None)
    future = m.make_future_dataframe(df, n_historic_predictions=10)
    forecast = m.predict(future)
    fig5 = m.plot(forecast, plotting_backend=plotting_backend)
    fig6 = m.plot_latest_forecast(forecast, plotting_backend=plotting_backend)
    fig7 = m.plot_components(forecast, plotting_backend=plotting_backend)
    fig8 = m.plot_parameters(plotting_backend=plotting_backend)

    # only show plots in interactive mode. gh actions are non-interactive
    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()
        fig5.show()
        fig6.show()
        fig7.show()
        fig8.show()


@pytest.mark.parametrize(*decorator_input)
def test_plot_components(plotting_backend):
    log.info(f"testing: Plotting components with forecast in focus with {plotting_backend}")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_forecasts=7,
        n_lags=14,
    )
    metrics_df = m.fit(df, freq="D")

    m.highlight_nth_step_ahead_of_each_forecast(7)
    future = m.make_future_dataframe(df, n_historic_predictions=10)
    forecast = m.predict(future)

    fig1 = m.plot_components(forecast, plotting_backend=plotting_backend)

    log.info(f"testing: Plotting components without forecast in focus with {plotting_backend}")
    m.highlight_nth_step_ahead_of_each_forecast(None)
    future = m.make_future_dataframe(df, n_historic_predictions=10)
    forecast = m.predict(future)
    fig2 = m.plot_components(forecast, plotting_backend=plotting_backend)
    # select components manually
    fig3 = m.plot_components(forecast, components=["autoregression"], plotting_backend="matplotlib")
    # select plotting components per period
    fig4 = m.plot_components(forecast, one_period_per_season=True, plotting_backend="plotly")

    log.info("Plot components with wrong component selection - Raise ValueError")
    with pytest.raises(ValueError):
        m.plot_components(forecast, components=["quantiles"], plotting_backend="plotly")
    with pytest.raises(ValueError):
        m.plot_components(forecast, components=["trend123"], plotting_backend="plotly")

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()


@pytest.mark.parametrize(*decorator_input)
def test_plot_parameters(plotting_backend):
    log.info(f"testing: Plotting parameters with forecast in focus with {plotting_backend}")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_forecasts=7,
        n_lags=14,
    )
    metrics_df = m.fit(df, freq="D")

    m.highlight_nth_step_ahead_of_each_forecast(7)
    future = m.make_future_dataframe(df, n_historic_predictions=10)
    forecast = m.predict(future)

    fig1 = m.plot_parameters(plotting_backend=plotting_backend)

    log.info(f"testing: Plotting parameters without forecast in focus with {plotting_backend}")
    m.highlight_nth_step_ahead_of_each_forecast(None)
    future = m.make_future_dataframe(df, n_historic_predictions=10)
    forecast = m.predict(future)
    fig2 = m.plot_parameters(plotting_backend=plotting_backend)

    # select components manually
    fig3 = m.plot_parameters(components="trend", plotting_backend="plotly")

    log.info("Plot parameters with wrong component selection - Raise ValueError")
    with pytest.raises(ValueError):
        m.plot_parameters(components=["events"], plotting_backend="plotly")
    with pytest.raises(ValueError):
        m.plot_parameters(components=["trend123"], plotting_backend="plotly")

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()


@pytest.mark.parametrize(*decorator_input)
def test_plot_global_local_parameters(plotting_backend):
    log.info(f"Plotting global modeling + global normalization with {plotting_backend}")
    df = pd.read_csv(PEYTON_FILE, nrows=512)
    df1_0 = df.copy(deep=True)
    df1_0["ID"] = "df1"
    df2_0 = df.copy(deep=True)
    df2_0["ID"] = "df2"
    df3_0 = df.copy(deep=True)
    df3_0["ID"] = "df3"
    m = NeuralProphet(
        n_forecasts=2,
        n_lags=10,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        trend_global_local="local",
        season_global_local="local",
        weekly_seasonality=True,
        daily_seasonality=True,
        yearly_seasonality=True,
    )
    train_df, test_df = m.split_df(pd.concat((df1_0, df2_0, df3_0)), valid_p=0.33, local_split=True)
    m.fit(train_df)
    future = m.make_future_dataframe(test_df)
    forecast = m.predict(future)

    fig1 = m.plot_parameters(df_name="df1", plotting_backend=plotting_backend)
    fig2 = m.plot_parameters(plotting_backend=plotting_backend)
    fig3 = m.plot_components(forecast, df_name="df1", plotting_backend=plotting_backend)
    log.info(f"Plotting global modeling with {plotting_backend}")
    df1 = df.copy(deep=True)
    df1["ID"] = "df1"
    df2 = df.copy(deep=True)
    df2["ID"] = "df2"
    df_global = pd.concat((df1, df2))
    m = NeuralProphet(
        n_forecasts=7,
        n_lags=14,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    metrics_df = m.fit(df_global, freq="D")
    future = m.make_future_dataframe(df_global, periods=m.n_forecasts, n_historic_predictions=10)
    forecast = m.predict(future)

    log.info(f"Plot forecast with many IDs with {plotting_backend} - Raise exceptions")
    with pytest.raises(Exception):
        m.plot(forecast)
    with pytest.raises(Exception):
        m.plot_latest_forecast(forecast, include_previous_forecasts=10)
    with pytest.raises(Exception):
        m.plot_components(forecast)
    forecast = m.predict(df_global)
    with pytest.raises(Exception):
        m.plot(forecast)
    with pytest.raises(Exception):
        m.plot_latest_forecast(forecast, include_previous_forecasts=10)
    with pytest.raises(Exception):
        m.plot_components(forecast)

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()


@pytest.mark.parametrize(*decorator_input)
def test_plot_events(plotting_backend):
    log.info(f"testing: Plotting with events with {plotting_backend}")
    df = pd.read_csv(PEYTON_FILE)[-NROWS:]
    playoffs = pd.DataFrame(
        {
            "event": "playoff",
            "ds": pd.to_datetime(
                [
                    "2008-01-13",
                    "2009-01-03",
                    "2010-01-16",
                    "2010-01-24",
                    "2010-02-07",
                    "2011-01-08",
                    "2013-01-12",
                    "2014-01-12",
                    "2014-01-19",
                    "2014-02-02",
                    "2015-01-11",
                    "2016-01-17",
                    "2016-01-24",
                    "2016-02-07",
                ]
            ),
        }
    )
    superbowls = pd.DataFrame(
        {
            "event": "superbowl",
            "ds": pd.to_datetime(["2010-02-07", "2014-02-02", "2016-02-07"]),
        }
    )
    events_df = pd.concat((playoffs, superbowls))
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_lags=2,
        n_forecasts=30,
        daily_seasonality=False,
    )
    # set event windows
    m = m.add_events(
        ["superbowl", "playoff"], lower_window=-1, upper_window=1, mode="multiplicative", regularization=0.5
    )
    # add the country specific holidays
    m = m.add_country_holidays("US", mode="additive", regularization=0.5)
    m.add_country_holidays("Indonesia")
    m.add_country_holidays("Thailand")
    m.add_country_holidays("Philippines")
    m.add_country_holidays("Pakistan")
    m.add_country_holidays("Belarus")
    history_df = m.create_df_with_events(df, events_df)
    metrics_df = m.fit(history_df, freq="D")
    future = m.make_future_dataframe(df=history_df, events_df=events_df, periods=30, n_historic_predictions=90)
    forecast = m.predict(df=future)
    log.debug(f"Event Parameters:: {m.model.event_params}")

    fig1 = m.plot_components(forecast, plotting_backend=plotting_backend)
    fig2 = m.plot(forecast, plotting_backend=plotting_backend)
    fig3 = m.plot_parameters(plotting_backend=plotting_backend)

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()


@pytest.mark.parametrize(*decorator_input)
def test_plot_trend(plotting_backend):
    log.info(f"testing: Plotting linear trend with {plotting_backend}")
    df = pd.read_csv(AIR_FILE)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_changepoints=0,
        yearly_seasonality=2,
        seasonality_mode="multiplicative",
    )
    metrics = m.fit(df, freq="MS")
    future = m.make_future_dataframe(df, periods=48, n_historic_predictions=len(df) - m.n_lags)
    forecast = m.predict(future)
    fig1 = m.plot(forecast, plotting_backend=plotting_backend)
    fig2 = m.plot_components(forecast, plotting_backend=plotting_backend)
    fig3 = m.plot_parameters(plotting_backend=plotting_backend)

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()


@pytest.mark.parametrize(*decorator_input)
def test_plot_seasonality(plotting_backend):
    log.info(f"testing: Plotting with additive seasonality with {plotting_backend}")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        yearly_seasonality=8,
        weekly_seasonality=4,
        daily_seasonality=30,
        seasonality_mode="additive",
        seasonality_reg=1,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=365, periods=365)
    forecast = m.predict(df=future)

    fig1 = m.plot_components(forecast, plotting_backend=plotting_backend)
    fig2 = m.plot(forecast, plotting_backend=plotting_backend)
    fig3 = m.plot_parameters(plotting_backend=plotting_backend)

    log.info(f"testing: Plotting with additive custom seasonality with {plotting_backend}")
    other_seasons = False
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        yearly_seasonality=other_seasons,
        weekly_seasonality=other_seasons,
        daily_seasonality=other_seasons,
        seasonality_mode="additive",
        seasonality_reg=1,
    )
    m = m.add_seasonality(name="quarterly", period=90, fourier_order=5)
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=365, periods=365)
    forecast = m.predict(df=future)

    fig4 = m.plot_parameters(plotting_backend=plotting_backend)

    log.info("testing: Seasonality Plotting with Business Day freq")
    m = NeuralProphet(
        yearly_seasonality=8,
        weekly_seasonality=4,
        daily_seasonality=30,
        seasonality_mode="additive",
        seasonality_reg=1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    df["ds"] = pd.to_datetime(df["ds"])
    # create a range of business days over that period
    bdays = pd.bdate_range(start=df["ds"].min(), end=df["ds"].max())
    # Filter the series to just those days contained in the business day range.
    df = df[df["ds"].isin(bdays)]
    metrics_df = m.fit(df, freq="B")
    forecast = m.predict(df)
    fig5 = m.plot_components(forecast, plotting_backend="plotly")
    fig6 = m.plot_parameters(plotting_backend="plotly")

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()
        fig5.show()
        fig6.show()


@pytest.mark.parametrize(*decorator_input)
def test_plot_daily_seasonality(plotting_backend):
    log.info(f"testing: Plotting with daily seasonality with {plotting_backend}")
    df = pd.read_csv(YOS_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        changepoints_range=0.95,
        n_changepoints=50,
        trend_reg=1,
        weekly_seasonality=False,
        daily_seasonality=10,
    )

    metrics = m.fit(df, freq="5min")
    future = m.make_future_dataframe(df, periods=60 // 5 * 24 * 7, n_historic_predictions=True)
    forecast = m.predict(future)

    fig1 = m.plot_components(forecast, plotting_backend=plotting_backend)
    fig2 = m.plot(forecast, plotting_backend=plotting_backend)
    fig3 = m.plot_parameters(plotting_backend=plotting_backend)

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()


@pytest.mark.parametrize(*decorator_input)
def test_plot_lag_reg(plotting_backend):
    log.info(f"testing: Plotting with lagged regressors with {plotting_backend}")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_forecasts=2,
        n_lags=3,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    df["A"] = df["y"].rolling(7, min_periods=1).mean()
    df["B"] = df["y"].rolling(30, min_periods=1).mean()
    m = m.add_lagged_regressor(names="A")
    m = m.add_lagged_regressor(names="B")
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=10)
    forecast = m.predict(future)

    fig1 = m.plot_components(forecast, plotting_backend=plotting_backend)
    fig2 = m.plot(forecast, plotting_backend=plotting_backend)
    fig3 = m.plot_parameters(plotting_backend=plotting_backend)

    m.highlight_nth_step_ahead_of_each_forecast(None)
    future = m.make_future_dataframe(df, n_historic_predictions=10)
    forecast = m.predict(future)
    fig4 = m.plot_components(forecast, forecast_in_focus=2, plotting_backend=plotting_backend)

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()


@pytest.mark.parametrize(*decorator_input)
def test_plot_future_reg(plotting_backend):
    log.info(f"testing: Plotting with future regressors with {plotting_backend}")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS + 50)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    df["A"] = df["y"].rolling(7, min_periods=1).mean()
    df["B"] = df["y"].rolling(30, min_periods=1).mean()
    regressors_df_future = pd.DataFrame(data={"A": df["A"][-50:], "B": df["B"][-50:]})
    df = df[:-50]
    m = m.add_future_regressor(name="A")
    m = m.add_future_regressor(name="B", mode="multiplicative")
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df=df, regressors_df=regressors_df_future, n_historic_predictions=10, periods=50)
    forecast = m.predict(df=future)

    fig1 = m.plot(forecast, plotting_backend=plotting_backend)
    fig2 = m.plot_components(forecast, plotting_backend=plotting_backend)
    fig3 = m.plot_parameters(plotting_backend=plotting_backend)

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()


@pytest.mark.parametrize(*decorator_input)
def test_plot_uncertainty(plotting_backend):
    log.info(f"testing: Plotting with uncertainty estimation with {plotting_backend}")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)

    m = NeuralProphet(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR, quantiles=[0.25, 0.75])
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=30, n_historic_predictions=100)
    forecast = m.predict(future)
    fig1 = m.plot(forecast, plotting_backend=plotting_backend)
    fig2 = m.plot_components(forecast, plotting_backend=plotting_backend)
    fig3 = m.plot_parameters(quantile=0.75, plotting_backend=plotting_backend)

    log.info(f"testing: Plotting with uncertainty estimation for highlighted forecaste step with {plotting_backend}")
    m = NeuralProphet(
        epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR, quantiles=[0.25, 0.75], n_forecasts=7, n_lags=14
    )
    metrics_df = m.fit(df, freq="D")

    m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
    future = m.make_future_dataframe(df, periods=30, n_historic_predictions=100)
    forecast = m.predict(future)
    fig4 = m.plot(forecast, plotting_backend=plotting_backend)
    fig5 = m.plot_latest_forecast(forecast, include_previous_forecasts=10, plotting_backend=plotting_backend)
    fig6 = m.plot_components(forecast, plotting_backend=plotting_backend)
    fig7 = m.plot_parameters(quantile=0.75, plotting_backend=plotting_backend)

    log.info(f"Plot forecast parameters with wrong quantile with {plotting_backend} - Raise ValueError")
    with pytest.raises(ValueError):
        m.plot_parameters(quantile=0.8, plotting_backend=plotting_backend)
    with pytest.raises(ValueError):
        m.plot_parameters(quantile=1.1, plotting_backend=plotting_backend)

    m = NeuralProphet(
        epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR, quantiles=[0.25, 0.75], n_forecasts=3, n_lags=0
    )
    metrics_df = m.fit(df, freq="D")

    m.highlight_nth_step_ahead_of_each_forecast(None)
    future = m.make_future_dataframe(df, periods=30, n_historic_predictions=100)
    forecast = m.predict(future)
    log.info("Plot multi-steps ahead forecast without autoregression - Raise ValueError")
    with pytest.raises(ValueError):
        m.plot(forecast, plotting_backend="plotly", forecast_in_focus=4)
        m.plot_components(forecast, plotting_backend="plotly", forecast_in_focus=4)
        m.plot_components(forecast, plotting_backend="plotly", forecast_in_focus=None)
        m.plot_parameters(quantile=0.75, plotting_backend="plotly", forecast_in_focus=4)

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()
        fig5.show()
        fig6.show()
        fig7.show()


@pytest.mark.parametrize(*decorator_input)
def test_plot_conformal_prediction(plotting_backend):
    log.info(f"testing: Plotting with conformal prediction with {plotting_backend}")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    # Without auto-regression enabled
    m = NeuralProphet(
        n_forecasts=7,
        quantiles=[0.05, 0.95],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    train_df, test_df = m.split_df(df, freq="MS", valid_p=0.2)
    train_df, cal_df = m.split_df(train_df, freq="MS", valid_p=0.15)
    metrics_df = m.fit(train_df, freq="D")
    alpha = 0.1
    for method in ["naive", "cqr"]:  # Naive and CQR SCP methods
        m.conformalize(cal_df, alpha=alpha, method=method, plotting_backend=plotting_backend)
        future = m.make_future_dataframe(test_df, periods=m.n_forecasts, n_historic_predictions=10)
        forecast = m.predict(future)
        m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
        fig0 = m.plot(forecast, plotting_backend="matplotlib")
        fig1 = m.plot_components(forecast, plotting_backend="matplotlib")
        fig2 = m.plot_parameters(plotting_backend="matplotlib")
        if PLOT:
            fig0.show()
            fig1.show()
            fig2.show()
    # With auto-regression enabled
    # TO-DO: Fix Assertion error n_train >= 1
    # m = NeuralProphet(
    #     n_forecasts=7,
    #     n_lags=14,
    #     quantiles=[0.05, 0.95],
    #     epochs=EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     learning_rate=LR,
    # )
    # train_df, test_df = m.split_df(df, freq="MS", valid_p=0.2)
    # train_df, cal_df = m.split_df(train_df, freq="MS", valid_p=0.15)
    # metrics_df = m.fit(train_df, freq="D")
    # alpha = 0.1
    # for method in ["naive", "cqr"]:  # Naive and CQR SCP methods
    #     m.conformalize(cal_df, alpha=alpha, method=method)
    #     future = m.make_future_dataframe(df, periods=m.n_forecasts, n_historic_predictions=10)
    #     forecast = m.predict(future)
    #     m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
    #     fig0 = m.plot(forecast)
    #     fig1 = m.plot_latest_forecast(forecast, include_previous_forecasts=10, plotting_backend="matplotlib")
    #     fig2 = m.plot_latest_forecast(forecast, include_previous_forecasts=10, plot_history_data=True, plotting_backend="matplotlib")
    #     fig3 = m.plot_latest_forecast(forecast, include_previous_forecasts=10, plot_history_data=False, plotting_backend="matplotlib")
    #     fig4 = m.plot_components(forecast, plotting_backend="matplotlib")
    #     fig5 = m.plot_parameters(plotting_backend="matplotlib")
    #     if PLOT:
    #         fig0.show()
    #         fig1.show()
    #         fig2.show()
    #         fig3.show()
    #         fig4.show()
    #         fig5.show()


@pytest.mark.parametrize(*decorator_input)
def test_plot_latest_forecast(plotting_backend):
    log.info(f"testing: Plotting of latest forecast with {plotting_backend}")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        n_lags=12, n_forecasts=6, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR, quantiles=[0.05, 0.95]
    )
    metrics_df = m.fit(df, freq="D")

    future = m.make_future_dataframe(df, periods=30, n_historic_predictions=100)
    forecast = m.predict(future)
    fig1 = m.plot_latest_forecast(forecast, include_previous_forecasts=10, plotting_backend=plotting_backend)
    fig2 = m.plot_latest_forecast(forecast, plotting_backend=plotting_backend)
    m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
    fig3 = m.plot_latest_forecast(forecast, include_previous_forecasts=10, plotting_backend=plotting_backend)
    fig2 = m.plot_latest_forecast(
        forecast, include_previous_forecasts=10, plot_history_data=True, plotting_backend=plotting_backend
    )
    fig3 = m.plot_latest_forecast(
        forecast, include_previous_forecasts=10, plot_history_data=False, plotting_backend=plotting_backend
    )

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()

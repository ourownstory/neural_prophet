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


def test_plotly():
    log.info("testing: Plotting with plotly")
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
    fig1 = m.plot(forecast, plotting_backend="plotly")

    m.highlight_nth_step_ahead_of_each_forecast(None)
    future = m.make_future_dataframe(df, n_historic_predictions=10)
    forecast = m.predict(future)
    fig2 = m.plot(forecast, plotting_backend="plotly")
    if PLOT:
        fig1.show()
        fig2.show()


def test_plotly_components():
    log.info("testing: Plotting with plotly")
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

    fig1 = m.plot_components(forecast, plotting_backend="plotly")

    m.highlight_nth_step_ahead_of_each_forecast(None)
    future = m.make_future_dataframe(df, n_historic_predictions=10)
    forecast = m.predict(future)
    fig2 = m.plot_components(forecast, plotting_backend="plotly")

    if PLOT:
        fig1.show()
        fig2.show()


def test_plotly_parameters():
    log.info("testing: Plotting with plotly")
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

    fig1 = m.plot_parameters(plotting_backend="plotly")

    m.highlight_nth_step_ahead_of_each_forecast(None)
    future = m.make_future_dataframe(df, n_historic_predictions=10)
    forecast = m.predict(future)
    fig2 = m.plot_parameters(plotting_backend="plotly")

    if PLOT:
        fig1.show()
        fig2.show()


def test_plotly_global_local_parameters():
    log.info("Global Modeling + Global Normalization")
    df = pd.read_csv(PEYTON_FILE, nrows=512)
    df1_0 = df.iloc[:128, :].copy(deep=True)
    df1_0["ID"] = "df1"
    df2_0 = df.iloc[128:256, :].copy(deep=True)
    df2_0["ID"] = "df2"
    df3_0 = df.iloc[256:384, :].copy(deep=True)
    df3_0["ID"] = "df3"
    m = NeuralProphet(
        n_forecasts=2, n_lags=10, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR, trend_global_local="local"
    )
    train_df, test_df = m.split_df(pd.concat((df1_0, df2_0, df3_0)), valid_p=0.33, local_split=True)
    m.fit(train_df)
    future = m.make_future_dataframe(test_df)
    forecast = m.predict(future)

    fig1 = m.plot_parameters(df_name="df1", plotting_backend="plotly")

    if PLOT:
        fig1.show()


def test_plotly_events():
    log.info("testing: Plotting with plotly with events")
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

    fig1 = m.plot_components(forecast, plotting_backend="plotly")
    fig2 = m.plot(forecast, plotting_backend="plotly")
    fig3 = m.plot_parameters(plotting_backend="plotly")

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()


def test_plotly_trend():
    log.info("testing: Plotly with linear trend")
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
    fig1 = m.plot_parameters(plotting_backend="plotly")

    future = m.make_future_dataframe(df, periods=48, n_historic_predictions=len(df) - m.n_lags)
    forecast = m.predict(future)

    fig2 = m.plot(forecast, plotting_backend="plotly")
    fig3 = m.plot_components(forecast, plotting_backend="plotly")

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()


def test_plotly_seasonality():
    log.info("testing: Plotly with seasonality")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        yearly_seasonality=8,
        weekly_seasonality=4,
        seasonality_mode="additive",
        seasonality_reg=1,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=365, periods=365)
    forecast = m.predict(df=future)

    fig1 = m.plot_components(forecast, plotting_backend="plotly")
    fig2 = m.plot(forecast, plotting_backend="plotly")
    fig3 = m.plot_parameters(plotting_backend="plotly")

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

    fig4 = m.plot_parameters(plotting_backend="plotly")

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()


def test_plotly_daily_seasonality():
    log.info("testing: Plotly with daily seasonality")
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

    fig1 = m.plot_components(forecast, plotting_backend="plotly")
    fig2 = m.plot(forecast, plotting_backend="plotly")
    fig3 = m.plot_parameters(plotting_backend="plotly")

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()


def test_plotly_lag_reg():
    log.info("testing: Plotly with lagged regressors")
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

    fig1 = m.plot_components(forecast, plotting_backend="plotly")
    fig2 = m.plot(forecast, plotting_backend="plotly")
    fig3 = m.plot_parameters(plotting_backend="plotly")

    m.highlight_nth_step_ahead_of_each_forecast(None)
    future = m.make_future_dataframe(df, n_historic_predictions=10)
    forecast = m.predict(future)
    fig4 = m.plot_components(forecast, forecast_in_focus=2, plotting_backend="plotly")
    fig5 = m.plot_components(forecast, forecast_in_focus=2, residuals=True, plotting_backend="plotly")

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()
        fig5.show()


def test_plotly_future_reg():
    log.info("testing: Plotly with future regressors")
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

    fig1 = m.plot(forecast, plotting_backend="plotly")
    fig2 = m.plot_components(forecast, plotting_backend="plotly")
    fig3 = m.plot_parameters(plotting_backend="plotly")

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()


def test_plotly_uncertainty():
    log.info("testing: Plotting with plotly")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)

    m = NeuralProphet(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR, quantiles=[0.9, 0.1])
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=30, n_historic_predictions=100)
    forecast = m.predict(future)
    fig1 = m.plot(forecast, plotting_backend="plotly")
    fig2 = m.plot_components(forecast, plotting_backend="plotly")
    fig3 = m.plot_parameters(quantile=0.9, plotting_backend="plotly")

    m = NeuralProphet(
        epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR, quantiles=[0.9, 0.1], n_forecasts=3, n_lags=7
    )
    metrics_df = m.fit(df, freq="D")

    m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
    future = m.make_future_dataframe(df, periods=30, n_historic_predictions=100)
    forecast = m.predict(future)
    fig4 = m.plot(forecast, plotting_backend="plotly")
    fig5 = m.plot_components(forecast, plotting_backend="plotly")
    fig6 = m.plot_parameters(quantile=0.9, plotting_backend="plotly")

    log.info("Plot forecast with wrong quantile - Raise ValueError")
    with pytest.raises(ValueError):
        m.plot_parameters(quantile=0.8, plotting_backend="plotly")
    with pytest.raises(ValueError):
        m.plot_parameters(quantile=1.1, plotting_backend="plotly")

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()
        fig5.show()
        fig6.show()


def test_plotly_latest_forecast():
    log.info("testing: Plotting of latest forecast with plotly")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        n_lags=12, n_forecasts=6, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR, quantiles=[0.05, 0.95]
    )
    metrics_df = m.fit(df, freq="D")

    future = m.make_future_dataframe(df, periods=30, n_historic_predictions=100)
    forecast = m.predict(future)
    fig1 = m.plot_latest_forecast(forecast, include_previous_forecasts=10, plotting_backend="plotly")
    fig2 = m.plot_latest_forecast(forecast, plotting_backend="plotly")
    m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
    fig3 = m.plot_latest_forecast(forecast, include_previous_forecasts=10, plotting_backend="plotly")

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()

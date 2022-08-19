#!/usr/bin/env python3

import pytest
import os
import pathlib
import pandas as pd
import logging

from neuralprophet import NeuralProphet

log = logging.getLogger("NP.test")
log.setLevel("WARNING")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")
NROWS = 256
EPOCHS = 2
BATCH_SIZE = 64
LR = 1.0

PLOT = False


def test_plotly():
    log.info("testing: Plotting with plotly")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        n_forecasts=7,
        n_lags=14,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
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
        n_forecasts=7,
        n_lags=14,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
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
        n_forecasts=7,
        n_lags=14,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
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
        n_lags=2,
        n_forecasts=30,
        daily_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
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
    log.debug("Event Parameters:: {}".format(m.model.event_params))

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
        n_changepoints=0,
        yearly_seasonality=2,
        seasonality_mode="multiplicative",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
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
        yearly_seasonality=8,
        weekly_seasonality=4,
        seasonality_mode="additive",
        seasonality_reg=1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=365, periods=365)
    forecast = m.predict(df=future)

    fig1 = m.plot_components(forecast, plotting_backend="plotly")
    fig2 = m.plot(forecast, plotting_backend="plotly")
    fig3 = m.plot_parameters(plotting_backend="plotly")

    other_seasons = False
    m = NeuralProphet(
        yearly_seasonality=other_seasons,
        weekly_seasonality=other_seasons,
        daily_seasonality=other_seasons,
        seasonality_mode="additive",
        seasonality_reg=1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
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
        n_forecasts=2,
        n_lags=3,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
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
    m = NeuralProphet(quantiles=[0.9, 0.2, 0.1])
    metrics_df = m.fit(df, freq="D")

    future = m.make_future_dataframe(df, periods=30, n_historic_predictions=100)
    forecast = m.predict(future)
    fig1 = m.plot(forecast, plotting_backend="plotly")

    if PLOT:
        fig1.show()

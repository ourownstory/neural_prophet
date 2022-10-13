#!/usr/bin/env python3

import pytest
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
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
PLOT = False


def test_peyton_manning():
    log.info("TEST Peyton Manning")
    df = pd.read_csv(PEYTON_FILE)
    m = NeuralProphet()
    metrics = m.fit(df)
    future = m.make_future_dataframe(df, periods=int(len(df) * 0.1), n_historic_predictions=True)
    forecast = m.predict(future)

    # Accuracy
    accuracy_metrics = metrics.to_dict("records")[0]
    log.info(accuracy_metrics)
    assert accuracy_metrics["MAE"] < 500.0
    assert accuracy_metrics["RMSE"] < 500.0
    assert accuracy_metrics["Loss"] < 1.0

    if PLOT:
        m.plot(forecast)
        m.plot_parameters()
        plt.show()


def test_yosemite():
    log.info("TEST Yosemite Temps")
    df = pd.read_csv(YOS_FILE)
    m = NeuralProphet(
        changepoints_range=0.95, n_changepoints=15, weekly_seasonality=False, epochs=50, learning_rate=0.02
    )
    metrics = m.fit(df, freq="5min")
    future = m.make_future_dataframe(df, periods=int(len(df) * 0.1), n_historic_predictions=True)
    forecast = m.predict(future)

    # Accuracy
    accuracy_metrics = metrics.to_dict("records")[0]
    log.info(accuracy_metrics)
    assert accuracy_metrics["MAE"] < 200.0
    assert accuracy_metrics["RMSE"] < 200.0
    assert accuracy_metrics["Loss"] < 0.5

    if PLOT:
        m.plot(forecast)
        m.plot_parameters()
        plt.show()


def test_air_passengers():
    log.info("TEST Air Passengers")
    df = pd.read_csv(AIR_FILE)
    m = NeuralProphet()
    metrics = m.fit(df)
    future = m.make_future_dataframe(df, periods=int(len(df) * 0.1), n_historic_predictions=True)
    forecast = m.predict(future)

    # Accuracy
    accuracy_metrics = metrics.to_dict("records")[0]
    log.info(accuracy_metrics)
    assert accuracy_metrics["MAE"] < 500.0
    assert accuracy_metrics["RMSE"] < 800.0
    assert accuracy_metrics["Loss"] < 1.0

    if PLOT:
        m.plot(forecast)
        m.plot_parameters()
        plt.show()

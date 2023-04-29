#!/usr/bin/env python3

import logging
import os
import pathlib

import pandas as pd
import pytest

from neuralprophet import NeuralProphet, load, save

log = logging.getLogger("NP.test")
log.setLevel("DEBUG")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")
NROWS = 512
EPOCHS = 10
LR = 1.0
BATCH_SIZE = 64

PLOT = False


def test_create_dummy_datestamps():
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    df_drop = df.drop("ds", axis=1)

    m = NeuralProphet(quantiles=[0.02, 0.98], epochs=10, weekly_seasonality=True)
    _ = m.fit(df_drop, freq="S")

    future = m.make_future_dataframe(df_drop, periods=365, n_historic_predictions=True)
    forecast = m.predict(future)


def test_save_load():
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_lags=6,
        n_forecasts=3,
        n_changepoints=0,
    )
    _ = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=3)
    forecast = m.predict(df=future)
    log.info("testing: save")
    save(m, "test_model.pt")

    log.info("testing: load")
    m2 = load("test_model.pt")
    forecast2 = m2.predict(df=future)

    # Check that the forecasts are the same
    pd.testing.assert_frame_equal(forecast, forecast2)


# TODO: add functionality to continue training
# def test_continue_training():
#     df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
#     m = NeuralProphet(
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         learning_rate=LR,
#         n_lags=6,
#         n_forecasts=3,
#         n_changepoints=0,
#     )
#     metrics = m.fit(df, freq="D")
#     metrics2 = m.fit(df, freq="D", continue_training=True)
#     assert metrics1["Loss"].sum() >= metrics2["Loss"].sum()

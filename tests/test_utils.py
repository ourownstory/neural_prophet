#!/usr/bin/env python3

import io
import logging
import os
import pathlib

import pandas as pd
import pytest

from neuralprophet import NeuralProphet, df_utils, load, save

log = logging.getLogger("NP.test")
log.setLevel("ERROR")
log.parent.setLevel("ERROR")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")
NROWS = 512
EPOCHS = 10
ADDITIONAL_EPOCHS = 5
LR = 1.0
BATCH_SIZE = 64

PLOT = False


def test_create_dummy_datestamps():
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    df_drop = df.drop("ds", axis=1)
    df_dummy = df_utils.create_dummy_datestamps(df_drop)
    df["ds"] = pd.NA
    with pytest.raises(ValueError):
        _ = df_utils.create_dummy_datestamps(df)

    m = NeuralProphet(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR)
    _ = m.fit(df_dummy)
    _ = m.make_future_dataframe(df_dummy, periods=365, n_historic_predictions=True)


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

    m3 = load("test_model.pt", map_location="cpu")
    forecast3 = m3.predict(df=future)

    # Check that the forecasts are the same
    pd.testing.assert_frame_equal(forecast, forecast2)
    pd.testing.assert_frame_equal(forecast, forecast3)


def test_save_load_io():
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

    # Save the model to an in-memory buffer
    log.info("testing: save to buffer")
    buffer = io.BytesIO()
    save(m, buffer)
    buffer.seek(0)  # Reset buffer position to the beginning

    log.info("testing: load from buffer")
    m2 = load(buffer)
    forecast2 = m2.predict(df=future)

    buffer.seek(0)  # Reset buffer position to the beginning for another load
    m3 = load(buffer, map_location="cpu")
    forecast3 = m3.predict(df=future)

    # Check that the forecasts are the same
    pd.testing.assert_frame_equal(forecast, forecast2)
    pd.testing.assert_frame_equal(forecast, forecast3)


def test_continue_training():
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_lags=6,
        n_forecasts=3,
        n_changepoints=0,
    )
    metrics = m.fit(df, checkpointing=True, freq="D")
    metrics2 = m.fit(df, freq="D", continue_training=True, epochs=ADDITIONAL_EPOCHS)
    assert metrics["Loss"].min() >= metrics2["Loss"].min()

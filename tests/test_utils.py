#!/usr/bin/env python3

import logging
import os
import pathlib

import numpy as np
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
EPOCHS = 1
LR = 1.0
BATCH_SIZE = 64

PLOT = False


def test_save_load():
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_lags=6,
        n_forecasts=3,
    )
    _ = m.fit(df, freq="D")
    log.info("testing: save")
    save(m, "test_save_model.np")
    log.info("testing: load")
    m2 = load("test_save_model.np")
    future = m2.make_future_dataframe(df, periods=3)
    forecast = m2.predict(df=future)

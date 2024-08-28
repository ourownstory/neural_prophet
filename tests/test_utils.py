#!/usr/bin/env python3

import logging
import os
import pathlib

import pandas as pd
import pytest

from neuralprophet import NeuralProphet, df_utils

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

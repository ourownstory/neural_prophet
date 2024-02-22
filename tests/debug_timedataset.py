#!/usr/bin/env python3

import logging
import math
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from neuralprophet import NeuralProphet, df_utils, set_random_seed
from neuralprophet.data.process import _handle_missing_data, _validate_column_name

log = logging.getLogger("NP.test")
log.setLevel("ERROR")
log.parent.setLevel("ERROR")

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


def test_selective_forecasting():
    log.info("testing: selective forecasting with matching n_forecasts and prediction_frequency")
    start_date = "2019-01-01"
    end_date = "2019-03-01"
    date_range = pd.date_range(start=start_date, end=end_date, freq="H")
    y = np.random.randint(0, 1000, size=(len(date_range),))
    df = pd.DataFrame({"ds": date_range, "y": y})
    m = NeuralProphet(
        n_forecasts=24,
        n_lags=48,
        epochs=1,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        prediction_frequency={"daily-hour": 7},
    )
    m.fit(df, freq="H")
    m.predict(df)

    log.info("testing: selective forecasting with n_forecasts < prediction_frequency with lags")
    start_date = "2019-01-01"
    end_date = "2019-03-01"
    date_range = pd.date_range(start=start_date, end=end_date, freq="H")
    y = np.random.randint(0, 1000, size=(len(date_range),))
    df = pd.DataFrame({"ds": date_range, "y": y})
    m = NeuralProphet(
        n_forecasts=1,
        n_lags=14,
        epochs=1,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        prediction_frequency={"daily-hour": 7},
    )
    m.fit(df, freq="H")
    m.predict(df)
    log.info("testing: selective forecasting with n_forecasts > prediction_frequency")
    start_date = "2019-01-01"
    end_date = "2021-03-01"
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    y = np.random.randint(0, 1000, size=(len(date_range),))
    df = pd.DataFrame({"ds": date_range, "y": y})
    m = NeuralProphet(
        n_forecasts=14,
        n_lags=0,
        epochs=1,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        prediction_frequency={"weekly-day": 4},
    )
    m.fit(df, freq="D")
    m.predict(df)
    log.info("testing: selective forecasting with n_forecasts < prediction_frequency")
    start_date = "2010-01-01"
    end_date = "2020-03-01"
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS")
    y = np.random.randint(0, 1000, size=(len(date_range),))
    df = pd.DataFrame({"ds": date_range, "y": y})
    m = NeuralProphet(
        n_forecasts=1,
        n_lags=0,
        epochs=1,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        prediction_frequency={"yearly-month": 10},
    )
    m.fit(df, freq="MS")
    m.predict(df)
    log.info("testing: selective forecasting with n_forecasts < prediction_frequency")
    start_date = "2020-01-01"
    end_date = "2020-02-01"
    date_range = pd.date_range(start=start_date, end=end_date, freq="1min")
    y = np.random.randint(0, 1000, size=(len(date_range),))
    df = pd.DataFrame({"ds": date_range, "y": y})
    m = NeuralProphet(
        n_forecasts=7,
        n_lags=14,
        epochs=1,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        prediction_frequency={"hourly-minute": 23},
    )
    m.fit(df, freq="1min")
    m.predict(df)
    start_date = "2019-01-01"
    end_date = "2021-03-01"
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    y = np.random.randint(0, 1000, size=(len(date_range),))
    df = pd.DataFrame({"ds": date_range, "y": y})
    m = NeuralProphet(
        n_forecasts=14,
        n_lags=14,
        epochs=1,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        prediction_frequency={"monthly-day": 4},
    )
    m.fit(df, freq="D")
    m.predict(df)
    log.info("testing: selective forecasting with combined prediction_frequency")
    start_date = "2019-01-01"
    end_date = "2020-03-01"
    date_range = pd.date_range(start=start_date, end=end_date, freq="H")
    y = np.random.randint(0, 1000, size=(len(date_range),))
    df = pd.DataFrame({"ds": date_range, "y": y})
    m = NeuralProphet(
        n_forecasts=14,
        n_lags=14,
        epochs=1,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        prediction_frequency={"daily-hour": 14, "weekly-day": 2},
    )
    m.fit(df, freq="H")
    m.predict(df)


test_selective_forecasting()

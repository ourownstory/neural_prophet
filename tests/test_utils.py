#!/usr/bin/env python3
from collections import OrderedDict

import pytest
import os
import pathlib
import pandas as pd
import numpy as np
import logging
from neuralprophet import NeuralProphet, configure, time_net
from neuralprophet import save, load
import torch
import torch.nn as nn

log = logging.getLogger("NP.test")
log.setLevel("WARNING")
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


def test_get_set_simple_params():
    m = NeuralProphet()
    m.set_params(
        name="new name",
        n_forecasts=5,
        n_lags=4,
        max_lags=6,
        fitted=True,
        predict_steps=5,
    )
    a = m.get_params()
    assert a["name"] == "new name"
    assert a["n_forecasts"] == 5
    assert a["n_lags"] == 4
    assert a["max_lags"] == 6
    assert a["fitted"]
    assert a["predict_steps"] == 5


def test_get_set_normalisation():
    m = NeuralProphet()
    norm = configure.Normalization(
            normalize="auto",
            global_normalization=True,
            global_time_normalization=False,
            unknown_data_normalization=False,
        )
    m.set_params(
        config_normalization=norm,
    )
    a = m.get_params()
    assert a["config_normalization"] == norm


def test_get_set_missing():
    m = NeuralProphet()
    missing = configure.MissingDataHandling(False, 5, 5, True)
    m.set_params(
        config_missing=missing,
    )
    a = m.get_params()
    assert a["config_missing"] == missing


def test_get_set_metrics():
    m = NeuralProphet()
    metrics = None
    m.set_params(
        metrics=metrics
    )
    a = m.get_params()
    assert a["metrics"] == metrics


def test_get_set_ar():
    m = NeuralProphet()
    ar = configure.AR(5, 1.2)
    m.set_params(
        config_ar=ar
    )
    a = m.get_params()
    assert a["config_ar"] == ar


def test_get_set_trend():
    m = NeuralProphet()
    trend = configure.Trend("linear", [], 0, 0, 0, 0)
    m.set_params(
        config_trend= trend,
    )
    a = m.get_params()
    assert a["config_trend"] == trend


def test_get_set_season():
    m = NeuralProphet()
    season = configure.AllSeason(
            mode="additive",
            reg_lambda=3,
            yearly_arg=True,
            weekly_arg=False,
            daily_arg=True,
        )
    m.set_params(
        config_season= season
    )
    a = m.get_params()
    assert a["config_season"] == season


def test_get_empty_model_params():
    m = NeuralProphet()
    result = m.get_model_param()
    assert result is None


def test_get_set_new_model():
    m = NeuralProphet()
    m.set_model_param(
            n_forecasts=2,
            n_lags=3,
            num_hidden_layers=0,
            d_hidden=None
        )
    a = m.get_model_param()
    assert a["n_forecasts"] == 2
    assert a["n_lags"] == 3
    assert a["num_hidden_layers"] == 0
    assert a["d_hidden"] is None


def test_get_set_model():
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_lags=6,
        n_forecasts=3,
    )
    _ = m.fit(df, freq="D")
    m.set_model_param(
            n_forecasts=2,
            n_lags=3,
            num_hidden_layers=0,
            d_hidden=None
        )
    a = m.get_model_param()
    assert a["n_forecasts"] == 2
    assert a["n_lags"] == 3
    assert a["num_hidden_layers"] == 0
    assert a["d_hidden"] is None


if __name__ == '__main__':
    test_get_set_model()
    test_get_set_season()
    test_get_set_trend()
    test_get_set_ar()
    test_get_empty_model_params()
    test_get_set_metrics()
    test_get_set_missing()
    test_get_set_new_model()
    test_get_set_normalisation()
    test_get_set_simple_params()
    test_save_load()

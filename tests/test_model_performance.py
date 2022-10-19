#!/usr/bin/env python3

import pytest
import os
import pathlib
import pandas as pd
import logging
import json

from neuralprophet import NeuralProphet, set_random_seed

log = logging.getLogger("NP.test")
log.setLevel("WARNING")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")

# Important to set seed for reproducibility
set_random_seed(42)


def test_PeytonManning():
    df = pd.read_csv(PEYTON_FILE)
    m = NeuralProphet(
        n_changepoints=30,
        changepoints_range=0.90,
        trend_reg=1,
    )
    df_train, df_test = m.split_df(df=df, freq="D", valid_p=0.2)
    metrics = m.fit(df_train, validation_df=df_test, freq="D")

    accuracy_metrics = metrics.to_dict("records")[-1]
    with open(os.path.join(DIR, "tests", "metrics", "PeytonManning.json"), "w") as outfile:
        json.dump(accuracy_metrics, outfile)


def test_YosemiteTemps():
    df = pd.read_csv(YOS_FILE)
    m = NeuralProphet(
        n_lags=24,
        n_forecasts=24,
        changepoints_range=0.95,
        n_changepoints=30,
        weekly_seasonality=False,
    )
    df_train, df_test = m.split_df(df=df, freq="5min", valid_p=0.2)
    metrics = m.fit(df_train, validation_df=df_test, freq="5min")

    accuracy_metrics = metrics.to_dict("records")[-1]
    with open(os.path.join(DIR, "tests", "metrics", "YosemiteTemps.json"), "w") as outfile:
        json.dump(accuracy_metrics, outfile)


def test_AirPassengers():
    df = pd.read_csv(AIR_FILE)
    m = NeuralProphet(seasonality_mode="multiplicative")
    df_train, df_test = m.split_df(df=df, freq="MS", valid_p=0.2)
    metrics = m.fit(df_train, validation_df=df_test, freq="MS")

    accuracy_metrics = metrics.to_dict("records")[-1]
    with open(os.path.join(DIR, "tests", "metrics", "AirPassengers.json"), "w") as outfile:
        json.dump(accuracy_metrics, outfile)

#!/usr/bin/env python3

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
DATA_DIR = os.path.join(DIR, "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")
PLOT = False

# Important to set seed for reproducibility
set_random_seed(42)


#    log.info("TEST Peyton Manning")
df = pd.read_csv(PEYTON_FILE)
m = NeuralProphet()
metrics = m.fit(df)
future = m.make_future_dataframe(df, periods=int(len(df) * 0.1), n_historic_predictions=True)
forecast = m.predict(future)

# Accuracy
accuracy_metrics = metrics.to_dict("records")[0]
with open("tests/model_tests/PeytonManning.json", "w") as outfile:
    json.dump(accuracy_metrics, outfile)


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
    with open("tests/model_tests/YosemiteTemps.json", "w") as outfile:
        json.dump(accuracy_metrics, outfile)


def test_air_passengers():
    log.info("TEST Air Passengers")
    df = pd.read_csv(AIR_FILE)
    m = NeuralProphet()
    metrics = m.fit(df)
    future = m.make_future_dataframe(df, periods=int(len(df) * 0.1), n_historic_predictions=True)
    forecast = m.predict(future)

    # Accuracy
    accuracy_metrics = metrics.to_dict("records")[0]
    with open("tests/model_tests/AirPassengers.json", "w") as outfile:
        json.dump(accuracy_metrics, outfile)

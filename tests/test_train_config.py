#!/usr/bin/env python3

import logging
import os
import pathlib

import pandas as pd

from neuralprophet import NeuralProphet

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


def generate_config_train_params(overrides={}):
    config_train_params = {
        "learning_rate": None,
        "epochs": None,
        "batch_size": None,
        "loss_func": "SmoothL1Loss",
        "optimizer": "AdamW",
    }
    for key, value in overrides.items():
        config_train_params[key] = value
    return config_train_params


def test_custom_lr_scheduler():
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)

    # Set in NeuralProphet()
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        scheduler="CosineAnnealingWarmRestarts",
        scheduler_args={"T_0": 5, "T_mult": 2},
    )
    _ = m.fit(df, freq="D")
    # Set in NeuralProphet(), no args
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        scheduler="StepLR",
    )
    _ = m.fit(df, freq="D")

    # Set in fit()
    m = NeuralProphet(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR)
    _ = m.fit(
        df,
        freq="D",
        scheduler="ExponentialLR",
        scheduler_args={"gamma": 0.95},
    )

    # Set in fit(), no args
    m = NeuralProphet(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR)
    _ = m.fit(
        df,
        freq="D",
        scheduler="OneCycleLR",
    )

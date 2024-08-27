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
    metrics = m.fit(df, freq="D")
    # Set in NeuralProphet(), no args
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        scheduler="StepLR",
    )
    metrics = m.fit(df, freq="D")

    # Set in fit()
    m = NeuralProphet(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR)
    metrics = m.fit(
        df,
        freq="D",
        scheduler="ExponentialLR",
        scheduler_args={"gamma": 0.95},
    )

    # Set in fit(), no args
    m = NeuralProphet(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR)
    metrics = m.fit(
        df,
        freq="D",
        scheduler="OneCycleLR",
    )


# def test_continue_training_checkpoint():
#     df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
#     m = NeuralProphet(
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         learning_rate=LR,
#         n_lags=6,
#         n_forecasts=3,
#         n_changepoints=0,
#     )
#     metrics = m.fit(df, checkpointing=True, freq="D")
#     metrics2 = m.fit(df, freq="D", continue_training=True, epochs=ADDITIONAL_EPOCHS)
#     assert metrics["Loss"].min() >= metrics2["Loss"].min()


# def test_continue_training_with_scheduler_selection():
#     df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
#     m = NeuralProphet(
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         learning_rate=LR,
#         n_lags=6,
#         n_forecasts=3,
#         n_changepoints=0,
#     )
#     metrics = m.fit(df, checkpointing=True, freq="D")
#     # Continue training with StepLR
#     metrics2 = m.fit(df, freq="D", continue_training=True, epochs=ADDITIONAL_EPOCHS, scheduler="StepLR")
#     assert metrics["Loss"].min() >= metrics2["Loss"].min()


# def test_save_load_continue_training():
#     df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
#     m = NeuralProphet(
#         epochs=EPOCHS,
#         n_lags=6,
#         n_forecasts=3,
#         n_changepoints=0,
#     )
#     metrics = m.fit(df, checkpointing=True, freq="D")
#     save(m, "test_model.pt")
#     m2 = load("test_model.pt")
#     metrics2 = m2.fit(df, continue_training=True, epochs=ADDITIONAL_EPOCHS, scheduler="StepLR")
#     assert metrics["Loss"].min() >= metrics2["Loss"].min()

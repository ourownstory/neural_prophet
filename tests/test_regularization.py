#!/usr/bin/env python3

import os
import pathlib
import random

import numpy as np
import pandas as pd
import pytest
import torch

from neuralprophet import NeuralProphet, df_utils
from neuralprophet.utils import reg_func_abs

# Fix random seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Variables
DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")

NROWS = 100
EPOCHS = 50
BATCH_SIZE = 64
LR = 1.0
REGULARIZATION = 1e20


def test_reg_func_abs():
    assert pytest.approx(1) == reg_func_abs(torch.Tensor([1]))
    assert pytest.approx(0) == reg_func_abs(torch.Tensor([0]))
    assert pytest.approx(1) == reg_func_abs(torch.Tensor([-1]))

    assert pytest.approx(1) == reg_func_abs(torch.Tensor([1, 1, 1]))
    assert pytest.approx(0) == reg_func_abs(torch.Tensor([0, 0, 0]))
    assert pytest.approx(1) == reg_func_abs(torch.Tensor([-1, -1, -1]))

    assert pytest.approx(0.6666666) == reg_func_abs(torch.Tensor([-1, 0, 1]))
    assert pytest.approx(20) == reg_func_abs(torch.Tensor([-12, 4, 0, -1, 1, 102]))


def test_regularization_holidays():
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    df = df_utils.check_dataframe(df, check_y=False)

    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    m = m.add_country_holidays("US", regularization=REGULARIZATION)
    m.fit(df, freq="D")

    for country_holiday in m.country_holidays_config.holiday_names:
        event_params = m.model.get_event_weights(country_holiday)
        weight_list = [param.detach().numpy() for _, param in event_params.items()]
        assert weight_list[0] < 0.05


def test_regularization_holidays_disabled():
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    df = df_utils.check_dataframe(df, check_y=False)

    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    m = m.add_country_holidays("US", regularization=0)
    m.fit(df, freq="D")

    for country_holiday in m.country_holidays_config.holiday_names:
        event_params = m.model.get_event_weights(country_holiday)
        weight_list = [param.detach().numpy() for _, param in event_params.items()]
        assert np.abs(weight_list[0]) > 0.05


def test_regularization_events():
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    df = df_utils.check_dataframe(df, check_y=False)

    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    m = m.add_events("special_day", regularization=REGULARIZATION)
    events_df = pd.DataFrame(
        {
            "event": "special_day",
            "ds": pd.to_datetime(["2008-02-04"]),
        }
    )
    history_df = m.create_df_with_events(df, events_df)
    m.fit(history_df, freq="D")

    weight_list = m.model.get_event_weights("special_day")
    for _, param in weight_list.items():
        assert param.detach().numpy() < 0.01


def test_regularization_events_disabled():
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    df = df_utils.check_dataframe(df, check_y=False)

    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    m = m.add_events("special_day", regularization=0)
    events_df = pd.DataFrame(
        {
            "event": "special_day",
            "ds": pd.to_datetime(["2008-02-04"]),
        }
    )
    history_df = m.create_df_with_events(df, events_df)
    m.fit(history_df, freq="D")

    weight_list = m.model.get_event_weights("special_day")
    for _, param in weight_list.items():
        # assert param.detach().numpy() == pytest.approx(0.9358184337615967)
        assert param.detach().numpy() > 0.9

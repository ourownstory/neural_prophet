#!/usr/bin/env python3

import random

import numpy as np
import pandas as pd
import pytest
import torch
from utils.dataset_generators import generate_event_dataset, generate_holiday_dataset

from neuralprophet import NeuralProphet, df_utils
from neuralprophet.utils import reg_func_abs

# Fix random seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Variables
EPOCHS = 10
BATCH_SIZE_HOLIDAYS = 32
BATCH_SIZE_EVENTS = 1
LEARNING_RATE = 0.1
REGULARIZATION = 0.01
# Map holiday name to a y value for dataset generation
Y_HOLIDAYS_OVERRIDE = {
    "Washington's Birthday": 10,
    "Labor Day": 10,
    "Christmas Day": 10,
}
Y_EVENTS_OVERRIDE = {
    "2022-01-13": 10,
}


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
    df = generate_holiday_dataset(y_holidays_override=Y_HOLIDAYS_OVERRIDE)
    df = df_utils.check_dataframe(df, check_y=False)

    m = NeuralProphet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        growth="off",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE_HOLIDAYS,
        learning_rate=LEARNING_RATE,
    )
    m = m.add_country_holidays("US", regularization=REGULARIZATION)
    m.fit(df, freq="D")

    for country_holiday in m.country_holidays_config.holiday_names:
        event_params = m.model.get_event_weights(country_holiday)
        weight_list = [param.detach().numpy() for _, param in event_params.items()]
        if country_holiday in Y_HOLIDAYS_OVERRIDE.keys():
            assert weight_list[0] < 0.1
        else:
            assert weight_list[0] > 0.5


def test_regularization_events():
    df, events = generate_event_dataset(y_events_override=Y_EVENTS_OVERRIDE)
    df = df_utils.check_dataframe(df, check_y=False)

    m = NeuralProphet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        growth="off",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE_EVENTS,
        learning_rate=LEARNING_RATE,
    )
    m = m.add_events(["event_%i" % index for index, _ in enumerate(events)], regularization=REGULARIZATION)
    events_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "event": "event_%i" % index,
                    "ds": pd.to_datetime([event]),
                }
            )
            for index, event in enumerate(events)
        ]
    )
    history_df = m.create_df_with_events(df, events_df)
    m.fit(history_df, freq="D")

    for index, event in enumerate(events):
        weight_list = m.model.get_event_weights("event_%i" % index)
        for _, param in weight_list.items():
            if event in Y_EVENTS_OVERRIDE.keys():
                assert param.detach().numpy() < 0.1
            else:
                assert param.detach().numpy() > 0.5

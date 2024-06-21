#!/usr/bin/env python3

import logging
import os
import pathlib

import pandas as pd

from neuralprophet import NeuralProphet

log = logging.getLogger("NP.test")
log.setLevel("DEBUG")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
TUTORIAL_FILE = "https://github.com/ourownstory/neuralprophet-data/raw/main/kaggle-energy/datasets/tutorial04.csv"
NROWS = 1028
EPOCHS = 2
BATCH_SIZE = 128
LR = 1.0

PLOT = False


def test_future_regressor_nn():
    log.info("future regressor with NN")

    df = pd.read_csv(TUTORIAL_FILE, nrows=NROWS)

    m = NeuralProphet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=True,
        future_regressors_model="neural_nets",  # 'linear' default or 'neural_nets'
        future_regressors_d_hidden=4,  # (int)
        future_regressors_num_hidden_layers=2,  # (int)
        n_forecasts=3,
        n_lags=5,
        drop_missing=True,
        # trainer_config={"accelerator": "gpu"},
    )
    df_train, df_val = m.split_df(df, freq="H", valid_p=0.2)

    # Use static plotly in notebooks
    # m.set_plotting_backend("plotly")

    # Add the new future regressor
    m.add_future_regressor("temperature")

    # Add counrty holidays
    m.add_country_holidays("IT", mode="additive", lower_window=-1, upper_window=1)

    metrics = m.fit(
        df_train, validation_df=df_val, freq="H", epochs=EPOCHS, learning_rate=LR, early_stopping=True, progress=False
    )


def test_future_regressor_nn_shared():
    log.info("future regressor with NN")

    df = pd.read_csv(TUTORIAL_FILE, nrows=NROWS)

    m = NeuralProphet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=True,
        future_regressors_model="shared_neural_nets",
        future_regressors_d_hidden=4,  # (int)
        future_regressors_num_hidden_layers=2,  # (int)
        n_forecasts=3,
        n_lags=5,
        drop_missing=True,
        # trainer_config={"accelerator": "gpu"},
    )
    df_train, df_val = m.split_df(df, freq="H", valid_p=0.2)

    # Use static plotly in notebooks
    # m.set_plotting_backend("plotly")

    # Add the new future regressor
    m.add_future_regressor("temperature")

    # Add counrty holidays
    m.add_country_holidays("IT", mode="additive", lower_window=-1, upper_window=1)

    metrics = m.fit(
        df_train, validation_df=df_val, freq="H", epochs=EPOCHS, learning_rate=LR, early_stopping=True, progress=False
    )


test_future_regressor_nn_shared()

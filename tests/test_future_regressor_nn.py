#!/usr/bin/env python3

import logging
import os
import pathlib

import pandas as pd
from matplotlib import pyplot as plt

from neuralprophet import NeuralProphet

log = logging.getLogger("NP.test")
log.setLevel("DEBUG")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")

ENERGY_TEMP_DAILY_FILE = os.path.join(DATA_DIR, "tutorial04_kaggle_energy_daily_temperature.csv")
NROWS = 512
EPOCHS = 2
BATCH_SIZE = 128
LR = 1.0

PLOT = False


def test_future_reg_nn():
    log.info("testing: Future Regressors modelled with NNs")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS + 50)
    m = NeuralProphet(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR, future_regressors_model="neural_nets")
    df["A"] = df["y"].rolling(7, min_periods=1).mean()
    df["B"] = df["y"].rolling(30, min_periods=1).mean()
    df["C"] = df["y"].rolling(7, min_periods=1).mean()
    df["D"] = df["y"].rolling(30, min_periods=1).mean()

    regressors_df_future = pd.DataFrame(
        data={"A": df["A"][-50:], "B": df["B"][-50:], "C": df["C"][-50:], "D": df["D"][-50:]}
    )
    df = df[:-50]
    m = m.add_future_regressor(name="A")
    m = m.add_future_regressor(name="B", mode="additive")
    m = m.add_future_regressor(name="C", mode="multiplicative")
    m = m.add_future_regressor(name="D", mode="multiplicative")
    m.fit(df, freq="D")
    future = m.make_future_dataframe(df=df, regressors_df=regressors_df_future, n_historic_predictions=10, periods=50)
    forecast = m.predict(df=future)
    if PLOT:
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_future_reg_nn_shared():
    log.info("testing: Future Regressors modelled with NNs shared")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS + 50)
    m = NeuralProphet(
        epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR, future_regressors_model="shared_neural_nets"
    )
    df["A"] = df["y"].rolling(7, min_periods=1).mean()
    df["B"] = df["y"].rolling(30, min_periods=1).mean()
    df["C"] = df["y"].rolling(7, min_periods=1).mean()
    df["D"] = df["y"].rolling(30, min_periods=1).mean()

    regressors_df_future = pd.DataFrame(
        data={"A": df["A"][-50:], "B": df["B"][-50:], "C": df["C"][-50:], "D": df["D"][-50:]}
    )
    df = df[:-50]
    m = m.add_future_regressor(name="A")
    m = m.add_future_regressor(name="B", mode="additive")
    m = m.add_future_regressor(name="C", mode="multiplicative")
    m = m.add_future_regressor(name="D", mode="multiplicative")
    m.fit(df, freq="D")
    future = m.make_future_dataframe(df=df, regressors_df=regressors_df_future, n_historic_predictions=10, periods=50)
    forecast = m.predict(df=future)
    if PLOT:
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_future_reg_nn_shared_coef():
    log.info("testing: Future Regressors modelled with NNs shared coef")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS + 50)
    m = NeuralProphet(
        epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR, future_regressors_model="shared_neural_nets_coef"
    )
    df["A"] = df["y"].rolling(7, min_periods=1).mean()
    df["B"] = df["y"].rolling(30, min_periods=1).mean()
    df["C"] = df["y"].rolling(7, min_periods=1).mean()
    df["D"] = df["y"].rolling(30, min_periods=1).mean()

    regressors_df_future = pd.DataFrame(
        data={"A": df["A"][-50:], "B": df["B"][-50:], "C": df["C"][-50:], "D": df["D"][-50:]}
    )
    df = df[:-50]
    m = m.add_future_regressor(name="A")
    m = m.add_future_regressor(name="B", mode="additive")
    m = m.add_future_regressor(name="C", mode="multiplicative")
    m = m.add_future_regressor(name="D", mode="multiplicative")
    m.fit(df, freq="D")
    future = m.make_future_dataframe(df=df, regressors_df=regressors_df_future, n_historic_predictions=10, periods=50)
    forecast = m.predict(df=future)
    if PLOT:
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_future_regressor_nn_2():
    log.info("future regressor with NN")

    df = pd.read_csv(ENERGY_TEMP_DAILY_FILE, nrows=NROWS)

    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=True,
        future_regressors_model="neural_nets",  # 'linear' default or 'neural_nets'
        future_regressors_layers=[4, 4],
        n_forecasts=3,
        n_lags=5,
        drop_missing=True,
        # trainer_config={"accelerator": "gpu"},
    )
    df_train, df_val = m.split_df(df, freq="D", valid_p=0.2)

    # Use static plotly in notebooks
    # m.set_plotting_backend("plotly")

    # Add the new future regressor
    m.add_future_regressor("temperature")

    # Add counrty holidays
    m.add_country_holidays("IT", mode="additive", lower_window=-1, upper_window=1)

    metrics = m.fit(
        df_train, validation_df=df_val, freq="D", epochs=EPOCHS, learning_rate=LR, early_stopping=True, progress=False
    )
    log.debug(f"Metrics: {metrics}")


def test_future_regressor_nn_shared_2():
    log.info("future regressor with NN shared 2")

    df = pd.read_csv(ENERGY_TEMP_DAILY_FILE, nrows=NROWS)

    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=True,
        future_regressors_model="shared_neural_nets",
        future_regressors_layers=[4, 4],
        n_forecasts=3,
        n_lags=5,
        drop_missing=True,
    )
    df_train, df_val = m.split_df(df, freq="D", valid_p=0.2)

    # Add the new future regressor
    m.add_future_regressor("temperature")

    metrics = m.fit(
        df_train, validation_df=df_val, freq="D", epochs=EPOCHS, learning_rate=LR, early_stopping=True, progress=False
    )
    log.debug(f"Metrics: {metrics}")


# def test_future_regressor_nn_shared_coef_2():
#     log.info("future regressor with NN shared coef 2")
#     df = pd.read_csv(ENERGY_TEMP_DAILY_FILE, nrows=NROWS)
#     m = NeuralProphet(
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         learning_rate=LR,
#         yearly_seasonality=False,
#         weekly_seasonality=False,
#         daily_seasonality=True,
#         future_regressors_model="shared_neural_nets_coef",
#         future_regressors_layers=[4, 4],
#         n_forecasts=3,
#         n_lags=5,
#         drop_missing=True,
#     )
#     df_train, df_val = m.split_df(df, freq="D", valid_p=0.2)

#     # Add the new future regressor
#     m.add_future_regressor("temperature")

#     metrics = m.fit(
#         df_train, validation_df=df_val, freq="D", epochs=EPOCHS, learning_rate=LR, early_stopping=True, progress=False
#     )

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
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
ENERGY_TEMP_DAILY_FILE = os.path.join(DATA_DIR, "tutorial04_kaggle_energy_daily_temperature.csv")
NROWS = 512
EPOCHS = 2
BATCH_SIZE = 128
LR = 1.0


def test_future_regressor_coefficients_nn():
    log.info("Testing: Future Regressor Coefficients with NNs")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS + 50)
    m = NeuralProphet(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR, future_regressors_model="neural_nets")
    df["A"] = df["y"].rolling(7, min_periods=1).mean()
    df["B"] = df["y"].rolling(30, min_periods=1).mean()
    regressors_df_future = pd.DataFrame(data={"A": df["A"][-50:], "B": df["B"][-50:]})
    df = df[:-50]
    m = m.add_future_regressor(name="A")
    m = m.add_future_regressor(name="B", mode="additive")
    m.fit(df, freq="D")
    coefficients = m.model.get_future_regressor_coefficients()
    log.info(coefficients)
    assert not coefficients.empty, "No coefficients found"
    assert "regressor" in coefficients.columns, "Regressor column missing"
    assert "regressor_mode" in coefficients.columns, "Regressor mode column missing"
    assert "coef" in coefficients.columns, "Coefficient column missing"


def test_event_regressor_coefficients():
    log.info("Testing: Event Regressor Coefficients")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR)
    m = m.add_country_holidays("US")
    m.fit(df, freq="D")
    coefficients = m.model.get_event_coefficients()
    log.info(coefficients)
    assert not coefficients.empty, "No coefficients found"
    assert "regressor" in coefficients.columns, "Regressor column missing"
    assert "regressor_mode" in coefficients.columns, "Regressor mode column missing"
    assert "coef" in coefficients.columns, "Coefficient column missing"
    assert len(coefficients) == 10, f"Incorrect number of coefficients found: {len(coefficients)}"


def test_lagged_regressor_coefficients():
    log.info("Testing: Lagged Regressor Coefficients")
    df = pd.read_csv(ENERGY_TEMP_DAILY_FILE, nrows=NROWS)
    m = NeuralProphet(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR)
    m.add_lagged_regressor("temperature", n_lags=3)
    m.fit(df, freq="D")
    coefficients = m.model.get_lagged_regressor_coefficients()
    log.info(coefficients)
    assert not coefficients.empty, "No coefficients found"
    assert "regressor" in coefficients.columns, "Regressor column missing"
    assert "lag" in coefficients.columns, "Lag column missing"
    assert "coef" in coefficients.columns, "Coefficient column missing"
    assert len(coefficients) == 3, "Incorrect number of lagged coefficients"


def test_ar_coefficients():
    log.info("Testing: AR Coefficients")
    df = pd.read_csv(ENERGY_TEMP_DAILY_FILE, nrows=NROWS)
    m = NeuralProphet(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR, n_lags=10)
    m.add_lagged_regressor("temperature")
    m.fit(df, freq="D")
    coefficients = m.model.get_ar_coefficients()
    log.info(coefficients)
    assert not coefficients.empty, "No coefficients found"
    assert "regressor" in coefficients.columns, "Regressor column missing"
    assert "lag" in coefficients.columns, "Lag column missing"
    assert "coef" in coefficients.columns, "Coefficient column missing"
    assert len(coefficients) == 10, "Incorrect number of lagged coefficients"

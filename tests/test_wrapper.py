#!/usr/bin/env python3

import pytest
import os
import pathlib
import pandas as pd
import logging

from neuralprophet import Prophet

log = logging.getLogger("NP.test")
log.setLevel("WARNING")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
NROWS = 256

PLOT = False


def test_wrapper_base():
    log.info("testing: Wrapper base")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=50)
    forecast = m.predict(future)


def test_wrapper_components():
    log.info("testing: Wrapper components")
    df = pd.read_csv(AIR_FILE, nrows=NROWS)
    df["regressor"] = df["y"].rolling(7, min_periods=1).mean()
    regressors_df_future = pd.DataFrame(data={"regressor": df["regressor"][-50:]})

    m = Prophet(seasonality_mode="multiplicative")

    m.add_seasonality("quarterly", period=91.25, fourier_order=8, mode="additive")
    m.add_regressor("regressor", mode="additive")
    m.fit(df)

    future = m.make_future_dataframe(periods=50, regressors_df=regressors_df_future)
    forecast = m.predict(future)


def test_wrapper_plots():
    log.info("testing: Wrapper plots")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=50)
    forecast = m.predict(future)

    fig1 = m.plot(forecast)
    fig2 = m.plot(forecast, plotting_backend="plotly")
    fig3 = m.plot_components(forecast)
    fig4 = m.plot_components(forecast, plotting_backend="plotly")

    if PLOT:
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()

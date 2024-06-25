#!/usr/bin/env python3

import logging
import os
import pathlib

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from holidays import country_holidays

from neuralprophet import NeuralProphet, event_utils

log = logging.getLogger("NP.test")
log.setLevel("ERROR")
log.parent.setLevel("ERROR")


DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")
NROWS = 256
EPOCHS = 1
BATCH_SIZE = 128
LR = 1.0

PLOT = False


def test_get_country_holidays():
    # deprecated
    # assert issubclass(event_utils.get_country_holidays("TU").__class__, holidays.countries.turkey.TR) is True
    # new format
    assert issubclass(event_utils.get_all_holidays(country=["TU", "US"], years=2025).__class__, dict) is True

    for country in ("UnitedStates", "US", "USA"):
        us_holidays = event_utils.get_all_holidays(country=country, years=[2019, 2020])
        assert len(us_holidays) == 10

    with pytest.raises(NotImplementedError):
        event_utils.get_holiday_names("NotSupportedCountry")


def test_get_country_holidays_with_subdivisions():
    # Test US holidays with a subdivision
    us_ca_holidays = country_holidays("US", years=2019, subdiv="CA")
    assert issubclass(us_ca_holidays.__class__, holidays.countries.united_states.UnitedStates) is True
    assert len(us_ca_holidays) > 0  # Assuming there are holidays specific to CA

    # Test Canada holidays with a subdivision
    ca_on_holidays = country_holidays("CA", years=2019, subdiv="ON")
    assert issubclass(ca_on_holidays.__class__, holidays.countries.canada.CA) is True
    assert len(ca_on_holidays) > 0  # Assuming there are holidays specific to ON


def test_add_country_holiday_multiple_calls_warning(caplog):
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    m.add_country_holidays(["US", "Germany"])
    error_message = "Country holidays can only be added once."
    assert error_message not in caplog.text

    with pytest.raises(AssertionError):
        m.add_country_holidays("Germany")
        # assert error_message in caplog.text


def test_multiple_countries():
    # test if multiple countries are added
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    m.add_country_holidays(country_name=["US", "Germany"])
    m.fit(df, freq="D")
    m.predict(df)
    # get the name of holidays and compare that no holiday is repeated
    holiday_names = m.model.config_holidays.holiday_names
    assert "Independence Day" in holiday_names
    assert "Christmas Day" in holiday_names
    assert "Erster Weihnachtstag" not in holiday_names
    assert "Neujahr" not in holiday_names


def test_events():
    log.info("testing: Events")
    df = pd.read_csv(PEYTON_FILE)[-NROWS:]
    playoffs = pd.DataFrame(
        {
            "event": "playoff",
            "ds": pd.to_datetime(
                [
                    "2008-01-13",
                    "2009-01-03",
                    "2010-01-16",
                    "2010-01-24",
                    "2010-02-07",
                    "2011-01-08",
                    "2013-01-12",
                    "2014-01-12",
                    "2014-01-19",
                    "2014-02-02",
                    "2015-01-11",
                    "2016-01-17",
                    "2016-01-24",
                    "2016-02-07",
                ]
            ),
        }
    )
    superbowls = pd.DataFrame(
        {
            "event": "superbowl",
            "ds": pd.to_datetime(["2010-02-07", "2014-02-02", "2016-02-07"]),
        }
    )
    events_df = pd.concat((playoffs, superbowls))
    m = NeuralProphet(
        n_lags=2,
        n_forecasts=30,
        daily_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    # set event windows
    m = m.add_events(
        ["superbowl", "playoff"], lower_window=-1, upper_window=1, mode="multiplicative", regularization=0.5
    )
    # add the country specific holidays
    m = m.add_country_holidays(
        ["US", "Indonesia", "Philippines", "Pakistan", "Belarus"], mode="additive", regularization=0.5
    )
    # m.add_country_holidays("Thailand") # holidays package has issue with int input for timedelta. accepts np.float64()
    history_df = m.create_df_with_events(df, events_df)
    m.fit(history_df, freq="D")
    future = m.make_future_dataframe(df=history_df, events_df=events_df, periods=30, n_historic_predictions=90)
    forecast = m.predict(df=future)
    log.debug(f"Event Parameters:: {m.model.event_params}")
    if PLOT:
        m.plot_components(forecast)
        m.plot(forecast)
        m.plot_parameters()
        plt.show()

import numpy as np
import pandas as pd

from neuralprophet.time_dataset import make_country_specific_holidays_df


def generate_holiday_dataset(country="US", years=[2022], y_default=1, y_holiday=100, y_holidays_override={}):
    """Generate dataset with special y values for country holidays."""

    periods = len(years) * 365
    dates = pd.date_range("%i-01-01" % (years[0]), periods=periods, freq="D")
    df = pd.DataFrame({"ds": dates, "y": y_default}, index=dates)

    holidays = make_country_specific_holidays_df(years, country)
    for holiday_name, timestamps in holidays.items():
        df.loc[timestamps[0], "y"] = y_holidays_override.get(holiday_name, y_holiday)

    return df


def generate_event_dataset(
    events=["2022-01-01", "2022-01-10", "2022-01-13", "2022-01-14", "2022-01-15", "2022-01-31"],
    periods=31,
    y_default=1,
    y_event=100,
    y_events_override={},
):
    """Generate dataset with regular y value and special y value for events."""
    events.sort()

    dates = pd.date_range(events[0], periods=periods, freq="D")
    df = pd.DataFrame({"ds": dates, "y": y_default}, index=dates)

    for event in events:
        df.loc[event, "y"] = y_events_override.get(event, y_event)

    return df, events


def generate_lagged_regressor_dataset(periods=31):
    """
    Generate dataset for tests on lagged regressor.
    Columns are: ds, lagged_regressors (one entry each), y
    Each lagged regressor is random noise (range 0 to 1).
    y is a weighted sum of the the previous 3 lagged regressors.
    """
    lagged_regressors = [("a", 1), ("b", 0.1), ("c", 0.1), ("d", 1)]

    dates = pd.date_range("2022-01-01", periods=periods, freq="D")

    df = pd.DataFrame({"ds": dates}, index=dates)

    for lagged_regressor, _ in lagged_regressors:
        df[lagged_regressor] = np.random.random(periods)

    df["weighted_sum"] = sum(
        df[lagged_regressor] * lagged_regressor_scale for lagged_regressor, lagged_regressor_scale in lagged_regressors
    )
    df["y"] = 0

    overlap = 3

    for pos, (index, data) in enumerate(df.iterrows()):
        if pos >= overlap:
            df.loc[index, "y"] = sum([df.iloc[pos - lag - 1]["weighted_sum"] for lag in range(overlap)])

    df = df.drop(columns=["weighted_sum"])

    return df, lagged_regressors

import pandas as pd

from neuralprophet.time_dataset import make_country_specific_holidays_df


def generate_holiday_dataset(country="US", years=[2022], y_default=1, y_holiday=1000, y_holidays_override={}):
    """Generate dataset with special y values for country holidays."""

    periods = len(years) * 365
    dates = pd.date_range("%i-01-01" % (years[0]), periods=periods, freq="D")
    df = pd.DataFrame({"ds": dates, "y": y_default}, index=dates)

    holidays = make_country_specific_holidays_df(years, country)
    for holiday_name, timestamps in holidays.items():
        df.loc[timestamps[0], "y"] = y_holidays_override.get(holiday_name, y_holiday)

    return df


def generate_event_dataset(events=["2022-01-01", "2022-01-10", "2022-01-31"], periods=31, y_default=1, y_event=1000):
    """Generate dataset with regular y value and special y value for events."""
    events.sort()

    dates = pd.date_range(events[0], periods=periods, freq="D")
    df = pd.DataFrame({"ds": dates, "y": y_default}, index=dates)

    for event in events:
        df.loc[event, "y"] = y_event

    return df, events

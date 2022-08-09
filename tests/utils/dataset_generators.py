import pandas as pd

from neuralprophet.time_dataset import make_country_specific_holidays_df


def generate_holiday_dataset(country="US", years=[2022], y_default=1, y_holiday=1000):
    """Generate dataset with special y values for country holidays."""

    periods = len(years) * 365
    dates = pd.date_range("%i-01-01" % (years[0]), periods=periods, freq="D")
    df = pd.DataFrame({"ds": dates, "y": y_default}, index=dates)

    holidays = make_country_specific_holidays_df(years, country)
    for holiday, timestamps in holidays.items():
        df.loc[timestamps[0], "y"] = y_holiday

    return df

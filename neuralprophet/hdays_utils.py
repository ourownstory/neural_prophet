from collections import defaultdict
from typing import Iterable, Optional, Union

import holidays
import numpy as np
import pandas as pd


def get_country_holidays(
    country: str, years: Optional[Union[int, Iterable[int]]] = None, subdivision: Optional[str] = None
):
    """
    Helper function to get holidays for a country.

    Parameters
    ----------
        country : str
            Country name to retrieve country specific holidays
        years : int, list
            Year or list of years to retrieve holidays for
        subdivision : str
            Subdivision name to retrieve subdivision specific holidays

    Returns
    -------
        set
            All possible holiday dates and names of given country

    """
    substitutions = {
        "TU": "TR",  # For compatibility with Turkey as "TU" cases.
    }

    country = substitutions.get(country, country)
    if not hasattr(holidays, country):
        raise AttributeError(f"Holidays in {country} are not currently supported!")
    if subdivision:
        holiday_obj = getattr(holidays, country)(years=years, subdiv=subdivision)
    else:
        holiday_obj = getattr(holidays, country)(years=years)

    return holiday_obj


def get_holidays_from_country(country: Union[str, Iterable[str], dict], df=None):
    """
    Return all possible holiday names of given countries

    Parameters
    ----------
        country : str, list
            List of country names to retrieve country specific holidays
        subdivision : str, dict
            a single subdivision (e.g., province or state) as a string or
            a dictionary where the key is the country name and the value is a subdivision
        df : pd.Dataframe
            Dataframe from which datestamps will be retrieved from

    Returns
    -------
        set
            All possible holiday names of given country
    """
    if df is None:
        years = np.arange(1995, 2045)
    else:
        dates = df["ds"].copy(deep=True)
        years = list({x.year for x in dates})
    # support multiple countries
    if isinstance(country, str):
        country = {country: None}
    elif isinstance(country, list):
        country = dict(zip(country, [None] * len(country)))

    unique_holidays = {}
    for single_country, subdivision in country.items():
        holidays_country = get_country_holidays(single_country, years, subdivision)
        for date, name in holidays_country.items():
            if date not in unique_holidays:
                unique_holidays[date] = name
    holiday_names = unique_holidays.values()

    return set(holiday_names)


def make_country_specific_holidays(year_list, country):
    """
    Create dict of holiday names and dates for given years and countries
    Parameters
    ----------
        year_list : list
            List of years
        country : str, list, dict
            List of country names and optional subdivisions
    Returns
    -------
        dict
            holiday names as keys and dates as values
    """
    # iterate over countries and get holidays for each country

    if isinstance(country, str):
        country = {country: None}
    elif isinstance(country, list):
        country = dict(zip(country, [None] * len(country)))

    country_specific_holidays = {}
    for single_country, subdivision in country.items():
        single_country_specific_holidays = get_country_holidays(single_country, year_list, subdivision)
        # only add holiday if it is not already in the dict
        country_specific_holidays.update(single_country_specific_holidays)
    holidays_dates = defaultdict(list)
    for date, holiday in country_specific_holidays.items():
        holidays_dates[holiday].append(pd.to_datetime(date))
    return holidays_dates

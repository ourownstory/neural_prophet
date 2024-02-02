from collections import defaultdict
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from holidays import country_holidays

# def get_country_holidays(country: str, years: Optional[Union[int, Iterable[int]]] = None):
#     """
#     Helper function to get holidays for a country.

#     Parameters
#     ----------
#         country : str
#             Country name to retrieve country specific holidays
#         years : int, list
#             Year or list of years to retrieve holidays for

#     Returns
#     -------
#         set
#             All possible holiday dates and names of given country

#     """
#     # For compatibility with Turkey as "TU" cases.
#     country = "TUR" if country == "TU" else country
#     holiday_dict = country_holidays(country=country, years=years, expand=True, observed=False)
#     return holiday_dict


def get_holiday_names(country: Union[str, Iterable[str]], df=None):
    """
    Return all possible holiday names for a list of countries over time period in df

    Parameters
    ----------
        country : str, list
            List of country names to retrieve country specific holidays
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
        years = pd.unique(dates.apply(lambda x: x.year))
        # years = list({x.year for x in dates})
    # support multiple countries, convert to list if not already
    if isinstance(country, str):
        country = [country]

    all_holidays = get_all_holidays(years=years, country=country)
    return set(all_holidays.keys())


def get_all_holidays(years, country):
    """
    Make dataframe of country specific holidays for given years and countries
    Parameters
    ----------
        year_list : list
            List of years
        country : str, list
            List of country names
    Returns
    -------
        pd.DataFrame
            Containing country specific holidays df with columns 'ds' and 'holiday'
    """
    # convert to list if not already
    if isinstance(country, str):
        country = [country]
    all_holidays = defaultdict(list)
    # iterate over countries and get holidays for each country
    for single_country in country:
        # For compatibility with Turkey as "TU" cases.
        single_country = "TUR" if single_country == "TU" else single_country
        # get dict of dates and their holiday name
        single_country_specific_holidays = country_holidays(
            country=single_country, years=years, expand=True, observed=False
        )
        # invert order - for given holiday, store list of dates
        for date, name in single_country_specific_holidays.items():
            all_holidays[name].append(pd.to_datetime(date))
    return all_holidays

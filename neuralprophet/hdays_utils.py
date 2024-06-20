from typing import Iterable, Optional, Union

import holidays


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
    Return all possible holiday names of given country

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
        country = {country, None}
    elif isinstance(country, list):
        country = dict(zip(country, [None] * len(country)))

    unique_holidays = {}
    for single_country in country:
        subdivision = subdivision.get(single_country)
        holidays_country = get_country_holidays(single_country, years, subdivision)
        for date, name in holidays_country.items():
            if date not in unique_holidays:
                unique_holidays[date] = name
    holiday_names = unique_holidays.values()

    return set(holiday_names)

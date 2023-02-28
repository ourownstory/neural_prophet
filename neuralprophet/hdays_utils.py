from typing import Iterable, Optional, Union

import holidays


def get_country_holidays(country: str, years: Optional[Union[int, Iterable[int]]] = None):
    """
    Helper function to get holidays for a country.

    Parameters
    ----------
        country : str
            Country name to retrieve country specific holidays
        years : int, list
            Year or list of years to retrieve holidays for

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

    return getattr(holidays, country)(years=years)

from typing import Iterable, Optional, Union

import holidays as pyholidays

from neuralprophet import hdays as hdays_part2


def get_country_holidays(country: str, years: Optional[Union[int, Iterable[int]]] = None):
    """Helper function to get holidays for a country."""
    try:
        holidays_country = getattr(hdays_part2, country)(years=years)
    except AttributeError:
        try:
            holidays_country = getattr(pyholidays, country)(years=years)
        except AttributeError:
            raise AttributeError(f"Holidays in {country} are not currently supported!")

    return holidays_country

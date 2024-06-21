#!/usr/bin/env python3

import holidays
import pytest
from holidays import country_holidays


def test_get_country_holidays():
    for country in ("UnitedStates", "US", "USA"):
        us_holidays = country_holidays(country=country, years=2019)
        assert issubclass(us_holidays.__class__, holidays.countries.united_states.UnitedStates) is True
        assert len(us_holidays) == 10


def test_get_country_holidays_with_subdivisions():
    # Test US holidays with a subdivision
    us_ca_holidays = country_holidays("US", years=2019, subdiv="CA")
    assert issubclass(us_ca_holidays.__class__, holidays.countries.united_states.UnitedStates) is True
    assert len(us_ca_holidays) > 0  # Assuming there are holidays specific to CA

    # Test Canada holidays with a subdivision
    ca_on_holidays = country_holidays("CA", years=2019, subdiv="ON")
    assert issubclass(ca_on_holidays.__class__, holidays.countries.canada.CA) is True
    assert len(ca_on_holidays) > 0  # Assuming there are holidays specific to ON

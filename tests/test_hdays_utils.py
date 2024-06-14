#!/usr/bin/env python3

import holidays
import pytest

from neuralprophet import hdays_utils


def test_get_country_holidays():
    assert issubclass(hdays_utils.get_country_holidays("TU").__class__, holidays.countries.turkey.TR) is True

    for country in ("UnitedStates", "US", "USA"):
        us_holidays = hdays_utils.get_country_holidays(country, years=2019)
        assert issubclass(us_holidays.__class__, holidays.countries.united_states.UnitedStates) is True
        assert len(us_holidays) == 10

    with pytest.raises(AttributeError):
        hdays_utils.get_country_holidays("NotSupportedCountry")


def test_get_country_holidays_with_subdivisions():
    # Test US holidays with a subdivision
    us_ca_holidays = hdays_utils.get_country_holidays("US", years=2019, subdivision="CA")
    assert issubclass(us_ca_holidays.__class__, holidays.countries.united_states.UnitedStates) is True
    assert len(us_ca_holidays) > 0  # Assuming there are holidays specific to CA

    # Test Canada holidays with a subdivision
    ca_on_holidays = hdays_utils.get_country_holidays("CA", years=2019, subdivision="ON")
    assert issubclass(ca_on_holidays.__class__, holidays.countries.canada.CA) is True
    assert len(ca_on_holidays) > 0  # Assuming there are holidays specific to ON

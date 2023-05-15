#!/usr/bin/env python3

import holidays
import pytest

from neuralprophet import hdays_utils


def test_get_country_holidays():
    assert issubclass(hdays_utils.get_country_holidays("TU").__class__, holidays.Turkey) == True

    for country in ("UnitedStates", "US", "USA"):
        us_holidays = hdays_utils.get_country_holidays(country, years=2019)
        assert issubclass(us_holidays.__class__, holidays.UnitedStates) == True
        assert len(us_holidays) == 10

    with pytest.raises(AttributeError):
        hdays_utils.get_country_holidays("NotSupportedCountry")

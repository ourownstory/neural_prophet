#!/usr/bin/env python3

import pytest

from neuralprophet import event_utils


def test_get_country_holidays():
    # deprecated
    # assert issubclass(event_utils.get_country_holidays("TU").__class__, holidays.countries.turkey.TR) is True
    # new format
    assert issubclass(event_utils.get_all_holidays(country=["TU", "US"], years=2025).__class__, dict) is True

    for country in ("UnitedStates", "US", "USA"):
        us_holidays = event_utils.get_all_holidays(country=country, years=[2019, 2020])
        assert len(us_holidays) == 10

    with pytest.raises(NotImplementedError):
        event_utils.get_holiday_names("NotSupportedCountry")

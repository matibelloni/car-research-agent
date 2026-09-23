import pytest
from tools import convert_units


def test_mph_to_kmh():
    result = convert_units(unit_from="mph", unit_to="km/h", value=60)
    assert "96.56" in result


def test_mpg_to_l_per_100km():
    result = convert_units(unit_from="mpg", unit_to="l/100km", value=30)
    assert "7.84" in result


def test_fahrenheit_to_celsius():
    result = convert_units(unit_from="f", unit_to="c", value=212)
    assert "100" in result


def test_unknown_unit_raises():
    with pytest.raises(ValueError):
        convert_units(unit_from="bananas", unit_to="km", value=10)

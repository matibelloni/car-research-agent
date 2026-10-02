import pytest
from tools import convert_units


@pytest.mark.parametrize(
    "unit_from, unit_to, value, expected",
    [
        ("mph", "km/h", 60, "96.56 km/h"),
        ("mi", "km", 10, "16.09 km"),
        ("gal", "l", 1, "3.785 l"),
        ("hp", "kw", 100, "74.57 kW"),
        ("lb", "kg", 220, "99.79 kg"),
        ("in", "cm", 5, "12.7 cm"),
        ("psi", "bar", 30, "2.068 bar"),
        ("f", "c", 32, "0 °C"),
        ("f", "c", 212, "100 °C"),
        ("mpg", "l/100km", 25, "9.409 l/100km"),
        ("mpg", "l/100km", 30, "7.841 l/100km"),
        ("ft", "m", 10, "3.048 m"),
        ("feet", "meters", 10, "3.048 m"),
    ],
)
def test_convert_units(unit_from, unit_to, value, expected):
    result = convert_units(unit_from=unit_from, unit_to=unit_to, value=value)
    assert expected in result


def test_unknown_unit_raises():
    with pytest.raises(ValueError):
        convert_units(unit_from="bananas", unit_to="km", value=10)

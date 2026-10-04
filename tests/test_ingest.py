import pytest
from scripts.ingest_recalls import (
    normalize,
    is_vehicle,
    parse_date,
    brands_for,
    AVAILABLE_BRANDS,
)


def test_normalize_removes_accents_and_case():
    assert normalize("Vehículos") == "vehiculos"


def test_is_vehicle():
    assert is_vehicle("  VEHÍCULOS, partes y accesorios ") is True


def test_parse_date_before_switch_uses_month_day():
    assert parse_date("2026/05/09", 10, 499) == ("2026-05-09", True)


def test_parse_date_after_switch_uses_day_month():
    assert parse_date("2026/05/09", 500, 499) == ("2026-09-05", True)


def test_parse_date_unambiguous_is_not_inferred():
    assert parse_date("2026/15/05", 500, 499) == ("2026-05-15", False)


def test_parse_date_empty_returns_none():
    assert parse_date("", 500, 499) == (None, False)


def test_parse_date_at_switch_uses_day_month():
    assert parse_date("2026/05/09", 499, 499) == ("2026-09-05", True)


@pytest.mark.parametrize(
    "company, expected",
    [
        ("FCA AUTOMOBILES ARGENTINA", ["Chrysler", "Dodge", "Fiat", "Jeep", "RAM"]),
        ("YAHAMA", ["Yamaha"]),
        ("testRAMtest", []),
        (
            "FIAT, JEEP, RAM, CHRYSLER y DODGE",
            sorted(["Fiat", "Jeep", "RAM", "Chrysler", "Dodge"]),
        ),
        ("ram", ["RAM"]),
        ("Eximar (Volvo)", sorted(["Volvo", "Jaguar", "Land Rover"])),
        ("VOLKSWAGEN ARGENTINA SA", ["Audi", "Seat", "Volkswagen"]),
    ],
)
def test_brands_for(company, expected):
    assert brands_for(company) == expected


@pytest.mark.parametrize("brand", AVAILABLE_BRANDS)
def test_every_brand_recognizes_itself(brand):
    assert brand in brands_for(brand)

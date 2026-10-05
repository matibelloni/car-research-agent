import pytest
from recalls_ar import get_recalls_ar, query_brands

FAKE_RECALLS = [
    {
        "date": "2026-01-01",
        "company": "VW",
        "brands": ["Volkswagen"],
        "product": "Gol",
        "defect": "airbag",
        "risk": "Lesiones",
    },
    {
        "date": "2025-01-01",
        "company": "VW",
        "brands": ["Audi"],
        "product": "A4",
        "defect": "frenos",
        "risk": "Lesiones",
    },
    {
        "date": "2024-01-01",
        "company": "FCA",
        "brands": ["Fiat"],
        "product": "Cronos y Argo",
        "defect": "freno de mano",
        "risk": "Lesiones",
    },
    {
        "date": "2023-01-01",
        "company": "FCA",
        "brands": ["Fiat"],
        "product": "Toro",
        "defect": "airbag",
        "risk": "Lesiones",
    },
    {
        "date": "2022-01-01",
        "company": "VW",
        "brands": ["Volkswagen"],
        "product": "Gol",
        "defect": "frenos",
        "risk": "Lesiones",
    },
]


@pytest.fixture(autouse=True)
def fake_data(monkeypatch):
    monkeypatch.setattr("recalls_ar.load_recalls", lambda: FAKE_RECALLS)


@pytest.mark.parametrize("brand", ["Tesla", "Ranger", ""])
def test_unknown_brand_returns_error_with_options(brand):
    result = get_recalls_ar(brand)
    assert result["error"] == "unknown_brand"
    assert "Volkswagen" in result["available_brands"]


def test_vw_aliases_return_same_result():
    result_full = get_recalls_ar("Volkswagen")
    result_upper = get_recalls_ar("VW")
    result_lower = get_recalls_ar("vw")
    assert result_full == result_upper == result_lower
    assert len(result_full["recalls"]) > 0


def test_keyword_filters_every_row():
    result = get_recalls_ar("Fiat", "Cronos")
    assert result["total"] == 1
    assert result["recalls"][0]["product"] == "Cronos y Argo"


def test_multi_word_keyword_matches_across_fields():
    result = get_recalls_ar("Volkswagen", "Gol airbag")
    assert result["total"] == 1
    assert result["recalls"][0]["defect"] == "airbag"


@pytest.mark.parametrize(
    "name, expected",
    [("VW", ["Volkswagen"]), ("Toyota", ["Toyota"]), ("Mercedes", ["Mercedes-Benz"])],
)
def test_query_resolves_single_brand(name, expected):
    assert query_brands(name) == expected


def test_vw_excludes_audi_rows():
    result = get_recalls_ar("VW")
    assert result["total"] == 2
    assert result["recalls"][0]["product"] == "Gol"

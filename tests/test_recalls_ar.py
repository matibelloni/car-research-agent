import pytest
from recalls_ar import get_recalls_ar, normalize


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
    recalls = get_recalls_ar("Fiat", "Cronos")["recalls"]
    assert recalls
    assert all(
        "cronos" in normalize(recall["product"])
        or "cronos" in normalize(recall["defect"])
        for recall in recalls
    )

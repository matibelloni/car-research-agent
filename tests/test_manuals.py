import pytest
from manuals import available_models, citation, resolve_manual


@pytest.mark.parametrize(
    "model", ["Cronos", "cronos", "Fiat Cronos", "FIAT CRONOS 1.3", "el crónos"]
)
def test_resolves_model_and_aliases_ignoring_case_and_accents(model):
    assert resolve_manual(model, 2021)["manual"] == "fiat-cronos-2021"


def test_exact_year_has_no_note():
    assert resolve_manual("Peugeot 208", 2024) == {
        "manual": "peugeot-208-2024",
        "note": None,
    }


def test_other_year_uses_the_closest_manual_and_says_so():
    resolved = resolve_manual("Cronos", 2024)
    assert resolved["manual"] == "fiat-cronos-2021"
    assert resolved["note"] == (
        "Using the 2021 manual; the user asked about 2024: "
        "specs may differ between model years."
    )


def test_no_year_uses_the_newest_manual_and_says_so():
    resolved = resolve_manual("208")
    assert resolved["manual"] == "peugeot-208-2024"
    assert "didn't give a model year" in resolved["note"]


@pytest.mark.parametrize("model", ["Toyota Etios", "Peugeot 2008", "Corsa"])
def test_unknown_model_lists_the_available_ones(model):
    resolved = resolve_manual(model, 2020)
    assert resolved["error"] == "unknown_model"
    assert resolved["available_models"] == available_models() == [
        "Fiat Cronos",
        "Peugeot 208",
    ]


def test_closest_year_picks_the_newer_manual_on_a_tie(monkeypatch):
    import manuals

    older = dict(manuals.MANUALS["fiat-cronos-2021"], model_year=2019)
    newer = dict(manuals.MANUALS["fiat-cronos-2021"], model_year=2023)
    monkeypatch.setattr(manuals, "MANUALS", {"cronos-2019": older, "cronos-2023": newer})
    assert resolve_manual("Cronos", 2021)["manual"] == "cronos-2023"
    assert resolve_manual("Cronos", 2020)["manual"] == "cronos-2019"


def test_citation():
    assert citation("peugeot-208-2024", "MG-22") == "Manual del Peugeot 208 2024, p. MG-22"
    assert citation("fiat-cronos-2021", None) == "Manual del Fiat Cronos 2021"

import json
import re
from functools import cache
from pathlib import Path

from brands import AVAILABLE_BRANDS, normalize

RECALLS_PATH = Path(__file__).resolve().parent.parent / "data" / "recalls_clean.json"
MAX_RESULTS = 15
FIELDS = ("date", "company", "product", "defect", "risk")

RECALLS_AR_TOOL = {
    "name": "get_recalls_ar",
    "description": (
        "Looks up official vehicle recalls published in Argentina by the national "
        "consumer protection agency (Dirección Nacional de Defensa del Consumidor). "
        "Each result has the publication date, the company that issued it, the "
        "affected product, the defect and the risk. Use it for any question about "
        "recalls or safety defects of cars sold in Argentina, before searching the "
        "web. Results rarely include model years, and recalls usually cover specific "
        "production ranges: never state that a particular vehicle is affected. "
        "Instead, tell the user to check their VIN on the brand's website or at a dealer."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "brand": {
                "type": "string",
                "description": (
                    "The vehicle brand, not the model. Examples: 'Volkswagen', "
                    "'Fiat', 'Toyota'. Pass 'Ford', not 'Ranger'. If the brand is not "
                    "recognized, the result lists the available brands."
                ),
            },
            "keyword": {
                "type": "string",
                "description": (
                    "Optional. One or more short words to narrow the results; every "
                    "word must appear as a whole word (plurals also match) in the "
                    "product or defect text. Use a model "
                    "('Cronos', 'Amarok'), a defect type ('airbag', 'freno'), or both "
                    "when the user asks about a specific model AND a defect "
                    "(e.g. 'Gol airbag'). The data is in Spanish: pass defect "
                    "keywords in Spanish (e.g. 'freno', not 'brake'). Use it when "
                    "'total' is greater than 'shown'."
                ),
            },
        },
        "required": ["brand"],
    },
}


@cache
def load_recalls() -> list[dict]:
    return json.loads(RECALLS_PATH.read_text(encoding="utf-8"))


QUERY_ALIASES = {"vw": "Volkswagen", "mercedes": "Mercedes-Benz", "chevy": "Chevrolet"}


def query_brands(name: str) -> list[str]:
    """Canonical brands a user or the model refers to, without publisher expansion."""
    text = normalize(name)
    found = {
        b
        for b in AVAILABLE_BRANDS
        if re.search(rf"\b{re.escape(normalize(b))}\b", text)
    }
    found |= {
        b for alias, b in QUERY_ALIASES.items() if re.search(rf"\b{alias}\b", text)
    }
    return sorted(found)


def matches_word(word: str, text: str) -> bool:
    """Whole-word match, so "gol" skips "Golf" and "e" (as in "Clase E") skips
    every word containing an e. Words of 3+ letters match in singular or plural
    either way ("freno" <-> "frenos"); shorter ones match exactly, or "e" would
    match "es"."""
    if len(word) < 3:
        return re.search(rf"\b{re.escape(word)}\b", text) is not None
    stems = {word, word.removesuffix("s"), word.removesuffix("es")}
    alternatives = "|".join(re.escape(stem) for stem in stems if len(stem) >= 3)
    return re.search(rf"\b(?:{alternatives})(?:s|es)?\b", text) is not None


def get_recalls_ar(brand: str, keyword: str | None = None) -> dict:
    target_brands = query_brands(brand)
    if not target_brands:
        return {
            "error": "unknown_brand",
            "message": f"Brand '{brand}' not recognized.",
            "available_brands": AVAILABLE_BRANDS,
        }
    target = set(target_brands)
    recalls = [recall for recall in load_recalls() if target & set(recall["brands"])]
    if keyword:
        words = normalize(keyword).split()
        recalls = [
            r
            for r in recalls
            if all(
                matches_word(word, normalize(r["product"] + " " + r["defect"]))
                for word in words
            )
        ]
    recalls.sort(key=lambda r: r["date"] or "", reverse=True)
    shown = [
        {field: recall[field] for field in FIELDS} for recall in recalls[:MAX_RESULTS]
    ]
    result = {"total": len(recalls), "shown": len(shown), "recalls": shown}
    if len(recalls) > len(shown):
        result["note"] = (
            f"Showing {len(shown)} of {len(recalls)}. Add a model or defect "
            "to keyword to narrow the results before answering."
        )
    return result

import json
from functools import cache
from pathlib import Path
from scripts.ingest_recalls import (
    AVAILABLE_BRANDS,
    brands_for,
    normalize,
)

RECALLS_PATH = Path(__file__).parent / "data" / "recalls_clean.json"


@cache
def load_recalls() -> list[dict]:
    return json.loads(RECALLS_PATH.read_text(encoding="utf-8"))


def get_recalls_ar(brand: str, keyword: str | None = None) -> dict:
    target_brands = brands_for(brand)
    if not target_brands:
        return {
            "error": "unknown_brand",
            "message": f"Brand '{brand}' not recognized.",
            "available_brands": AVAILABLE_BRANDS,
        }
    target = set(target_brands)
    recalls = [recall for recall in load_recalls() if target & set(recall["brands"])]
    if keyword:
        key = normalize(keyword)
        recalls = [
            recall
            for recall in recalls
            if key in normalize(recall["product"]) or key in normalize(recall["defect"])
        ]

    return {"recalls": recalls}

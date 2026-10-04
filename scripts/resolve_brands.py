"""Narrow each multi-brand recall to the brands it actually covers.

Runs after ingest_recalls.py. Publishers like FCA or Volkswagen Argentina
issue recalls for several brands, so ingestion assigns every brand the
publisher may cover. This step asks an LLM to pick, from those candidates,
the brands the product and defect actually refer to. It never adds a brand
outside the candidates, and keeps all of them when the text doesn't say.
"""

import hashlib
import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

DATA = Path("data/recalls_clean.json")
CACHE = Path("data/brand_cache.json")
MODEL = "claude-haiku-4-5"

PROMPT = """A vehicle recall was published in Argentina by "{company}".
That publisher may cover these brands: {candidates}.

Product: {product}
Defect: {defect}

Which of the candidate brands does this recall apply to? Decide from the model
names in the product and defect (e.g. "Renegade" is Jeep, "A4" is Audi,
"C4 Cactus" is Citroën). Only return brands from the candidate list, spelled
exactly as listed. If the text doesn't make it clear, return all the candidates."""


class BrandResolution(BaseModel):
    brands: list[str]


def resolve(claude: anthropic.Anthropic, recall: dict, candidates: list[str]) -> list[str]:
    response = claude.messages.parse(
        model=MODEL,
        max_tokens=200,
        extra_body={"temperature": 0},
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(
                    company=recall["company"],
                    candidates=", ".join(candidates),
                    product=recall["product"],
                    defect=recall["defect"],
                ),
            }
        ],
        output_format=BrandResolution,
    )
    parsed = response.parsed_output
    if parsed is None:
        return candidates
    # Drop anything outside the candidates; if nothing valid is left, keep them all.
    chosen = sorted({b for b in parsed.brands if b in candidates})
    return chosen or candidates


def cache_key(recall: dict, candidates: list[str]) -> str:
    """Same text and same candidates -> same key, so unchanged rows are never re-sent."""
    raw = json.dumps([recall["company"], recall["product"], recall["defect"], candidates])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    load_dotenv()
    claude = anthropic.Anthropic(max_retries=3)
    recalls = json.loads(DATA.read_text(encoding="utf-8"))
    cache = load_cache()

    # candidate_brands keeps the publisher-level mapping, so re-running this
    # script always starts from the same candidates.
    ambiguous = [r for r in recalls if len(r.get("candidate_brands", r["brands"])) > 1]
    print(f"Resolving {len(ambiguous)} of {len(recalls)} recalls")

    narrowed = 0
    failed = 0
    llm_calls = 0
    for i, recall in enumerate(ambiguous, 1):
        candidates = recall.setdefault("candidate_brands", recall["brands"])
        key = cache_key(recall, candidates)
        if key in cache:
            recall["brands"] = cache[key]
        else:
            try:
                recall["brands"] = resolve(claude, recall, candidates)
            except anthropic.APIError as e:
                recall["brands"] = candidates
                failed += 1
                print(f"  row {recall['row']}: API error, keeping candidates ({e})")
                continue
            cache[key] = recall["brands"]
            llm_calls += 1
        if recall["brands"] != candidates:
            narrowed += 1
        print(f"  [{i}/{len(ambiguous)}] row {recall['row']}: {candidates} -> {recall['brands']}")

    DATA.write_text(json.dumps(recalls, ensure_ascii=False, indent=2), encoding="utf-8")
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"LLM calls: {llm_calls} | From cache: {len(ambiguous) - llm_calls - failed}")
    print(f"Narrowed: {narrowed} | Kept all candidates: {len(ambiguous) - narrowed - failed} | Failed: {failed}")
    print(f"Saved to {DATA}")


if __name__ == "__main__":
    main()
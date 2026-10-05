"""Eval of how the agent uses get_recalls_ar.

Two deterministic checks per run:
- lookup_ok: some recalls lookup used the expected brand and keyword fragments.
- total_ok: the answer states the total the last lookup returned in that run.

Usage:
    python -m evals.eval_recalls                 # all cases
    python -m evals.eval_recalls model_and_defect  # only the cases whose id is given
"""

import re
import sys

from agent import langfuse, run_agent_stream
from recalls_ar import query_brands
from scripts.ingest_recalls import normalize

RUNS_PER_CASE = 3

RECALLS_CASES = [
    {
        "id": "model_and_defect",
        "question": "¿Hay recalls de airbags para el Volkswagen Gol en Argentina?",
        "expected_lookup": {
            "brand": "Volkswagen",
            "keyword_contains": ["gol", "airbag"],
        },
    },
    {
        "id": "brand_only",
        "question": "hay recalls para renault?",
        "expected_lookup": {"brand": "Renault", "keyword_contains": []},
    },
    {
        "id": "brand_alias",
        "question": "dame los recalls para VW",
        "expected_lookup": {"brand": "Volkswagen", "keyword_contains": []},
    },
    # BYD is not in the sheet: the lookup must come back as unknown_brand.
    {
        "id": "unknown_brand",
        "question": "hay recalls para BYD?",
        "expected_lookup": {"brand": None, "keyword_contains": []},
    },
    # The data is in Spanish: "brake" must become "freno"/"frenos".
    {
        "id": "english_defect",
        "question": "Are there brake recalls for Peugeot?",
        "expected_lookup": {"brand": "Peugeot", "keyword_contains": ["fren"]},
    },
    {
        "id": "english_model",
        "question": "Are there recalls for the VW Taos?",
        "expected_lookup": {"brand": "Volkswagen", "keyword_contains": ["taos"]},
    },
    # The user never says Toyota: the model has to infer it.
    {
        "id": "model_without_brand",
        "question": "¿Hay recalls para la Hilux?",
        "expected_lookup": {"brand": "Toyota", "keyword_contains": ["hilux"]},
    },
    {
        "id": "known_brand_zero_results",
        "question": "Are there airbag recalls for the VW Amarok?",
        "expected_lookup": {
            "brand": "Volkswagen",
            "keyword_contains": ["amarok", "airbag"],
        },
    },
    # More results than the tool shows: the answer must still report the total.
    {
        "id": "more_than_max_results",
        "question": "dame los recalls de la Ford Ranger",
        "expected_lookup": {"brand": "Ford", "keyword_contains": ["ranger"]},
    },
    {
        "id": "brand_accent",
        "question": "recalls de citroen",
        "expected_lookup": {"brand": "Citroën", "keyword_contains": []},
    },
    # Behaviour to judge later: never claim this specific car is affected.
    {
        "id": "no_vin_claim",
        "question": "¿Mi Cronos 2020 está afectado por algún recall?",
        "expected_lookup": {"brand": "Fiat", "keyword_contains": ["cronos"]},
    },
    {
        "id": "brand_in_group",
        "question": "hay recalls para Audi?",
        "expected_lookup": {"brand": "Audi", "keyword_contains": []},
    },
]


def run_once(question: str) -> dict:
    """Runs the agent once and collects what the eval needs."""
    lookups = []
    answer = ""
    error = None
    for event in run_agent_stream(question):
        if event["type"] == "recalls_lookup":
            lookups.append(
                {
                    "brand": event["brand"],
                    "keyword": event["keyword"],
                    "total": event["total"],
                    "error": event["error"],
                }
            )
        elif event["type"] == "result":
            answer = event["answer"] or ""
            error = event["error"]
    return {"answer": answer, "lookups": lookups, "error": error}


def lookup_matches(expected: dict, lookup: dict) -> bool:
    if expected["brand"] is None:
        return lookup["error"] == "unknown_brand"
    if query_brands(lookup["brand"]) != [expected["brand"]]:
        return False
    keyword = normalize(lookup["keyword"] or "")
    return all(fragment in keyword for fragment in expected["keyword_contains"])


def reports_total(lookups: list[dict], answer: str) -> bool | None:
    """None when the check doesn't apply: no lookup, unknown brand, or zero results."""
    if not lookups or not lookups[-1]["total"]:
        return None
    return re.search(rf"\b{lookups[-1]['total']}\b", answer) is not None


def score(case: dict, run: dict) -> dict:
    return {
        "lookup_ok": any(
            lookup_matches(case["expected_lookup"], l) for l in run["lookups"]
        ),
        "total_ok": reports_total(run["lookups"], run["answer"]),
    }


def summarize(results: list[dict], check: str) -> str:
    """'passed/applicable', or 'n/a' when the check never applied."""
    applicable = [r[check] for r in results if r[check] is not None]
    if not applicable:
        return "n/a"
    return f"{sum(applicable)}/{len(applicable)}"


def main(case_ids: list[str]) -> None:
    cases = [c for c in RECALLS_CASES if not case_ids or c["id"] in case_ids]
    if not cases:
        print(f"No case matches {case_ids}")
        return

    for case in cases:
        results = []
        evidence = []
        for _ in range(RUNS_PER_CASE):
            run = run_once(case["question"])
            result = score(case, run)
            results.append(result)
            # Show the evidence of failures, so you don't have to dig into Langfuse.
            if run["error"]:
                evidence.append(f"    agent error: {run['error']}")
            elif not result["lookup_ok"] or result["total_ok"] is False:
                evidence.append(f"    lookups: {run['lookups']}")
        print(
            f"{case['id']:<26} lookup {summarize(results, 'lookup_ok'):<6} "
            f"total {summarize(results, 'total_ok')}"
        )
        for line in evidence:
            print(line)

    langfuse.flush()


if __name__ == "__main__":
    main(sys.argv[1:])

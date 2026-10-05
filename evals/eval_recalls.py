"""Eval of how the agent uses get_recalls_ar.

Per run:
- lookup_ok (deterministic): some recalls lookup used the expected brand and keyword.
- total_ok (deterministic): the answer states the total the last lookup returned.
- behavior_ok (LLM judge): only for cases with a "judge" criterion.

Usage:
    python -m evals.eval_recalls                   # all cases
    python -m evals.eval_recalls model_and_defect  # only the cases whose id is given
    python -m evals.eval_recalls --calibrate       # check the judge on hand-written answers
"""

import re
import sys

import anthropic
from pydantic import BaseModel

from agent import langfuse, run_agent_stream
from tools.recalls_ar import query_brands
from brands import normalize

RUNS_PER_CASE = 3
JUDGE_MODEL = "claude-haiku-4-5"

CRITERIA = {
    "no_vin_claim": (
        "The answer must NOT state or imply that the user's specific vehicle is, or is "
        "not, affected by a recall. It must say that recalls cover specific production "
        "ranges and recommend checking the VIN with the brand or a dealer."
    ),
    "no_invented_recalls": (
        "The lookup found no recalls, or did not recognize the brand. The answer must NOT "
        "describe, list or count any recall, and must clearly say that no recalls were "
        "found in the data (or that the brand is not in it)."
    ),
}

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
    # BYD is not in the sheet: the agent still looks it up, and the tool must
    # answer unknown_brand.
    {
        "id": "unknown_brand",
        "question": "hay recalls para BYD?",
        "expected_lookup": {"error": "unknown_brand"},
        "judge": "no_invented_recalls",
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
        "judge": "no_invented_recalls",
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
    # Never claim this specific car is (or isn't) affected.
    {
        "id": "no_vin_claim",
        "question": "¿Mi Cronos 2020 está afectado por algún recall?",
        "expected_lookup": {"brand": "Fiat", "keyword_contains": ["cronos"]},
        "judge": "no_vin_claim",
    },
    {
        "id": "brand_in_group",
        "question": "hay recalls para Audi?",
        "expected_lookup": {"brand": "Audi", "keyword_contains": []},
    },
]


# Hand-written answers with a known verdict. If the judge disagrees, fix the
# criterion before trusting it on the agent.
CALIBRATION = [
    (
        "no_vin_claim",
        "¿Mi Cronos 2020 está afectado por algún recall?",
        [{"brand": "Fiat", "keyword": "Cronos", "total": 7, "error": None}],
        "Hay 7 recalls del Cronos (Defensa del Consumidor, 15/03/2021). Los recalls cubren "
        "rangos de producción específicos, así que no puedo saber si tu auto está afectado: "
        "verificá tu VIN en el sitio de Fiat o en un concesionario.",
        True,
    ),
    (
        "no_vin_claim",
        "¿Mi Cronos 2020 está afectado por algún recall?",
        [{"brand": "Fiat", "keyword": "Cronos", "total": 7, "error": None}],
        "Sí, tu Cronos 2020 está afectado por el recall de frenos (Defensa del Consumidor, "
        "15/03/2021). Llevalo al concesionario.",
        False,
    ),
    (
        "no_invented_recalls",
        "Are there airbag recalls for the VW Amarok?",
        [
            {
                "brand": "Volkswagen",
                "keyword": "Amarok airbag",
                "total": 0,
                "error": None,
            }
        ],
        "I found no airbag recalls for the VW Amarok in the Defensa del Consumidor data.",
        True,
    ),
    (
        "no_invented_recalls",
        "Are there airbag recalls for the VW Amarok?",
        [
            {
                "brand": "Volkswagen",
                "keyword": "Amarok airbag",
                "total": 0,
                "error": None,
            }
        ],
        "There are 2 airbag recalls for the Amarok, both about the gas generator "
        "(Defensa del Consumidor, 18/07/2024).",
        False,
    ),
    (
        "no_invented_recalls",
        "hay recalls para BYD?",
        [{"brand": "BYD", "keyword": None, "total": None, "error": "unknown_brand"}],
        "BYD no figura entre las marcas de la base de recalls de Defensa del Consumidor, "
        "así que no encontré recalls para esa marca.",
        True,
    ),
]


class Verdict(BaseModel):
    passed: bool
    reasoning: str


JUDGE_PROMPT = """You are a strict evaluator of a car research assistant.

CRITERION: {criterion}

The assistant looked up official recalls and got these results:
{lookups}

QUESTION: {question}

ANSWER TO EVALUATE:
{answer}

Decide only whether the answer meets the criterion. Ignore style, length and anything else."""

claude = anthropic.Anthropic(max_retries=3)


def judge(
    criterion_key: str, question: str, lookups: list[dict], answer: str
) -> Verdict | None:
    summary = (
        "\n".join(
            f"- brand={l['brand']!r} keyword={l['keyword']!r} total={l['total']} error={l['error']}"
            for l in lookups
        )
        or "- no lookup was made"
    )
    response = claude.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    criterion=CRITERIA[criterion_key],
                    lookups=summary,
                    question=question,
                    answer=answer,
                ),
            }
        ],
        output_format=Verdict,
    )
    return response.parsed_output


def calibrate() -> None:
    for criterion_key, question, lookups, answer, expected in CALIBRATION:
        verdict = judge(criterion_key, question, lookups, answer)
        got = verdict.passed if verdict else None
        status = "OK" if got == expected else "MISMATCH"
        print(
            f"{status:<9} {criterion_key:<20} expected {expected!s:<5} got {got!s:<5} "
            f"— {verdict.reasoning if verdict else 'judge failed'}"
        )


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
    if "error" in expected:
        return lookup["error"] == expected["error"]
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
    result = {
        "lookup_ok": any(
            lookup_matches(case["expected_lookup"], l) for l in run["lookups"]
        ),
        "total_ok": reports_total(run["lookups"], run["answer"]),
        "behavior_ok": None,
        "judge_reasoning": None,
    }
    if "judge" in case and run["answer"]:
        verdict = judge(case["judge"], case["question"], run["lookups"], run["answer"])
        if verdict:
            result["behavior_ok"] = verdict.passed
            result["judge_reasoning"] = verdict.reasoning
    return result


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
            if result["behavior_ok"] is False:
                evidence.append(f"    judge: {result['judge_reasoning']}")
                evidence.append(f"    answer: {run['answer'][:300]}")
        print(
            f"{case['id']:<26} lookup {summarize(results, 'lookup_ok'):<6} "
            f"total {summarize(results, 'total_ok'):<6} "
            f"behavior {summarize(results, 'behavior_ok')}"
        )
        for line in evidence:
            print(line)

    langfuse.flush()


if __name__ == "__main__":
    if sys.argv[1:] == ["--calibrate"]:
        calibrate()
    else:
        main(sys.argv[1:])

"""End-to-end eval of how the agent answers questions about an owner's manual.

Per run:
- route_ok (deterministic): "manual" cases cite an expected page label and make
  no web search; "web" cases make at least one web search.
- facts_ok (LLM judge): every fact of the case is stated in the answer. Facts
  come only from the manual's chunks, so "web" cases have none.

Usage:
    python -m evals.manual                  # all cases
    python -m evals.manual cronos-brake-fluid  # only the cases whose id is given
    python -m evals.manual --calibrate      # check the judge on hand-written answers
"""

import re
import sys

from pydantic import BaseModel

from agent import langfuse, run_agent_stream
from evals.judge import ask_judge, judge_model

RUNS = 3

# "page" is the page label a correct answer cites, or a tuple when more than one
# page answers the question. Every fact is copied from a chunk's text.
CASES = [
    {
        "id": "cronos-accessory-belts",
        "question": "¿Cada cuántos km se cambia la correa de accesorios del Fiat Cronos 2021?",
        "route": "manual",
        "page": "G-14",
        "facts": [
            "The accessory drive belts are replaced every 48 months or every "
            "60,000 km, whichever comes first.",
            "Under severe use (dusty roads, towing, taxi, delivery) this interval "
            "is halved.",
        ],
    },
    # "Taxi" only appears in the maintenance plan's footnote (G-14); the severe
    # use section (G-18) lists the conditions without it. Both answer it.
    {
        "id": "cronos-severe-use",
        "question": "Uso mi Fiat Cronos 2021 de remis, ¿tengo que hacerle el service más seguido?",
        "route": "manual",
        "page": ("G-18", "G-14"),
        "facts": [
            "Taxi use counts as severe use: items marked (*) must be replaced or "
            "checked at half the indicated intervals.",
            "Under severe use, checks such as front brake pads, engine oil and oil "
            "filter, and the air filter must be done more often than the scheduled "
            "maintenance plan says.",
        ],
    },
    {
        "id": "cronos-brake-fluid",
        "question": "¿Cada cuánto hay que cambiar el líquido de frenos del Fiat Cronos 2021?",
        "route": "manual",
        "page": "G-14",
        "facts": [
            "The brake fluid is replaced every 24 months or every 40,000 km, "
            "whichever comes first.",
            "If the car has a hydraulic clutch, its fluid should be replaced at "
            "the same time.",
        ],
    },
    {
        "id": "cronos-service-price",
        "question": "¿Cuánto cuesta el service de los 10.000 km del Fiat Cronos 2021?",
        "route": "web",
        "page": None,
        "facts": [],
    },
    {
        "id": "cronos-highway-consumption",
        "question": "¿Qué opinan los dueños del Fiat Cronos 2021 sobre el consumo en ruta?",
        "route": "web",
        "page": None,
        "facts": [],
    },
]

BRAKE_FLUID = CASES[2]

# Hand-written answers with a known verdict per fact. If the judge disagrees,
# fix the prompt before trusting it on the agent.
# Each case: (label, case, answer, expected verdict per fact).
CALIBRATION = [
    (
        "complete",
        BRAKE_FLUID,
        "Según el manual, el líquido de frenos del Fiat Cronos 2021 se cambia cada "
        "24 meses o cada 40.000 km, lo que ocurra primero. Si el auto tiene "
        "embrague hidráulico, también hay que cambiar ese fluido en el mismo "
        "momento (Manual del Fiat Cronos 2021, p. G-14).",
        [True, True],
    ),
    (
        "missing fact",
        BRAKE_FLUID,
        "Según el manual, el líquido de frenos del Fiat Cronos 2021 se cambia cada "
        "24 meses o cada 40.000 km, lo que ocurra primero (Manual del Fiat Cronos "
        "2021, p. G-14).",
        [True, False],
    ),
    (
        "changed number",
        BRAKE_FLUID,
        "Según el manual, el líquido de frenos del Fiat Cronos 2021 se cambia cada "
        "24 meses o cada 50.000 km, lo que ocurra primero. Si el auto tiene "
        "embrague hidráulico, también hay que cambiar ese fluido en el mismo "
        "momento (Manual del Fiat Cronos 2021, p. G-14).",
        [False, True],
    ),
    (
        "equivalent wording",
        BRAKE_FLUID,
        "El manual indica cambiar el líquido de frenos cada 2 años o 40 mil km, lo "
        "primero que se cumpla, y aprovechar para cambiar el del embrague "
        "hidráulico si lo tiene (Manual del Fiat Cronos 2021, p. G-14).",
        [True, True],
    ),
]


class FactVerdict(BaseModel):
    # Reasoning first: fields are generated in order, so the verdict comes
    # after the analysis instead of being justified after the fact.
    reasoning: str
    stated: bool


class FactsVerdict(BaseModel):
    facts: list[FactVerdict]


JUDGE_PROMPT = """You are a strict evaluator of a car research assistant.

For each numbered FACT, decide whether the ANSWER states it. The answer may be
in another language or use other words; what matters is the content.
- A fact is stated only if every figure in it (distances, months, years,
  fractions) appears in the answer with the same value. A different number
  means the fact is NOT stated.
- A fact is stated only if all its parts are there: a fact with an interval and
  a "whichever comes first" condition needs both.
- Facts are independent: judge each one on its own.

Return exactly one verdict per fact, in the same order.

QUESTION: {question}

FACTS:
{facts}

ANSWER TO EVALUATE:
{answer}"""


def judge_facts(case: dict, answer: str) -> list[bool] | None:
    """One verdict per fact, or None when the judge fails or miscounts them."""
    facts = "\n".join(f"{n}. {fact}" for n, fact in enumerate(case["facts"], start=1))
    prompt = JUDGE_PROMPT.format(question=case["question"], facts=facts, answer=answer)
    verdict = ask_judge(prompt, FactsVerdict)
    if verdict is None or len(verdict.facts) != len(case["facts"]):
        return None
    return [fact.stated for fact in verdict.facts]


def calibrate() -> None:
    print(f"Judge: {judge_model()}")
    for label, case, answer, expected in CALIBRATION:
        got = judge_facts(case, answer)
        status = "OK" if got == expected else "MISMATCH"
        print(f"{status:<9} {label:<18} expected {expected} got {got}")


def cites_page(answer: str, page: str | tuple[str, ...]) -> bool:
    pages = (page,) if isinstance(page, str) else page
    return any(re.search(rf"\b{re.escape(label)}\b", answer) for label in pages)


def run_once(question: str) -> dict:
    """Runs the agent once and collects what the eval needs."""
    for event in run_agent_stream(question):
        if event["type"] == "result":
            return {
                "answer": event["answer"] or "",
                "error": event["error"],
                "tokens": event["tokens"],
                "searches": event["searches"],
            }
    return {"answer": "", "error": "no_result", "tokens": 0, "searches": 0}


def score(case: dict, run: dict) -> dict:
    if case["route"] == "manual":
        route_ok = run["searches"] == 0 and cites_page(run["answer"], case["page"])
    else:
        route_ok = run["searches"] >= 1
    facts = None
    if case["facts"] and run["answer"]:
        facts = judge_facts(case, run["answer"])
    return {
        "route_ok": route_ok,
        # None when the check doesn't apply (no facts) or the judge failed.
        "facts_ok": all(facts) if facts is not None else None,
        "facts": facts,
    }


def summarize(results: list[dict], check: str) -> str:
    """'passed/applicable', or 'n/a' when the check never applied."""
    applicable = [r[check] for r in results if r[check] is not None]
    if not applicable:
        return "n/a"
    return f"{sum(applicable)}/{len(applicable)}"


def main(case_ids: list[str]) -> None:
    cases = [c for c in CASES if not case_ids or c["id"] in case_ids]
    if not cases:
        print(f"No case matches {case_ids}")
        return

    total_tokens = 0
    for case in cases:
        results = []
        evidence = []
        case_tokens = 0
        for _ in range(RUNS):
            run = run_once(case["question"])
            # Failed runs spent tokens too: count them, or the cost looks lower than it is.
            case_tokens += run["tokens"]
            result = score(case, run)
            results.append(result)
            # Show the evidence of failures, so you don't have to dig into Langfuse.
            if run["error"]:
                evidence.append(f"    agent error: {run['error']}")
            elif not result["route_ok"] or result["facts_ok"] is False:
                evidence.append(
                    f"    searches={run['searches']} facts={result['facts']}"
                )
                evidence.append(f"    answer: {run['answer']}")
            elif case["facts"] and result["facts"] is None:
                evidence.append("    judge failed")
        total_tokens += case_tokens
        print(
            f"{case['id']:<28} route {summarize(results, 'route_ok'):<6} "
            f"facts {summarize(results, 'facts_ok'):<6} tokens {case_tokens}"
        )
        for line in evidence:
            print(line)

    print(f"\nTotal tokens: {total_tokens}")
    langfuse.flush()


if __name__ == "__main__":
    if sys.argv[1:] == ["--calibrate"]:
        calibrate()
    else:
        main(sys.argv[1:])

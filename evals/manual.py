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
# Haiku failed the calibration cases where the halving is stated in passing or
# as a figure the answer says it computed; Sonnet passed every case 3 of 3.
FACTS_JUDGE = "claude-sonnet-5-5"

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
            "Under severe use, the front brake pads, the engine oil and oil filter, "
            "and the air filter must be checked more often than the scheduled "
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
    # The interval depends on the engine. The second fact isn't in the plan
    # (p. 22) but in the severe-use section (p. 19), another chunk.
    {
        "id": "208-accessory-belts",
        "question": "¿Cada cuántos km se cambia la correa de accesorios del Peugeot 208 1.6?",
        "route": "manual",
        "page": "MG-22",
        "facts": [
            "With the 1.6 16V engine, the accessory drive belt is replaced every "
            "80,000 km or 4 years, whichever comes first.",
            "Under severe use (e.g. taxi), it is replaced at half the indicated "
            "interval.",
        ],
    },
    # The manual gives pressures in psi; the agent converts them.
    {
        "id": "208-tire-pressure",
        "question": "¿Qué presión llevan las cubiertas del Peugeot 208 con llantas 16?",
        "route": "manual",
        "page": "211",
        "facts": [
            "With 195/55 R16 tyres, the front tyres take 29 psi with the driver and "
            "one passenger, 33 psi with more occupants and luggage, and 36 psi to "
            "reduce fuel consumption.",
            "The rear tyres take 29 psi with the driver and one passenger, and 36 psi "
            "with more occupants and luggage or to reduce fuel consumption.",
        ],
    },
    {
        "id": "208-severe-use",
        "question": "Uso mi Peugeot 208 de remis, ¿tengo que hacerle el service más seguido?",
        "route": "manual",
        "page": "MG-19",
        "facts": [
            "Taxi use counts as severe use.",
            "Under severe use, the engine oil and oil filter, the fuel filter and the "
            "engine air filter are replaced at half the indicated intervals.",
        ],
    },
    # Ambiguous in the manual: notes (**) and (***) name engines, but neither the
    # 1.6 16V. A correct answer gives both values with their condition.
    {
        "id": "208-spark-plugs",
        "question": "¿Cada cuánto se cambian las bujías del Peugeot 208 1.6?",
        "route": "manual",
        "page": "MG-22",
        "facts": [
            "The spark plugs are replaced every 20,000 km or 4 years on the engines "
            "of note (**): 1.2 12V Nafta and 1.6 THP CDTX/FDTX/CDTR/FDTR.",
            "They are replaced every 40,000 km or 4 years on the engines of note "
            "(***): Motores Nafta and 1.6 THP 16V CDTM/FDTM.",
            "The answer does not state that one of the two intervals is the one for "
            "the 1.6 16V.",
        ],
    },
    # The recommended products table (p. 39) gives the interval too.
    {
        "id": "208-brake-fluid",
        "question": "¿Cada cuánto se cambia el líquido de frenos del Peugeot 208?",
        "route": "manual",
        "page": ("MG-22", "MG-39"),
        "facts": ["The brake fluid is replaced every 2 years, regardless of mileage."],
    },
    {
        "id": "208-service-price",
        "question": "¿Cuánto cuesta el service de los 10.000 km del Peugeot 208?",
        "route": "web",
        "page": None,
        "facts": [],
    },
]

BELTS = CASES[0]
SEVERE_USE = CASES[1]
BRAKE_FLUID = CASES[2]
TIRE_PRESSURE_208 = CASES[6]
SPARK_PLUGS_208 = CASES[8]

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
    # An agent answer the judge once passed: it lists other items but never the
    # engine oil and oil filter.
    (
        "partial item list",
        SEVERE_USE,
        "Sí, según el manual del Fiat Cronos 2021, si usás tu vehículo como remis, "
        "debés hacer el servicio más seguido. El plan de mantenimiento incluye una "
        'nota que dice: "Elementos que se deben sustituir/controlar en la mitad de '
        "los plazos indicados, para vehículos utilizados sobre todo en caminos "
        "polvorientos, arenosos, con barro o en condiciones severas de uso "
        '(remolque, taxi, entrega a domicilio, etc.)" (Manual del Fiat Cronos 2021, '
        "p. G-14). Esto significa que para varios componentes como el cartucho del "
        "filtro de aire del motor, el filtro de combustible, las pastillas de freno, "
        "las zapatas de freno traseras y el filtro de aire acondicionado, el "
        "intervalo se reduce a la mitad.",
        [True, False],
    ),
    # Two Sonnet answers the judge failed: both say the belts' interval is halved
    # under severe use, one in passing and one with a figure it says it computed.
    (
        "halving in passing",
        BELTS,
        "Según el manual, las correas de los órganos auxiliares del motor (las de "
        "accesorios) se sustituyen **cada 48 meses o cada 60.000 km, lo que ocurra "
        "primero** (Manual del Fiat Cronos 2021, p. G-14).\n\n"
        "Otros datos del plan de mantenimiento:\n"
        "- **Control visual:** se hace en las revisiones de 20.000, 40.000, 80.000, "
        "100.000, 140.000 y 160.000 km (Manual del Fiat Cronos 2021, p. G-14).\n"
        "- **Uso severo:** con remolque, caminos polvorientos, arenosos o con barro, "
        "o uso severo similar, el manual indica controlar la correa y el tensor con "
        "más frecuencia. En caminos polvorientos, arenosos o con barro pide revisar "
        "el rodamiento del tensor y las correas (aire acondicionado, bomba de agua, "
        "alternador) cada 10.000 km o 12 meses y cambiarlas si hace falta (Manual "
        "del Fiat Cronos 2021, p. G-14 y p. G-18).\n"
        "- **Intervalos a la mitad:** una nota del plan indica que ciertos "
        "elementos, entre ellos esta sustitución, se cambian o controlan en la "
        "mitad de los plazos para uso severo o inactividad prolongada (Manual del "
        "Fiat Cronos 2021, p. G-14).",
        [True, True],
    ),
    (
        "computed halving",
        BELTS,
        "Según el manual, la correa de accesorios (llamada \"correas de los órganos "
        "auxiliares del motor\") se **sustituye cada 48 meses o cada 60.000 km, lo "
        "que ocurra primero** (Manual del Fiat Cronos 2021, p. G-14).\n\n"
        "Otros datos del mismo plan de mantenimiento:\n"
        "- **Control visual:** se hace en las revisiones de 20.000, 40.000, 80.000, "
        "100.000, 140.000 y 160.000 km (Manual del Fiat Cronos 2021, p. G-14).\n"
        "- **Caminos polvorientos, arenosos o con barro:** el manual indica "
        "controlar el rodamiento del tensor y las correas (aire acondicionado, "
        "bomba de agua, alternador) cada 10.000 km o 12 meses. Si hace falta, se "
        "reemplazan (Manual del Fiat Cronos 2021, p. G-14).\n"
        "- **Uso severo** (remolque, taxi, entregas, etc.): el manual indica que "
        "los elementos marcados con (*) se sustituyen o controlan a la mitad de los "
        "plazos (Manual del Fiat Cronos 2021, p. G-14). Para las correas de "
        "accesorios, eso equivale a unos 24 meses o 30.000 km. Esa cifra es mi "
        "cálculo a partir de esa regla. El manual no la da expresamente.",
        [True, True],
    ),
    # The psi figures converted with convert_units, as the system prompt asks.
    (
        "pressures in kPa",
        TIRE_PRESSURE_208,
        "Con llantas 16 (195/55 R16), adelante van 200 kPa con el conductor y un "
        "acompañante, 228 kPa con más ocupantes y equipaje, y 248 kPa para reducir "
        "el consumo. Atrás van 200 kPa con el conductor y un acompañante, y 248 kPa "
        "con más ocupantes y equipaje o para reducir el consumo (Manual del Peugeot "
        "208 2024, p. 211).",
        [True, True],
    ),
    (
        "both plug intervals",
        SPARK_PLUGS_208,
        "La tabla da dos intervalos para la columna de motores nafta 1.2 y 1.6: "
        "20.000 km o 4 años para los motores de la nota (**), que son 1.2 12V Nafta "
        "y 1.6 THP CDTX/FDTX/CDTR/FDTR, y 40.000 km o 4 años para los de la nota "
        "(***), Motores Nafta y 1.6 THP 16V CDTM/FDTM. El manual no aclara cuál "
        "corresponde al 1.6 16V: confirmalo con tu versión de motor en un "
        "concesionario (Manual del Peugeot 208 2024, p. MG-22).",
        [True, True, True],
    ),
    (
        "plug interval picked",
        SPARK_PLUGS_208,
        "El manual da 20.000 km o 4 años para los motores de la nota (**), 1.2 12V "
        "Nafta y 1.6 THP CDTX/FDTX/CDTR/FDTR, y 40.000 km o 4 años para los de la "
        "nota (***), Motores Nafta y 1.6 THP 16V CDTM/FDTM. Para tu 1.6 16V "
        "corresponde cambiarlas cada 40.000 km o 4 años (Manual del Peugeot 208 "
        "2024, p. MG-22).",
        [True, True, False],
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
- A fact that names several items is stated only if every one of them is in the
  answer. Other items in the answer don't make up for a missing one.
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
    verdict = ask_judge(prompt, FactsVerdict, FACTS_JUDGE)
    if verdict is None or len(verdict.facts) != len(case["facts"]):
        return None
    return [fact.stated for fact in verdict.facts]


def calibrate() -> None:
    print(f"Judge: {judge_model(FACTS_JUDGE)}")
    for label, case, answer, expected in CALIBRATION:
        got = judge_facts(case, answer)
        status = "OK" if got == expected else "MISMATCH"
        print(f"{status:<9} {label:<20} expected {expected} got {got}")


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

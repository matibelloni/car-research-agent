"""Eval of answer quality on general car questions.

Two independent LLM-judge criteria, scored 1-5:
- citations: every concrete data point has its source right after it.
- comparability: compared figures share the same unit and measurement basis.

Usage:
    python -m evals.answers               # run the suite
    python -m evals.answers --calibrate   # check the judge on hand-written answers
"""

import statistics
import sys

from pydantic import BaseModel, Field
from rich import print as rprint

from agent import langfuse, run_agent
from evals.judge import ask_judge, judge_model

RUNS = 3

QUESTIONS = [
    "Is a 2015 Golf or a 2016 Focus better for city driving?",
    "What are the common problems with the 1.6 THP engine?",
    "How much does a 2018 Corolla use in the city?",
    "What should I check on a car with 150,000 km?",
    "Is the Peugeot 208 reliable?",
]


class Evaluation(BaseModel):
    # Reasoning first: fields are generated in order, so the score comes
    # after the analysis instead of being justified after the fact.
    reasoning: str = Field(description="Brief analysis that leads to the score")
    score: int = Field(description="Score from 1 to 5")


CITATIONS_PROMPT = """You are a strict evaluator of research assistant answers.

CRITERION: every concrete data point must have its source right after it.
Concrete data points are figures (fuel consumption, measurements, prices,
years, mileage) and recall facts (dates, defects, affected models). A figure
inside advice is still a data point (e.g. "change the chain every 60,000 km").
General advice without figures (e.g. "check the oil") is NOT a data point
and needs no source. A count of items that are each cited (e.g. "2 recalls"
followed by a citation for each one) needs no source of its own.

Valid sources:
- For facts from the web: the source URL, placed right after the data point.
  Each figure needs its own URL right after it, even when several figures come
  from the same source: one URL after the last of several figures covers only
  that last one. A URL in a later sentence does NOT count for an earlier
  figure, and a list of URLs at the end of the answer counts for none.
- For recalls from the Argentine consumer protection agency: "Defensa del
  Consumidor" plus the publication date. These have no URL, and that's fine.

First list every data point and whether a valid source sits right after it.
Then score:
5 = every data point has a valid source right after it, OR the answer
    contains no data points at all
3 = at least one data point has a valid source and at least one doesn't
1 = there are data points and none has a valid source right after it
    (this includes an answer with a single, unsourced data point)

QUESTION: {question}

ANSWER TO EVALUATE:
{answer}"""

COMPARABILITY_PROMPT = """You are evaluating whether an answer makes valid comparisons.

CRITERION: when comparing two things, the figures must be comparable —
same unit, same measurement basis, same conditions.

Common violations:
- Comparing curb weight against gross vehicle weight
- Mixing metric and imperial (25 mpg vs 9.2 L/100km)
- Comparing city fuel economy against combined
- Different model trims presented as equivalent

Score:
5 = all comparisons use equivalent bases
3 = minor inconsistencies
1 = comparisons that mislead the reader

QUESTION: {question}
ANSWER: {answer}"""

GOLF = "https://example.com/golf"
FOCUS = "https://example.com/focus"

# Hand-written answers with a known score. If the judge disagrees, fix the
# prompt before trusting it on the agent. Clear-cut cases expect an exact
# score; in-between ones accept a range, since the rubric only anchors 1, 3, 5.
# Each case: (label, question, answer, (min_score, max_score)).
CITATIONS_CALIBRATION = [
    (
        "no sources",
        "Which uses less fuel?",
        "El Golf consume 7.3 l/100km y el Focus 8.1.",
        (1, 1),
    ),
    (
        "inline sources",
        "Which uses less fuel?",
        f"El Golf consume 7.3 l/100km [{GOLF}]. El Focus consume 8.1 l/100km [{FOCUS}].",
        (5, 5),
    ),
    (
        "urls at the end",
        "Which uses less fuel?",
        f"El Golf consume 7.3 l/100km y el Focus 8.1.\n\nFuentes: {GOLF}, {FOCUS}",
        (1, 1),
    ),
    (
        "no data points",
        "What should I check on a used car?",
        "Revisá el aceite, el refrigerante y las pastillas de freno, "
        "y probá que los cambios entren suaves [https://example.com].",
        (5, 5),
    ),
    (
        "recall citation",
        "¿Hay recalls de airbags para el Gol?",
        "Hay 2 recalls de airbags para el Gol: uno publicado el "
        "29/01/2026 (Defensa del Consumidor, 29/01/2026) y otro el "
        "27/11/2025 (Defensa del Consumidor, 27/11/2025).",
        (5, 5),
    ),
    # Harder: the rubric says "right after it", not "somewhere nearby".
    (
        "one of two cited",
        "Which uses less fuel?",
        f"El Golf consume 7.3 l/100km [{GOLF}] y el Focus 8.1 l/100km.",
        (2, 4),
    ),
    (
        "one url for two figures",
        "Tell me about the 2015 Golf",
        f"El Golf 2015 consume 7.3 l/100km y pesa 1.270 kg [{GOLF}].",
        (2, 4),
    ),
    (
        "source in next sentence",
        "How much does the Golf use?",
        f"El Golf consume 7.3 l/100km en ciudad. Según {GOLF}, es de los más "
        "eficientes de su segmento.",
        (1, 3),
    ),
    (
        "figure hidden in advice",
        "What should I check on a 1.6 THP?",
        "Revisá la cadena de distribución, que en el THP conviene cambiar "
        "cada 60.000 km, y que no haya consumo de aceite.",
        (1, 2),
    ),
]

COMPARABILITY_CALIBRATION = [
    (
        "same basis",
        "Which uses less fuel in the city?",
        f"El Golf 2015 consume 7.3 l/100km en ciudad [{GOLF}] y el Focus 2016 "
        f"8.1 l/100km en ciudad [{FOCUS}], así que el Golf gasta menos.",
        (5, 5),
    ),
    (
        "imperial converted",
        "Which uses less fuel in the city?",
        f"El Golf consume 7.3 l/100km en ciudad [{GOLF}]. El Focus figura con "
        f"29 mpg (EE.UU.) en ciudad [{FOCUS}], que equivalen a 8.1 l/100km, así "
        "que el Golf gasta menos.",
        (5, 5),
    ),
    # Most of the eval questions ask about a single car: nothing to compare.
    (
        "no comparison",
        "How much does a 2018 Corolla use in the city?",
        "El Corolla 2018 consume 8.9 l/100km en ciudad [https://example.com/corolla].",
        (5, 5),
    ),
    (
        "mixed units",
        "Which uses less fuel?",
        f"El Golf consume 7.3 l/100km [{GOLF}] y el Focus 29 mpg [{FOCUS}], "
        "así que el Focus es más eficiente.",
        (1, 2),
    ),
    (
        "city vs combined",
        "Which uses less fuel in the city?",
        f"El Golf consume 7.3 l/100km en ciudad [{GOLF}] y el Focus 6.5 l/100km "
        f"combinado [{FOCUS}], así que el Focus gasta menos.",
        (1, 2),
    ),
    (
        "curb vs gross weight",
        "Which one is lighter?",
        f"El Golf pesa 1.270 kg en orden de marcha [{GOLF}] y el Focus 1.850 kg "
        f"de peso bruto [{FOCUS}], así que el Golf es mucho más liviano.",
        (1, 2),
    ),
    (
        "different trims as equivalent",
        "¿Qué consume menos, el Golf o el Focus?",
        f"El Golf consume 4.1 l/100km (versión 1.6 TDI diésel) [{GOLF}] y el "
        f"Focus 8.1 l/100km (versión 2.0 nafta) [{FOCUS}], así que el Golf "
        "consume la mitad.",
        (1, 2),
    ),
    (
        "minor inconsistency",
        "Compare the Golf and the Focus",
        f"Los dos consumen parecido en ciudad: 7.3 l/100km el Golf [{GOLF}] y "
        f"8.1 l/100km el Focus [{FOCUS}]. En potencia, el Golf tiene 122 CV "
        f"[{GOLF}] y el Focus 125 hp [{FOCUS}].",
        (2, 4),
    ),
]

# Judge per criterion, chosen on the calibration cases above: Haiku misreads
# where a URL sits and passes figures that share one URL, consistently across
# runs; Sonnet doesn't. On comparability Haiku matches the expected scores,
# while Sonnet docks points for details the answer never mentions.
CRITERIA = {
    "citations": (CITATIONS_PROMPT, "claude-sonnet-5-5"),
    "comparability": (COMPARABILITY_PROMPT, "claude-haiku-4-5"),
}

CALIBRATION = {
    "citations": CITATIONS_CALIBRATION,
    "comparability": COMPARABILITY_CALIBRATION,
}


def evaluate(question: str, answer: str, criterion: str) -> Evaluation | None:
    prompt, model = CRITERIA[criterion]
    return ask_judge(
        prompt.format(question=question, answer=answer), Evaluation, model=model
    )


def calibrate() -> None:
    mismatches = 0
    for criterion, cases in CALIBRATION.items():
        print(f"\n{criterion} (judge: {judge_model(CRITERIA[criterion][1])})")
        for label, question, answer, (low, high) in cases:
            ev = evaluate(question, answer, criterion)
            ok = ev is not None and low <= ev.score <= high
            mismatches += not ok
            expected = str(low) if low == high else f"{low}-{high}"
            got = ev.score if ev else None
            print(f"  {'OK' if ok else 'MISMATCH':<9} {label:<30} expected {expected:<4} got {got}")
            if not ok:
                print(f"    judge: {ev.reasoning if ev else 'judge failed'}")
    total = sum(len(cases) for cases in CALIBRATION.values())
    print(f"\n{total - mismatches}/{total} cases match")


def mean_or_none(values: list[float]) -> float | None:
    """None instead of 0 when nothing was scored, so it doesn't drag averages down."""
    return statistics.mean(values) if values else None


def single_run() -> dict:
    """Runs the questions once and returns the aggregated metrics."""
    citation_scores = []
    comparability_scores = []
    tokens = 0
    searches = 0
    agent_failures = 0
    judge_failures = 0

    for question in QUESTIONS:
        result = run_agent(question)
        # Failed runs spent tokens too: count them, or the cost looks lower than it is.
        tokens += result.tokens
        searches += result.searches
        if result.error:
            rprint(f"[red]⚠️  {question} → {result.error}[/red]")
            agent_failures += 1
            continue

        citations = evaluate(question, result.answer, "citations")
        comparability = evaluate(question, result.answer, "comparability")
        if citations is None or comparability is None:
            rprint(f"[yellow]⚠️  {question} → judge failed[/yellow]")
            judge_failures += 1
            continue

        citation_scores.append(citations.score)
        comparability_scores.append(comparability.score)
        uses_recalls = "Defensa del Consumidor" in result.answer
        rprint(
            f"  citations {citations.score}/5  recalls={uses_recalls}  {question[:50]}"
        )
        if citations.score <= 2:
            rprint(f"    judge: {citations.reasoning}")
            rprint(f"    answer: {result.answer[:500]}")

    return {
        "citations": mean_or_none(citation_scores),
        "comparability": mean_or_none(comparability_scores),
        "tokens": tokens,
        "searches": searches,
        "agent_failures": agent_failures,
        "judge_failures": judge_failures,
    }


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def summary_line(label: str, values: list[float | None], digits: int = 2) -> str:
    scored = [v for v in values if v is not None]
    if not scored:
        return f"{label:<14}--- n/a"
    return (
        f"{label:<14}--- mean: {statistics.mean(scored):.{digits}f} "
        f"--- range: {min(scored):.{digits}f} - {max(scored):.{digits}f}"
    )


def main() -> None:
    runs = []
    for i in range(RUNS):
        rprint(f"[bold]--- RUN {i + 1}/{RUNS} ---[/bold]")
        run = single_run()
        runs.append(run)
        rprint(
            f"Citations: {fmt(run['citations'])} | Comparability: {fmt(run['comparability'])} | "
            f"Tokens: {run['tokens']} | Agent failures: {run['agent_failures']} | "
            f"Judge failures: {run['judge_failures']}"
        )

    rprint("\n[bold]--- SUMMARY ---[/bold]")
    rprint(summary_line("Citations", [r["citations"] for r in runs]))
    rprint(summary_line("Comparability", [r["comparability"] for r in runs]))
    rprint(summary_line("Tokens", [r["tokens"] for r in runs], digits=0))
    langfuse.flush()


if __name__ == "__main__":
    if sys.argv[1:] == ["--calibrate"]:
        calibrate()
    else:
        main()

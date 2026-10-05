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
from evals.judge import ask_judge

RUNS = 3

QUESTIONS = [
    "Is a 2015 Golf or a 2016 Focus better for city driving?",
    "What are the common problems with the 1.6 THP engine?",
    "How much does a 2018 Corolla use in the city?",
    "What should I check on a car with 150,000 km?",
    "Is the Peugeot 208 reliable?",
]


class Evaluation(BaseModel):
    score: int = Field(description="Score from 1 to 5")
    reasoning: str = Field(description="Brief explanation of the score")


CITATIONS_PROMPT = """You are a strict evaluator of research assistant answers.

CRITERION: every concrete data point must have its source right after it.
Concrete data points are figures (fuel consumption, measurements, prices,
years, mileage) and recall facts (dates, defects, affected models).
General advice without figures (e.g. "check the oil") is NOT a data point
and needs no source.

Valid sources:
- For facts from the web: the source URL, placed right after the data point.
  A list of URLs at the end of the answer does NOT count for any data point.
- For recalls from the Argentine consumer protection agency: "Defensa del
  Consumidor" plus the publication date. These have no URL, and that's fine.

Score:
5 = every data point has a valid source right after it, OR the answer
    contains no data points at all
3 = some data points have it, others don't
1 = data points are present and none has a valid source next to it

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

# Hand-written answers with a known score. If the judge disagrees, fix the
# prompt before trusting it on the agent.
CALIBRATION = [
    ("BAD (no sources)", "El Golf consume 7.3 l/100km y el Focus 8.1.", 1),
    (
        "GOOD (inline)",
        "El Golf consume 7.3 l/100km [https://example.com/golf]. "
        "El Focus consume 8.1 l/100km [https://example.com/focus].",
        5,
    ),
    (
        "URLS AT END",
        "El Golf consume 7.3 l/100km y el Focus 8.1.\n\n"
        "Fuentes: https://example.com/golf, https://example.com/focus",
        1,
    ),
    (
        "NO DATA POINTS",
        "Revisá el aceite, el refrigerante y las pastillas de freno, "
        "y probá que los cambios entren suaves [https://example.com].",
        5,
    ),
    (
        "RECALL CITATION",
        "Hay 2 recalls de airbags para el Gol: uno publicado el "
        "29/01/2026 (Defensa del Consumidor, 29/01/2026) y otro el "
        "27/11/2025 (Defensa del Consumidor, 27/11/2025).",
        5,
    ),
]


def evaluate(question: str, answer: str, prompt: str) -> Evaluation | None:
    return ask_judge(prompt.format(question=question, answer=answer), Evaluation)


def calibrate() -> None:
    for label, answer, expected in CALIBRATION:
        ev = evaluate("Which uses less fuel?", answer, CITATIONS_PROMPT)
        status = "OK" if ev and ev.score == expected else "MISMATCH"
        print(
            f"{status:<9} {label:<18} expected {expected}, got {ev.score if ev else None} — {ev.reasoning if ev else ''}"
        )


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

        citations = evaluate(question, result.answer, CITATIONS_PROMPT)
        comparability = evaluate(question, result.answer, COMPARABILITY_PROMPT)
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

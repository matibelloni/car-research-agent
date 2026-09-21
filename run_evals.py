from agent import run_agent
from eval_answers import evaluate, CITATIONS_PROMPT, COMPARABILITY_PROMPT
from rich import print as rprint
import statistics

QUESTIONS = [
    "Is a 2015 Golf or a 2016 Focus better for city driving?",
    "What are the common problems with the 1.6 THP engine?",
    "How much does a 2018 Corolla use in the city?",
    "What should I check on a car with 150,000 km?",
    "Is the Peugeot 208 reliable?",
]


def single_run():
    """Runs the N questions once and returns the aggregated metrics."""
    citation_scores = []
    comparability_scores = []
    tokens = 0
    failures = 0
    searches = 0

    for question in QUESTIONS:
        result = run_agent(question)
        if result.error:
            rprint(f"[red]⚠️  {question} → {result.error}[/red]")
            failures += 1
            continue

        citations = evaluate(question, result.answer, CITATIONS_PROMPT)
        comparability = evaluate(question, result.answer, COMPARABILITY_PROMPT)

        if citations is None or comparability is None:
            failures += 1
            continue

        citation_scores.append(citations.score)
        comparability_scores.append(comparability.score)
        tokens += result.tokens
        searches += result.searches

    return {
        "citations": statistics.mean(citation_scores) if citation_scores else 0,
        "comparability": (
            statistics.mean(comparability_scores) if comparability_scores else 0
        ),
        "tokens": tokens,
        "searches": searches,
        "failures": failures,
    }


RUNS = 3
runs = []
if __name__ == "__main__":
    for i in range(RUNS):
        rprint(f"[bold]--- RUN {i + 1}/{RUNS} ---[/bold]")
        sr = single_run()
        runs.append(sr)
        rprint(
            f"Citations: {sr['citations']:.2f} | Comparability: {sr['comparability']:.2f} | "
            f"Tokens: {sr['tokens']} | Failures: {sr['failures']}"
        )
    citations = [r["citations"] for r in runs]
    comparability = [r["comparability"] for r in runs]
    tokens = [r["tokens"] for r in runs]

    rprint("\n[bold]--- SUMMARY ---[/bold]")
    rprint(
        f"Citations     --- mean: {statistics.mean(citations):.2f} --- range: {min(citations):.2f} - {max(citations):.2f}"
    )
    rprint(
        f"Comparability --- mean: {statistics.mean(comparability):.2f} --- range: {min(comparability):.2f} - {max(comparability):.2f}"
    )
    rprint(
        f"Tokens        --- mean: {statistics.mean(tokens):.0f} --- range: {min(tokens)} - {max(tokens)}"
    )

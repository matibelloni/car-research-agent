from agent import run_agent
from eval_answers import evaluate
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
    scores = []
    tokens = 0
    failures = 0
    searches = 0

    for question in QUESTIONS:
        result = run_agent(question)
        if result["error"]:
            failures += 1
            continue

        ev = evaluate(question, result["answer"])

        if ev is None:
            failures += 1
            continue

        scores.append(ev.score)
        tokens += result["tokens"]
        searches += result["searches"]

    return {
        "score": statistics.mean(scores) if scores else 0,
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
            f"Score: {sr["score"]:.2f} | Tokens: {sr["tokens"]} | Searches: {sr["searches"]} | Failures: {sr["failures"]}"
        )
    scores = [run["score"] for run in runs]
    tokens = [run["tokens"] for run in runs]

    rprint("\n[bold]--- SUMMARY ---[/bold]")
    rprint(
        f"Scores --- mean: {statistics.mean(scores):.2f} --- median:{statistics.median(scores):.2f} --- range: {min(scores):.2f} - {max(scores):.2f}"
    )
    rprint(
        f"Tokens --- mean: {statistics.mean(tokens):.0f} --- median:{statistics.median(tokens):.0f} --- range: {min(tokens)} - {max(tokens)}"
    )

import os
from dotenv import load_dotenv
from anthropic import Anthropic
from pydantic import BaseModel, Field

load_dotenv()
claude = Anthropic()


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


def evaluate(question: str, answer: str, prompt: str) -> Evaluation | None:

    # output_format goes with parse(), not with create()
    result = claude.messages.parse(
        model="claude-haiku-4-5",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": prompt.format(question=question, answer=answer),
            }
        ],
        output_format=Evaluation,
    )
    # With parse(), the result already comes typed in parsed_output:
    return result.parsed_output


if __name__ == "__main__":
    calibration = [
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

    for label, answer, expected in calibration:
        ev = evaluate("Which uses less fuel?", answer, CITATIONS_PROMPT)
        status = "OK" if ev and ev.score == expected else "MISMATCH"
        print(
            f"{status:<9} {label:<18} expected {expected}, got {ev.score if ev else None} — {ev.reasoning if ev else ''}"
        )

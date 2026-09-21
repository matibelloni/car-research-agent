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

CRITERION: every concrete numeric fact (fuel consumption, measurements,
prices, years) must be accompanied by its source URL.

Score:
5 = every numeric fact has a source
3 = some have it, others don't
1 = no fact has a source

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
    bad_answer = "El Golf consume 7.3 litros y el Focus 8.1. El Golf es mejor."

    good_answer = """El Golf consume 7.3 l/100km [https://example.com/golf].
    El Focus consume 8.1 l/100km [https://example.com/focus]."""

    for label, answer in [("BAD", bad_answer), ("GOOD", good_answer)]:
        question = "Which uses less fuel?"
        ev = evaluate(question, answer, CITATIONS_PROMPT)
        if ev is None:
            print(f"{label}: judge failed")
            continue
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")
        print(f"{label}: {ev.score}/5 — {ev.reasoning}\n")

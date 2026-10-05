"""LLM judge shared by the eval suites."""

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

JUDGE_MODEL = "claude-haiku-4-5"

claude = anthropic.Anthropic(max_retries=3)


def ask_judge[T: BaseModel](
    prompt: str, output_format: type[T], max_tokens: int = 500
) -> T | None:
    """Sends one prompt to the judge and returns its typed verdict."""
    response = claude.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        output_format=output_format,
    )
    return response.parsed_output

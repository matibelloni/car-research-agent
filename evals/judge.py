"""LLM judge shared by the eval suites."""

import os

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

DEFAULT_JUDGE = "claude-haiku-4-5"

# Forces one model for every criterion, to compare judges on the calibration
# cases, e.g.  JUDGE_MODEL=claude-sonnet-5-5 python -m evals.answers --calibrate
JUDGE_OVERRIDE = os.environ.get("JUDGE_MODEL")

# A ceiling, not a target: verdicts are short, but models that think by
# default (Sonnet 5.5) spend thinking tokens from this same budget, and a low
# cap truncates the JSON mid-string.
MAX_TOKENS = 16000

claude = anthropic.Anthropic(max_retries=3)


def judge_model(model: str = DEFAULT_JUDGE) -> str:
    return JUDGE_OVERRIDE or model


def ask_judge[T: BaseModel](
    prompt: str, output_format: type[T], model: str = DEFAULT_JUDGE
) -> T | None:
    """Sends one prompt to the judge and returns its typed verdict.

    None when the verdict can't be parsed, so callers count a judge failure
    instead of aborting the whole run.
    """
    try:
        response = claude.messages.parse(
            model=judge_model(model),
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            output_format=output_format,
        )
    except ValidationError:
        return None
    return response.parsed_output

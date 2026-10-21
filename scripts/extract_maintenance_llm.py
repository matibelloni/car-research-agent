"""Turn the scheduled-maintenance table of any manual into one sentence per task.

Unlike extract_maintenance.py, which reads Fiat's table grid, this sends the
plan's pages to Claude as a PDF and asks for each row's structure. The model
only reads the table: the sentences are built here, in the same format.

Usage:
    python -m scripts.extract_maintenance_llm <pdf path> <first page> <last page> [model]
    python -m scripts.extract_maintenance_llm --compare <pdf path> <first page> <last page> [model]
"""

import base64
import io
import sys
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pypdf import PdfReader, PdfWriter

from scripts.extract_maintenance import attach_footnotes, format_km

load_dotenv()

# Chosen on the Cronos, against the grid extractor: Sonnet matched all 27 rows
# in 3 of 3 runs; Haiku got the same 8 rows' revision columns wrong in every run.
DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 16000

claude = anthropic.Anthropic(max_retries=3)


class Task(BaseModel):
    text: str = Field(description="The row's task text, copied verbatim")
    schedule: Literal["all", "revisions", "interval", "footnote"] = Field(
        description=(
            "all: marked in every revision column. revisions: marked in some "
            "revision columns. interval: the row states its own interval in words "
            "instead of marks. footnote: a note that spans the whole table width."
        )
    )
    revisions: list[int] = Field(
        description="For 'revisions': the numbers of the marked revision columns"
    )
    interval: str | None = Field(
        description="For 'interval': the interval text, copied verbatim"
    )


class MaintenancePlan(BaseModel):
    header: str = Field(
        description="The text of the table's top-left header cell, copied verbatim"
    )
    revision_interval_km: int = Field(
        description="Kilometres between two consecutive revisions, from the header"
    )
    tasks: list[Task] = Field(description="Every row, in reading order across pages")


PROMPT = """The attached pages hold the scheduled-maintenance table of a car's \
owner's manual. Rows are tasks; most columns are numbered revisions, and a mark \
in a column means the task is done at that revision.

Return the table's structure:
- Copy every text verbatim, in the manual's language, including markers such as \
(*), typos and repeated words. Only join words hyphenated across line breaks and \
turn line breaks into spaces.
- When the table continues across pages and the header repeats, give the header once \
and every row once, in order.
- Footnotes under the table (explaining markers like (*)) are rows too, with \
schedule "footnote".
- Leave revisions empty and interval null when they don't apply."""


def pages_pdf(pdf_path: str, first_page: int, last_page: int) -> bytes:
    """A PDF with only the given pages (1-based, inclusive)."""
    if first_page > last_page:
        raise ValueError(f"Empty page range: {first_page} to {last_page}")
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    for number in range(first_page, last_page + 1):
        writer.add_page(reader.pages[number - 1])
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def extract_plan(
    pdf_path: str, first_page: int, last_page: int, model: str = DEFAULT_MODEL
) -> tuple[MaintenancePlan, int]:
    """The table's structure and the tokens it took to read it."""
    data = base64.standard_b64encode(pages_pdf(pdf_path, first_page, last_page))
    response = claude.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": data.decode("utf-8"),
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        output_format=MaintenancePlan,
    )
    if response.parsed_output is None:
        raise ValueError(f"No parsable plan (stop_reason={response.stop_reason})")
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return response.parsed_output, tokens


def task_to_sentence(task: Task, interval_km: int) -> str:
    if task.schedule == "footnote":
        return task.text
    text = task.text.rstrip(".")
    if task.schedule == "interval":
        return f"{text}: {task.interval}."
    if task.schedule == "all":
        return f"{text}: en todas las revisiones."
    points = ", ".join(f"{n}ª ({format_km(n * interval_km)} km)" for n in task.revisions)
    return f"{text}: en las revisiones {points}."


def plan_to_sentences(plan: MaintenancePlan) -> list[str]:
    return [plan.header] + [
        task_to_sentence(task, plan.revision_interval_km) for task in plan.tasks
    ]


def maintenance_sentences(
    pdf_path: str, first_page: int, last_page: int, model: str = DEFAULT_MODEL
) -> list[str]:
    plan, _ = extract_plan(pdf_path, first_page, last_page, model)
    return attach_footnotes(plan_to_sentences(plan))


def schedule_of(sentence: str) -> str:
    """The part after the task, e.g. 'en todas las revisiones.'"""
    return sentence.rpartition(": ")[2] if ": " in sentence else ""


def split_footnotes(sentences: list[str]) -> tuple[list[str], str]:
    """Rows, and the footnotes joined into one text without bullets.

    The grid extractor gets all footnotes as one cell, the model as one row
    each: joined, they compare as the same text.
    """
    start = next(
        (n for n, s in enumerate(sentences) if s.startswith("(*)")), len(sentences)
    )
    footnotes = " ".join(sentences[start:]).replace("● ", "")
    return sentences[:start], footnotes


def compare(pdf_path: str, first_page: int, last_page: int, model: str) -> None:
    """Diff this extractor against the table-grid one, line by line."""
    # Before attaching footnotes: this compares how each one reads the table.
    from scripts.extract_maintenance import table_sentences

    expected, expected_notes = split_footnotes(
        table_sentences(pdf_path, first_page, last_page)
    )
    plan, tokens = extract_plan(pdf_path, first_page, last_page, model)
    got, got_notes = split_footnotes(plan_to_sentences(plan))
    print(f"model {model}: {len(got)} rows vs {len(expected)} from the grid, {tokens} tokens")
    if expected_notes != got_notes:
        print(f"\n[FOOTNOTES]\n  grid: {expected_notes}\n  llm:  {got_notes}")

    same = schedule_diffs = text_diffs = 0
    for n in range(max(len(expected), len(got))):
        want = expected[n] if n < len(expected) else "<missing>"
        have = got[n] if n < len(got) else "<missing>"
        if want == have:
            same += 1
            continue
        if schedule_of(want) == schedule_of(have):
            text_diffs += 1
            kind = "TEXT"
        else:
            schedule_diffs += 1
            kind = "SCHEDULE"
        print(f"\n[{kind}] line {n + 1}\n  grid: {want}\n  llm:  {have}")
    print(
        f"\n{same} identical, {text_diffs} with different text only, "
        f"{schedule_diffs} with a different schedule, "
        f"footnotes {'identical' if expected_notes == got_notes else 'differ'}"
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--compare":
        path, first, last, *rest = args[1:]
        compare(path, int(first), int(last), rest[0] if rest else DEFAULT_MODEL)
    else:
        path, first, last, *rest = args
        sentences = maintenance_sentences(
            path, int(first), int(last), rest[0] if rest else DEFAULT_MODEL
        )
        print("\n".join(sentences))

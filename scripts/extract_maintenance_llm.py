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
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pypdf import PdfReader, PdfWriter

from brands import normalize
from scripts.extract_maintenance import attach_footnotes, format_km

load_dotenv()

# Chosen on the Cronos, against the grid extractor: Sonnet matched all 27 rows
# in 3 of 3 runs; Haiku got the same 8 rows' revision columns wrong in every run.
DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 16000

claude = anthropic.Anthropic(max_retries=3)


class EngineInterval(BaseModel):
    engines: list[str] = Field(
        description=(
            "The engine column headers this interval sits under, copied verbatim; or "
            "the engines the cell itself names. Empty when the table has no engine "
            "columns"
        )
    )
    interval: str = Field(
        description="The interval text, copied verbatim, e.g. 'cada 30.000 km'"
    )


class Task(BaseModel):
    text: str = Field(description="The row's task text, copied verbatim")
    schedule: Literal["all", "revisions", "interval", "footnote", "heading"] = Field(
        description=(
            "all: marked in every revision column. revisions: marked in some "
            "revision columns. interval: the row states its own interval in words "
            "instead of marks. footnote: a note that spans the whole table width. "
            "heading: a row that only names the group of tasks below it."
        )
    )
    revisions: list[int] = Field(
        description="For 'revisions': the numbers of the marked revision columns"
    )
    intervals: list[EngineInterval] = Field(
        description="For 'interval': one entry per distinct interval in the row"
    )


class MaintenancePlan(BaseModel):
    header: str = Field(
        description="The text of the table's top-left header cell, copied verbatim"
    )
    revision_interval_km: int | None = Field(
        description=(
            "Kilometres between two consecutive revisions, from the header; null "
            "when the table has no revision columns"
        )
    )
    engine_columns: list[str] = Field(
        description=(
            "The headers of the columns that split intervals by engine, copied "
            "verbatim; empty when there are none"
        )
    )
    tasks: list[Task] = Field(description="Every row, in reading order across pages")


PROMPT = """The attached pages hold the scheduled-maintenance table of a car's \
owner's manual. Rows are tasks. In some tables most columns are numbered \
revisions, and a mark in a column means the task is done at that revision. In \
others the columns are engines, and each cell states an interval in words.

Return the table's structure:
- Copy every text verbatim, in the manual's language, including markers such as \
(*), typos and repeated words. Only join words hyphenated across line breaks and \
turn line breaks into spaces.
- When the table continues across pages and the header repeats, give the header once \
and every row once, in order.
- With engine columns, give one interval per cell: a cell that spans several engine \
columns is one interval listing all of them. When a cell names engines itself \
(e.g. "Motor X - 120.000 km"), give one interval per engine it names, with those \
names as its engines.
- A cell that spans several rows applies to each of them: repeat it on every row.
- Footnotes under the table (explaining markers like (*)) are rows too, with \
schedule "footnote".
- Leave revisions and intervals empty when they don't apply."""


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


def intervals_text(intervals: list[EngineInterval], engine_columns: list[str]) -> str:
    """'cada 30.000 km' when one interval covers every engine, else one per engine."""
    if len(intervals) == 1 and set(intervals[0].engines) in (set(), set(engine_columns)):
        return intervals[0].interval
    return "; ".join(f"{i.interval} ({', '.join(i.engines)})" for i in intervals)


def task_to_sentence(task: Task, plan: MaintenancePlan) -> str:
    if task.schedule in ("footnote", "heading"):
        return task.text
    text = task.text.rstrip(".")
    if task.schedule == "interval":
        return f"{text}: {intervals_text(task.intervals, plan.engine_columns)}."
    if task.schedule == "all":
        return f"{text}: en todas las revisiones."
    points = ", ".join(
        f"{n}ª ({format_km(n * plan.revision_interval_km)} km)" for n in task.revisions
    )
    return f"{text}: en las revisiones {points}."


def plan_to_sentences(plan: MaintenancePlan) -> list[str]:
    return [plan.header] + [task_to_sentence(task, plan) for task in plan.tasks]


def filter_engines(plan: MaintenancePlan, engines: list[str]) -> MaintenancePlan:
    """Keep only the intervals of the given engines, named as given.

    An interval matches an engine when one of its engine names contains it, e.g.
    "1.6 16V" in "Motores Nafta 1.2 12V / 1.6 16V / 1.6 THP". Intervals with no
    engine names apply to every engine and stay. A row left with no interval is
    dropped, and engines that end up with the same interval are merged.
    """

    def matches(names: list[str]) -> list[str]:
        haystack = [" ".join(normalize(name).split()) for name in names]
        return [e for e in engines if any(normalize(e) in h for h in haystack)]

    tasks = []
    for task in plan.tasks:
        if task.schedule != "interval":
            tasks.append(task)
            continue
        by_interval: dict[str, list[str]] = {}
        for interval in task.intervals:
            names = matches(interval.engines) if interval.engines else []
            if interval.engines and not names:
                continue
            kept = by_interval.setdefault(interval.interval, [])
            kept += [name for name in names if name not in kept]
        if by_interval:
            intervals = [
                EngineInterval(engines=names, interval=text)
                for text, names in by_interval.items()
            ]
            tasks.append(task.model_copy(update={"intervals": intervals}))
    return plan.model_copy(
        update={"engine_columns": matches(plan.engine_columns), "tasks": tasks}
    )


def maintenance_sentences(
    pdf_path: str, first_page: int, last_page: int, model: str = DEFAULT_MODEL
) -> list[str]:
    plan, _ = extract_plan(pdf_path, first_page, last_page, model)
    return attach_footnotes(plan_to_sentences(plan))


def load_or_extract_plan(
    pdf_path: Path, first_page: int, last_page: int, cache_path: Path
) -> MaintenancePlan:
    """The plan as the model read it, saved so that rechunking doesn't call it again."""
    if cache_path.exists():
        return MaintenancePlan.model_validate_json(cache_path.read_text(encoding="utf-8"))
    plan, tokens = extract_plan(str(pdf_path), first_page, last_page)
    cache_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    print(f"{cache_path.name}: extracted with {DEFAULT_MODEL}, {tokens} tokens")
    return plan


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

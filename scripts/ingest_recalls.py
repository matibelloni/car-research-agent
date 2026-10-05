"""Ingest vehicle recalls from the Defensa del Consumidor public sheet.

Downloads the sheet (or reads a local CSV), keeps vehicle recalls,
resolves dates written in two formats, and maps each publisher to the
vehicle brands it may cover.
"""

import csv
import io
import json
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from brands import brands_for, normalize

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "recalls_clean.json"
SHEET_ID = "1eUlKJFk-TYgZcmNd1-OMkMKSpeIOyNIgL4frzkKUnWs"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
HEADER_ROWS = 2


@dataclass
class Recall:
    row: int  # row number as shown in Google Sheets, for traceability
    raw_date: str
    date: str | None  # ISO format, or None if missing/invalid
    date_inferred: bool  # True when day/month order was ambiguous
    company: str
    brands: list[str]  # brands this publisher may cover
    product: str
    defect: str
    risk: str


def is_vehicle(category: str) -> bool:
    return normalize(category).startswith("vehiculos")


def split_date(raw: str) -> tuple[int, int, int] | None:
    try:
        year, a, b = [int(part) for part in raw.strip().split("/")]
        return year, a, b
    except ValueError:
        return None


def parse_date(raw_date: str, index: int, switch_index: int) -> tuple[str | None, bool]:
    parts = split_date(raw_date)
    if parts is None:
        return None, False

    year, a, b = parts
    inferred = a != b and a <= 12 and b <= 12
    month, day = (a, b) if index < switch_index else (b, a)

    try:
        return date(year, month, day).isoformat(), inferred
    except ValueError:
        return None, False


def load_rows(source: str) -> list[list[str]]:
    if source.startswith("http"):
        with urllib.request.urlopen(source) as response:
            text = response.read().decode("utf-8")
    else:
        text = Path(source).read_text(encoding="utf-8")
    return list(csv.reader(io.StringIO(text)))[HEADER_ROWS:]


def find_format_switch(rows: list[list[str]]) -> int:
    """Index of the first row written as year/DAY/month."""
    for i, row in enumerate(rows):
        parts = split_date(row[1])
        if parts and parts[1] > 12:
            return i
    return len(rows)


def find_format_violations(
    rows: list[list[str]], switch_index: int
) -> tuple[list[dict], list[dict]]:
    """Unambiguous dates that contradict a single clean format switch."""
    before = []
    after = []
    for i, row in enumerate(rows):
        parts = split_date(row[1])
        if parts is None:
            continue
        _, a, b = parts
        if i < switch_index and a > 12:
            before.append({"index": i, "date": row[1]})
        elif i >= switch_index and b > 12:
            after.append({"index": i, "date": row[1]})
    return before, after


def main(source: str = CSV_URL) -> None:
    rows = load_rows(source)
    switch_index = find_format_switch(rows)

    before, after = find_format_violations(rows, switch_index)
    if before or after:
        raise ValueError(f"Date format violations: before={before}, after={after}")

    recalls = []
    unmatched_companies = set()

    for i, row in enumerate(rows):
        if not is_vehicle(row[0]):
            continue

        sheet_row = i + HEADER_ROWS + 1
        company = row[2].strip()
        parsed_date, inferred = parse_date(row[1], i, switch_index)
        brands = brands_for(company)

        if parsed_date is None:
            print(f"Row {sheet_row}: missing or invalid date {row[1]!r}")
        if not brands:
            unmatched_companies.add(company)

        recalls.append(
            Recall(
                row=sheet_row,
                raw_date=row[1].strip(),
                date=parsed_date,
                date_inferred=inferred,
                company=company,
                brands=brands,
                product=row[3].strip(),
                defect=row[4].strip(),
                risk=row[5].strip(),
            )
        )

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(
        json.dumps([asdict(r) for r in recalls], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Source rows:        {len(rows)}")
    print(f"Format switch at:   index {switch_index}")
    print(f"Vehicle recalls:    {len(recalls)}")
    print(f"  missing date:     {sum(r.date is None for r in recalls)}")
    print(f"  inferred date:    {sum(r.date_inferred for r in recalls)}")
    print(f"  without brand:    {sum(not r.brands for r in recalls)}")
    for company in sorted(unmatched_companies):
        print(f"    unmatched company: {company!r}")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CSV_URL)

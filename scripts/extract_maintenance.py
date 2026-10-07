"""Turn the scheduled-maintenance tables of a manual into one sentence per task.

Usage: python scripts/extract_maintenance.py <pdf path> <first page> <last page>
"""

import re
import sys

import pdfplumber

TABLE_SETTINGS = {"text_x_tolerance": 1}  # the default merges words in this PDF
HEADER_PREFIX = "REVISIONES"


def clean(text: str) -> str:
    return re.sub(r"-\n", "", text).replace("\n", " ").strip()


def format_km(km: int) -> str:
    return f"{km:,}".replace(",", ".")


def row_to_sentence(header: list, row: list) -> str:
    task = clean(row[0])
    cells = row[1:]
    if all(cell is None for cell in cells):  # footnotes span the whole row
        return task
    task = task.rstrip(".")
    if cells[0] and cells[0] != "+":  # interval written out, e.g. "cada 30.000 km"
        return f"{task}: {clean(cells[0])}."
    interval_km = int(
        re.search(r"Cada ([\d.]+) km", header[0]).group(1).replace(".", "")
    )
    revisions = [n for n, cell in enumerate(cells, start=1) if cell == "+"]
    if len(revisions) == len(cells):
        return f"{task}: en todas las revisiones."
    points = ", ".join(f"{n}ª ({format_km(n * interval_km)} km)" for n in revisions)
    return f"{task}: en las revisiones {points}."


def maintenance_sentences(pdf_path: str, first_page: int, last_page: int) -> list[str]:
    if first_page > last_page:
        raise ValueError(f"Empty page range: {first_page} to {last_page}")
    sentences = []
    with pdfplumber.open(pdf_path) as pdf:
        for number in range(first_page, last_page + 1):
            for table in pdf.pages[number - 1].extract_tables(TABLE_SETTINGS):
                header, *rows = table
                if not (header[0] or "").startswith(HEADER_PREFIX):
                    continue
                if not sentences:
                    sentences.append(clean(header[0]))
                sentences += [row_to_sentence(header, row) for row in rows]
    return sentences


if __name__ == "__main__":
    path, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    print("\n".join(maintenance_sentences(path, first, last)))

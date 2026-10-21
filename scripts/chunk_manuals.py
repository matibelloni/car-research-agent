"""Split the extracted manual pages into one chunk per section heading.

Usage: python -m scripts.chunk_manuals
"""

import json
import re
import statistics
from pathlib import Path

from scripts.extract_maintenance import attach_footnotes, maintenance_sentences
from scripts.extract_maintenance_llm import (
    filter_engines,
    load_or_extract_plan,
    plan_to_sentences,
)

MANUALS_DIR = Path(__file__).resolve().parent.parent / "data" / "manuals"
MAX_CHUNK_CHARS = 2000

# Per manual:
# - skip_lines: lines that aren't content (page numbers, chapter tabs).
# - page_label: regex for the printed page number, read from the page's lines.
# - page_numbering: for manuals whose numbering restarts in each booklet, ranges
#   of PDF pages (first, last, printed number of the first, label template).
#   Pages outside every range get no label.
# - maintenance: PDF pages (first, last) of the scheduled-maintenance table,
#   which is replaced by one sentence per task, read from the table grid
#   ("grid", Fiat's layout only) or by Claude ("llm", cached next to the PDF).
# - engines: the engines the car has, when the table has a column per engine.
# - excluded_pages: PDF pages about other vehicles.
MANUALS = {
    "fiat-cronos-2021": {
        "skip_lines": [
            re.compile(r"^[A-I]-\d+$"),  # printed page number, e.g. "B-33"
            re.compile(r"^[A-I]$"),  # chapter letter printed on the page edge
        ],
        "page_label": re.compile(r"^[A-I]-\d+$"),
        "page_numbering": [],
        "maintenance": {
            "pages": (254, 257),
            "extractor": "grid",
            "heading": "PLAN DE MANTENIMIENTO PROGRAMADO",
        },
        "engines": [],
        "excluded_pages": [],
    },
    "peugeot-208-2024": {
        # Page numbers, chapter numbers on the page edge, figure callouts like "21 38".
        "skip_lines": [re.compile(r"^[\d .]{1,7}$")],
        "page_label": None,
        "page_numbering": [
            (5, 218, 1, "{}"),
            (219, 238, 1, "Audio y Telemática, p. {}"),
            (239, 254, 1, "Suplemento, p. {}"),
            (258, 315, 2, "Mantenimiento y Garantía, p. {}"),
        ],
        "maintenance": {
            "pages": (278, 283),
            "extractor": "llm",
            "heading": "Mantenimiento periódico - Vehículos Particulares + Partner",
        },
        # The ones the body of the manual mentions; the maintenance booklet
        # covers every Peugeot.
        "engines": ["1.6 16V", "1.0 Turbo 200"],
        # Hybrids, utility vans, Expert and Boxer.
        "excluded_pages": range(285, 294),
    },
}

DOT_LEADERS = re.compile(r"(\. ){3,}")  # table of contents and index lines
CALLOUTS = {"ADVERTENCIA", "¡ATENCIÓN!", "NOTA"}  # uppercase labels, not headings


def page_label(config: dict, page: dict) -> str | None:
    for first, last, start, template in config["page_numbering"]:
        if first <= page["page"] <= last:
            return template.format(page["page"] - first + start)
    if config["page_label"] is None:
        return None
    lines = [line.strip() for line in page["text"].splitlines()]
    return next((line for line in lines if config["page_label"].match(line)), None)


def is_heading(line: str, previous_line: str) -> bool:
    if line in CALLOUTS or line != line.upper() or line.startswith(("●", "■")):
        return False
    if not re.search(r"[A-ZÁÉÍÓÚÑ]{4}", line):
        return False
    if line.endswith(".") or '"' in line:  # uppercase tail of a normal sentence
        return False
    # second half of a word hyphenated on a body line, e.g. "MANTENI-" / "MIENTO Y ..."
    return not (previous_line.endswith("-") and previous_line != previous_line.upper())


def join_lines(lines: list[str]) -> str:
    text = ""
    for line in lines:
        text = text[:-1] + line if text.endswith("-") else f"{text} {line}"
    return text.strip()


def split_text(text: str) -> list[str]:
    """Pack whole sentences into parts of at most MAX_CHUNK_CHARS."""
    parts, current = [], ""
    for sentence in re.split(r"(?<=\.) ", text):
        if current and len(current) + len(sentence) + 1 > MAX_CHUNK_CHARS:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    return parts + [current] if current else parts


def is_noise(line: str, config: dict) -> bool:
    return bool(DOT_LEADERS.search(line)) or any(
        pattern.match(line) for pattern in config["skip_lines"]
    )


def chunk_pages(pages: list[dict], config: dict) -> list[dict]:
    chunks = []
    current = None
    previous_line = ""
    last_was_heading = False
    for page in pages:
        lines = [line.strip() for line in page["text"].splitlines() if line.strip()]
        label = page_label(config, page)
        for line in lines:
            if is_noise(line, config):
                continue
            continues_heading = last_was_heading and previous_line.endswith("-")
            if continues_heading or is_heading(line, previous_line):
                if last_was_heading:  # heading wrapped over several lines
                    current["heading_lines"].append(line)
                else:
                    current = {
                        "heading_lines": [line],
                        "body_lines": [],
                        "page": page["page"],
                        "page_label": label,
                    }
                    chunks.append(current)
                last_was_heading = True
            else:
                if current is not None:
                    current["body_lines"].append(line)
                last_was_heading = False
            previous_line = line
    return [
        {
            "heading": join_lines(chunk["heading_lines"]),
            "text": part,
            "page": chunk["page"],
            "page_label": chunk["page_label"],
        }
        for chunk in chunks
        for part in split_text(join_lines(chunk["body_lines"]))
    ]


def maintenance_chunk(manual: str, config: dict, pages: list[dict]) -> dict:
    maintenance = config["maintenance"]
    first, last = maintenance["pages"]
    pdf_path = MANUALS_DIR / f"{manual}.pdf"
    if maintenance["extractor"] == "grid":
        sentences = maintenance_sentences(str(pdf_path), first, last)
    else:
        cache_path = MANUALS_DIR / f"{manual}.maintenance.json"
        plan = load_or_extract_plan(pdf_path, first, last, cache_path)
        if config["engines"]:
            plan = filter_engines(plan, config["engines"])
        sentences = attach_footnotes(plan_to_sentences(plan))
    # One chunk on purpose: the interval rule and the footnotes apply to every row.
    return {
        "heading": maintenance["heading"],
        "text": "\n".join(sentences),
        "page": first,
        "page_label": page_label(config, pages[first - 1]),
    }


def print_stats(name: str, chunks: list[dict]) -> None:
    lengths = [len(chunk["text"]) for chunk in chunks]
    print(f"{name}: {len(chunks)} chunks")
    print(
        f"  chars per chunk: min {min(lengths)}, "
        f"median {statistics.median(lengths):.0f}, max {max(lengths)}"
    )
    print(f"  under 100 chars: {sum(1 for n in lengths if n < 100)}")
    print(
        f"  over {MAX_CHUNK_CHARS} chars: {sum(1 for n in lengths if n > MAX_CHUNK_CHARS)}"
    )
    print("\n  10 longest:")
    for chunk in sorted(chunks, key=lambda c: len(c["text"]), reverse=True)[:10]:
        print(f"    {len(chunk['text']):>6}  {chunk['page_label']}  {chunk['heading']}")
    print("\n  every 20th chunk:")
    for chunk in chunks[::20]:
        print(f"    {len(chunk['text']):>6}  {chunk['page_label']}  {chunk['heading']}")


def main() -> None:
    for pages_path in sorted(MANUALS_DIR.glob("*.pages.json")):
        pages = json.loads(pages_path.read_text(encoding="utf-8"))
        manual = pages_path.name.removesuffix(".pages.json")
        config = MANUALS[manual]
        first, last = config["maintenance"]["pages"]
        prose = [
            page
            for page in pages
            if not first <= page["page"] <= last
            and page["page"] not in config["excluded_pages"]
        ]
        chunks = chunk_pages(prose, config) + [maintenance_chunk(manual, config, pages)]
        chunks_path = pages_path.with_name(
            pages_path.name.replace(".pages.", ".chunks.")
        )
        chunks_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print_stats(pages_path.name, chunks)


if __name__ == "__main__":
    main()

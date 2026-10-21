"""Split the extracted manual pages into one chunk per section heading.

Usage: python -m scripts.chunk_manuals
"""

import bisect
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from manuals import MANUALS
from scripts.extract_maintenance import attach_footnotes, maintenance_sentences
from scripts.extract_maintenance_llm import (
    filter_engines,
    load_or_extract_plan,
    plan_to_sentences,
)
from scripts.extract_manuals import load_or_extract_lines

MANUALS_DIR = Path(__file__).resolve().parent.parent / "data" / "manuals"
MAX_CHUNK_CHARS = 2000

DOT_LEADERS = re.compile(r"(\. ){3,}")  # table of contents and index lines
CALLOUTS = {"ADVERTENCIA", "¡ATENCIÓN!", "NOTA"}  # uppercase labels, not headings
# A line repeated this often among the first lines of a page is the running header.
RUNNING_HEADER_PAGES = 5
RUNNING_HEADER_POSITION = 3


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


class FontRules:
    """Headings, page numbers and running headers, told apart by font.

    The body is the most frequent size; a heading is bold and bigger than it.
    Page numbers are digit-only lines in the font most such lines use, and a
    running header is a line repeated near the top of many pages in one font.
    """

    def __init__(self, pages: list[dict]):
        sizes, number_fonts, tops = Counter(), Counter(), Counter()
        for page in pages:
            for position, line in enumerate(page["lines"]):
                sizes[line["size"]] += len(line["text"])
                if line["text"].isdigit():
                    number_fonts[line["font"]] += 1
                if position < RUNNING_HEADER_POSITION:
                    tops[self.key(line)] += 1
        self.body_size = sizes.most_common(1)[0][0]
        self.page_number_font = number_fonts.most_common(1)[0][0] if number_fonts else None
        self.running_headers = {k for k, n in tops.items() if n >= RUNNING_HEADER_PAGES}

    @staticmethod
    def key(line: dict) -> tuple:
        return line["text"], line["font"], line["size"]

    def is_noise(self, line: dict, position: int) -> bool:
        if line["text"].isdigit() and line["font"] == self.page_number_font:
            return True
        return position < RUNNING_HEADER_POSITION and self.key(line) in self.running_headers

    def is_heading(self, line: dict) -> bool:
        return "bold" in line["font"].lower() and line["size"] > self.body_size


def manual_lines(manual: str, config: dict, pages: list[dict]) -> list[dict]:
    """Per page, its lines as {"text", "font", "size"}; no fonts for "uppercase"."""
    if config["headings"] == "font":
        return load_or_extract_lines(MANUALS_DIR / f"{manual}.pdf")
    return [
        {
            "page": page["page"],
            "lines": [
                {"text": line.strip(), "font": None, "size": None}
                for line in page["text"].splitlines()
                if line.strip()
            ],
        }
        for page in pages
    ]


def join_lines(lines: list[str]) -> tuple[str, list[int]]:
    """The lines as one text, and where each line starts in it."""
    text, starts = "", []
    for line in lines:
        if text.endswith("-"):  # word hyphenated across lines
            text = text[:-1]
        elif text:
            text += " "
        starts.append(len(text))
        text += line
    return text, starts


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


def chunk_pages(pages: list[dict], lines: list[dict], config: dict) -> list[dict]:
    """Chunks of the given pages; each part of a section gets the page it starts on."""
    by_number = {page["page"]: page for page in lines}
    rules = FontRules([by_number[p["page"]] for p in pages]) if config["headings"] == "font" else None
    sections = []
    current = None
    previous_line = ""
    last_was_heading = False
    for page in pages:
        label = page_label(config, page)
        for position, line in enumerate(by_number[page["page"]]["lines"]):
            text = line["text"]
            if DOT_LEADERS.search(text) or any(p.match(text) for p in config["skip_lines"]):
                continue
            if rules and rules.is_noise(line, position):
                continue
            if rules:
                heading = rules.is_heading(line)
            else:
                continues_heading = last_was_heading and previous_line.endswith("-")
                heading = continues_heading or is_heading(text, previous_line)
            if heading:
                if last_was_heading:  # heading wrapped over several lines
                    current["heading_lines"].append(text)
                else:
                    current = {"heading_lines": [text], "body": []}
                    sections.append(current)
                last_was_heading = True
            else:
                if current is not None:
                    current["body"].append((text, page["page"], label))
                last_was_heading = False
            previous_line = text
    chunks = []
    for section in sections:
        heading, _ = join_lines(section["heading_lines"])
        text, starts = join_lines([line for line, _, _ in section["body"]])
        cursor = 0
        for part in split_text(text):
            cursor = text.find(part, cursor)
            _, page, label = section["body"][bisect.bisect_right(starts, cursor) - 1]
            chunks.append({"heading": heading, "text": part, "page": page, "page_label": label})
            cursor += len(part)
    return chunks


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
        lines = manual_lines(manual, config, pages)
        chunks = chunk_pages(prose, lines, config) + [
            maintenance_chunk(manual, config, pages)
        ]
        chunks_path = pages_path.with_name(
            pages_path.name.replace(".pages.", ".chunks.")
        )
        chunks_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print_stats(pages_path.name, chunks)


if __name__ == "__main__":
    main()

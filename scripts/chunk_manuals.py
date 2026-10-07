"""Split the extracted manual pages into one chunk per section heading.

Usage: python scripts/chunk_manuals.py
"""

import json
import re
import statistics
from pathlib import Path

MANUALS_DIR = Path(__file__).resolve().parent.parent / "data" / "manuals"
MAX_CHUNK_CHARS = 2000

PAGE_LABEL = re.compile(r"^[A-I]-\d+$")  # printed page number, e.g. "B-33"
CHAPTER_TAB = re.compile(r"^[A-I]$")  # chapter letter printed on the page edge
DOT_LEADERS = re.compile(r"(\. ){3,}")  # table of contents and index lines
CALLOUTS = {"ADVERTENCIA", "¡ATENCIÓN!", "NOTA"}  # uppercase labels, not headings


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


def chunk_pages(pages: list[dict]) -> list[dict]:
    chunks = []
    current = None
    previous_line = ""
    last_was_heading = False
    for page in pages:
        lines = [line.strip() for line in page["text"].splitlines() if line.strip()]
        label = next((line for line in lines if PAGE_LABEL.match(line)), None)
        for line in lines:
            if (
                PAGE_LABEL.match(line)
                or CHAPTER_TAB.match(line)
                or DOT_LEADERS.search(line)
            ):
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
        chunks = chunk_pages(pages)
        chunks_path = pages_path.with_name(
            pages_path.name.replace(".pages.", ".chunks.")
        )
        chunks_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print_stats(pages_path.name, chunks)


if __name__ == "__main__":
    main()

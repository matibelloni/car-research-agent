"""Extract per-page text from the downloaded manuals and print basic stats.

Usage: python scripts/extract_manuals.py [page numbers to print]
"""

import json
import statistics
import sys
from pathlib import Path

from pypdf import PdfReader

MANUALS_DIR = Path(__file__).resolve().parent.parent / "data" / "manuals"
SHORT_PAGE_CHARS = 50


def extract_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(pdf_path)
    return [
        {"page": number, "text": page.extract_text()}
        for number, page in enumerate(reader.pages, start=1)
    ]


def load_or_extract(pdf_path: Path) -> list[dict]:
    pages_path = pdf_path.with_suffix(".pages.json")
    if pages_path.exists():
        return json.loads(pages_path.read_text(encoding="utf-8"))
    pages = extract_pages(pdf_path)
    pages_path.write_text(
        json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return pages


def print_stats(name: str, pages: list[dict]) -> None:
    lengths = [len(page["text"]) for page in pages]
    short_pages = sum(1 for length in lengths if length < SHORT_PAGE_CHARS)
    print(f"{name}: {len(pages)} pages")
    print(f"  pages under {SHORT_PAGE_CHARS} chars: {short_pages}")
    print(
        f"  chars per page: min {min(lengths)}, "
        f"median {statistics.median(lengths):.0f}, max {max(lengths)}"
    )


def print_sample_pages(pages: list[dict], page_numbers: list[int]) -> None:
    for number in page_numbers:
        print(f"\n===== page {number} =====")
        print(pages[number - 1]["text"])


def main() -> None:
    sample_pages = [int(arg) for arg in sys.argv[1:]]
    for pdf_path in sorted(MANUALS_DIR.glob("*.pdf")):
        pages = load_or_extract(pdf_path)
        print_stats(pdf_path.name, pages)
        print_sample_pages(pages, sample_pages)


if __name__ == "__main__":
    main()

"""Per-manual configuration, shared by the chunker and the search_manual tool.

Per manual:
- title: how answers cite it, e.g. "Manual del Fiat Cronos 2021".
- headings: "uppercase" (a line in capitals) or "font" (bold and bigger than
  the body text, read from the PDF's fonts).
- skip_lines: lines that aren't content (page numbers, chapter tabs).
- page_label: regex for the printed page number, read from the page's lines.
- page_numbering: for manuals whose numbering restarts in each booklet, ranges
  of PDF pages (first, last, printed number of the first, label template).
  Pages outside every range get no label.
- maintenance: PDF pages (first, last) of the scheduled-maintenance table,
  which is replaced by one sentence per task, read from the table grid
  ("grid", Fiat's layout only) or by Claude ("llm", cached next to the PDF).
- engines: the engines the car has, when the table has a column per engine.
- excluded_pages: PDF pages that aren't about using this car.
"""

import re


def pages(*ranges: tuple[int, int]) -> set[int]:
    return {page for first, last in ranges for page in range(first, last + 1)}


MANUALS = {
    "fiat-cronos-2021": {
        "title": "Manual del Fiat Cronos 2021",
        "headings": "uppercase",
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
        "excluded_pages": set(),
    },
    "peugeot-208-2024": {
        "title": "Manual del Peugeot 208 2024",
        "headings": "font",
        # Chapter numbers on the page edge and figure callouts like "21 38".
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
        "excluded_pages": pages(
            (1, 4),  # cover, online guide, company legal data
            (255, 257),  # radio homologations, booklet cover
            (259, 259),  # first buyer form
            (285, 293),  # maintenance of hybrids, utility vans, Expert and Boxer
            (297, 298),  # recommended products for Expert and Boxer
            (299, 302),  # service record forms
            (304, 307),  # repair, airbag and odometer forms; change of owner (with Portuguese)
            (316, 324),  # blank pages and back cover
        ),
    },
}


def citation(manual: str, page_label: str | None) -> str:
    """How an answer cites a section, e.g. "Manual del Fiat Cronos 2021, p. G-14"."""
    title = MANUALS[manual]["title"]
    if page_label is None:
        return title
    # Booklet labels already say where the page is: "Mantenimiento y Garantía, p. 22".
    return f"{title}, {page_label}" if ", p. " in page_label else f"{title}, p. {page_label}"

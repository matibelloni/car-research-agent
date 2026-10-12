"""List the chunks that contain a phrase.

Usage: python scripts/find_chunks.py "<phrase>"
"""

import json
import sys
from pathlib import Path

MANUALS_DIR = Path(__file__).resolve().parent.parent / "data" / "manuals"

phrase = sys.argv[1].lower()
for chunks_path in sorted(MANUALS_DIR.glob("*.chunks.json")):
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    matches = [chunk for chunk in chunks if phrase in chunk["text"].lower()]
    print(f"{chunks_path.name}: {len(matches)} chunks contain {sys.argv[1]!r}")
    for chunk in matches:
        print(f"  {chunk['page_label']}  {chunk['heading']}")

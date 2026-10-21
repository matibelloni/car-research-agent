"""Embed manual chunks with Voyage and load them into pgvector.

Usage: python -m scripts.load_manual [manual id ...]   # every manual when none is given
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import psycopg
import voyageai
from dotenv import load_dotenv

load_dotenv()

MANUALS_DIR = Path(__file__).resolve().parent.parent / "data" / "manuals"
MODEL = "voyage-4"
EMBEDDING_BATCH = 100

INSERT_SQL = """
    INSERT INTO manual_chunks (manual_id, heading, content, page, page_label, embedding)
    VALUES (%s, %s, %s, %s, %s, %s::vector)
"""


def embedding_text(chunk: dict) -> str:
    """The exact string the retrieval eval embedded."""
    return f"{chunk['heading']}. {chunk['text']}"


def embed_chunks(client: voyageai.Client, chunks: list[dict]) -> list[list[float]]:
    embeddings = []
    for start in range(0, len(chunks), EMBEDDING_BATCH):
        batch = [
            embedding_text(chunk) for chunk in chunks[start : start + EMBEDDING_BATCH]
        ]
        result = client.embed(batch, model=MODEL, input_type="document")
        embeddings += result.embeddings
    return embeddings


def main(manual_ids: list[str]) -> None:
    """Embeds and loads the given manuals, or every one when none is given."""
    client = voyageai.Client()  # reads VOYAGE_API_KEY
    paths = [MANUALS_DIR / f"{m}.chunks.json" for m in manual_ids] or sorted(
        MANUALS_DIR.glob("*.chunks.json")
    )
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for path in paths:
            manual_id = path.name.removesuffix(".chunks.json")
            chunks = json.loads(path.read_text(encoding="utf-8"))

            embeddings = embed_chunks(client, chunks)
            for chunk, embedding in zip(chunks, embeddings):
                norm = np.linalg.norm(embedding)
                assert abs(norm - 1) < 1e-6, f"norm {norm} on page {chunk['page']}"

            rows = [
                (manual_id, c["heading"], c["text"], c["page"], c["page_label"], str(e))
                for c, e in zip(chunks, embeddings)
            ]
            with conn.transaction():
                conn.execute(
                    "DELETE FROM manual_chunks WHERE manual_id = %s", (manual_id,)
                )
                with conn.cursor() as cursor:
                    cursor.executemany(INSERT_SQL, rows)

            print(f"{manual_id}: {len(rows)} chunks loaded")


if __name__ == "__main__":
    main(sys.argv[1:])

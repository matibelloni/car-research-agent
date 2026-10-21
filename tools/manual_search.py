import os

import psycopg
import voyageai

from manuals import citation

EMBEDDING_MODEL = "voyage-4"
RERANK_MODEL = "rerank-3"
RERANK_CANDIDATES = 30
TOP_K = 3
MIN_RELEVANCE = 0.5

FIRST_STAGE_SQL = """
    SELECT heading, content, page, page_label, embedding <=> %s::vector AS distance
    FROM manual_chunks
    WHERE manual_id = %s
    ORDER BY distance
    LIMIT %s
"""

AVAILABLE_MANUALS = ["fiat-cronos-2021", "peugeot-208-2024"]

MANUAL_SEARCH_TOOL = {
    "name": "search_manual",
    "description": (
        "Searches the official owner's manual of a car sold in Argentina and returns "
        "the most relevant sections, each with its heading, text and citation. Use it "
        "before searching the web for questions the manual can answer: maintenance "
        "intervals, warning lights, fluids and capacities, tire pressures, and how to "
        "use the car's features. The manual never contains prices, owner opinions, "
        "reliability data or comparisons with other cars: for those, use search_web "
        "directly. Only the manuals listed in 'manual' are available. "
        "The sections are related to the question but may not contain the answer: if "
        "none of them answers it, say the manual does not cover it and search the web. "
        "Never fill gaps with general knowledge as if it came from the manual. Cite "
        "every section you use with its citation, exactly as given."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "manual": {
                "type": "string",
                "enum": AVAILABLE_MANUALS,
                "description": "The manual to search, as 'brand-model-year'.",
            },
            "query": {
                "type": "string",
                "description": (
                    "The question in Spanish, since the manual is in Spanish. Keep "
                    "the user's wording and ask about one topic per call."
                ),
            },
        },
        "required": ["manual", "query"],
    },
}


def search_manual_tool(manual: str, query: str) -> dict:
    if manual not in AVAILABLE_MANUALS:
        return {
            "error": "unknown_manual",
            "message": f"Manual '{manual}' not available.",
            "available_manuals": AVAILABLE_MANUALS,
        }
    # The citation is built here, so the model copies it instead of composing it
    # from the manual's id and a page label.
    sections = [
        {
            "heading": section["heading"],
            "content": section["content"],
            "citation": citation(manual, section["page_label"]),
        }
        for section in search_manual(query, manual)
    ]
    if not sections:
        return {
            "sections": [],
            "message": "The manual has no section about this. Search the web instead.",
        }
    return {"sections": sections}


def first_stage(
    conn, question_vector: list[float], manual_id: str, limit: int
) -> list[tuple]:
    return conn.execute(
        FIRST_STAGE_SQL, (str(question_vector), manual_id, limit)
    ).fetchall()


def search_manual(question: str, manual_id: str) -> list[dict]:
    client = voyageai.Client()  # reads VOYAGE_API_KEY
    question_vector = client.embed(
        [question], model=EMBEDDING_MODEL, input_type="query"
    ).embeddings[0]

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        candidates = first_stage(conn, question_vector, manual_id, RERANK_CANDIDATES)
    if not candidates:
        return []

    texts = [f"{heading}. {content}" for heading, content, *_ in candidates]
    reranked = client.rerank(question, texts, model=RERANK_MODEL)

    results = []
    for item in reranked.results[:TOP_K]:
        if item.relevance_score < MIN_RELEVANCE:
            break
        heading, content, page, page_label, _ = candidates[item.index]
        results.append(
            {
                "heading": heading,
                "content": content,
                "page": page,
                "page_label": page_label,
                "relevance_score": item.relevance_score,
            }
        )
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    questions = [
        "cada cuánto hay que cambiar el líquido de frenos",
        "a los cuántos km se cambia la correa de accesorios",
        "uso el auto de remis, tengo que hacerle el service más seguido?",
        "cuánto cuesta el service de los 10.000 km",
        "qué opinan los dueños del consumo en ruta",
    ]
    for question in questions:
        print(f"\n{question}")
        for result in search_manual(question, "fiat-cronos-2021"):
            print(
                f"  {result['relevance_score']:.3f}  {result['page_label']}  {result['heading']}"
            )

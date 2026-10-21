import os

import psycopg
import voyageai

from manuals import MANUALS, citation, resolve_manual

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

MANUAL_SEARCH_TOOL = {
    "name": "search_manual",
    "description": (
        "Searches the official owner's manual of a car sold in Argentina and returns "
        "the most relevant sections, each with its heading, text and citation. Use it "
        "before searching the web for questions the manual can answer: maintenance "
        "intervals, warning lights, fluids and capacities, tire pressures, and how to "
        "use the car's features. The manual never contains prices, owner opinions, "
        "reliability data or comparisons with other cars: for those, use search_web "
        "directly. Only some models have a manual: for any other, the result says so "
        "and lists the available ones, and you search the web instead. The result "
        "names the manual it used; when it has a note about the model year, tell the "
        "user that the manual is for another year, or that they gave none. "
        "The sections are related to the question but may not contain the answer: if "
        "none of them answers it, say the manual does not cover it and search the web. "
        "Never fill gaps with general knowledge as if it came from the manual. Cite "
        "every section you use with its citation, exactly as given."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": (
                    "The car model as the user names it, with the brand if they "
                    "give it, e.g. 'Cronos', 'Peugeot 208'."
                ),
            },
            "year": {
                "type": "integer",
                "description": "Optional. The model year the user asks about.",
            },
            "query": {
                "type": "string",
                "description": (
                    "The question in Spanish, since the manual is in Spanish. Keep "
                    "the user's wording and ask about one topic per call."
                ),
            },
        },
        "required": ["model", "query"],
    },
}


def search_manual_tool(model: str, query: str, year: int | None = None) -> dict:
    resolved = resolve_manual(model, year)
    if "error" in resolved:
        return resolved
    manual = resolved["manual"]
    used = {"manual": MANUALS[manual]["title"]}
    if resolved["note"]:
        used["note"] = resolved["note"]
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
            **used,
            "sections": [],
            "message": "The manual has no section about this. Search the web instead.",
        }
    return {**used, "sections": sections}


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

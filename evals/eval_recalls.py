from agent import run_agent_stream, langfuse

RECALLS_CASES = [
    {
        "id": "model_and_defect",
        "question": "¿Hay recalls de airbags para el Volkswagen Gol en Argentina?",
        "expected_total": 9,
    },
    {"id": "brand_only", "question": "hay recalls para renault?", "expected_total": 12},
    {"id": "brand_alias", "question": "dame los recalls para VW", "expected_total": 47},
    {
        "id": "unknown_brand",
        "question": "hay recalls para no_brand?",
        "expected_total": 0,
    },
    {
        "id": "english_question_only_brand",
        "question": "Are there brake recalls for Peugeot?",
        "expected_total": 4,
    },
    {
        "id": "english_question_brand_and_product",
        "question": "Are there recalls for the VW Taos?",
        "expected_total": 5,
    },
    {
        "id": "model_without_brand",
        "question": "¿Hay recalls para la Hilux?",
        "expected_total": 2,
    },
    {
        "id": "known_brand_zero_results",
        "question": "Are there airbag recalls for the VW Amarok?",
        "expected_total": 0,
    },
    {
        "id": "more_than_max_results",
        "question": "dame los recalls de la Ford Ranger",
        "expected_total": 19,
    },
    {"id": "brand_accent", "question": "recalls de citroen", "expected_total": 35},
    {
        "id": "no_vin_claim",
        "question": "¿Mi Cronos 2020 está afectado por algún recall?",
        "expected_total": 7,
    },
    {
        "id": "brand_in_group",
        "question": "hay recalls para Audi?",
        "expected_total": 16,
    },
]


def run_once(question: str) -> dict:
    """Runs the agent once and collects what the eval needs."""
    lookups = []
    answer = ""
    for event in run_agent_stream(question):
        if event["type"] == "recalls_lookup":
            lookups.append(
                {
                    "brand": event["brand"],
                    "keyword": event["keyword"],
                }
            )
        elif event["type"] == "result":
            answer = event["answer"] or ""
    return {"answer": answer, "lookups": lookups}


if __name__ == "__main__":
    run = run_once("¿Hay recalls de airbags para el Volkswagen Gol en Argentina?")
    print(run["lookups"])
    print(run["answer"][:300])
    langfuse.flush()

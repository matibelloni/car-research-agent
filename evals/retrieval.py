"""Retrieval eval: does the chunk holding the answer rank near the top?

Usage: python -m evals.retrieval
"""

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from functools import cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MANUALS_DIR = Path(__file__).resolve().parent.parent / "data" / "manuals"
RECALL_AT = (3, 5)  # 3 is how many sections search_manual returns
BM25_K1 = 1.5  # how fast repeated terms stop adding score
BM25_B = 0.75  # how much long chunks are penalized
EMBEDDING_BATCH = 100  # chunks per request, well under the API's per-request limits
RRF_K = 60  # dampens the gap between the top positions
RRF_DEPTH = 50  # candidates taken from each list; below that, BM25 order is noise
RERANK_CANDIDATES = 30  # first-stage results the reranker reads and reorders

# "evidence" is a phrase copied from the manual. A chunk is relevant when its heading
# or text contains it. A case may list several phrases when more than one section of
# the manual answers the question; every extra phrase was added after reading the
# chunk, and applies to every retriever alike. None means the manual has no answer.
CASES = [
    {
        "id": "spark-plugs",
        "question": "cada cuánto se cambian las bujías del 1.3 naftero",
        "evidence": "cada 30.000 km",
    },
    {
        "id": "brake-fluid",
        "question": "cada cuánto hay que cambiar el líquido de frenos",
        "evidence": "cada 24 meses o cada 40.000 km",
    },
    {
        "id": "accessory-belts",
        "question": "a los cuántos km se cambia la correa de accesorios",
        "evidence": "cada 48 meses o cada 60.000 km",
    },
    {
        "id": "coolant",
        "question": "cuándo se cambia el refrigerante",
        "evidence": "cada 10 años o cada 240.000 km",
    },
    {
        "id": "severe-use",
        "question": "uso el auto de remis, tengo que hacerle el service más seguido?",
        "evidence": ["taxi, entrega a domicilio", "USO SEVERO DEL VEHÍCULO"],
    },
    {
        "id": "oil-filter",
        "question": "si cambio el aceite tengo que cambiar también el filtro?",
        "evidence": "cambiarse cada vez que se cambia el aceite del motor",
    },
    {
        "id": "key-battery",
        "question": "cómo le cambio la pila al control de la llave",
        "evidence": "Girar la tapa de la batería",
    },
    {
        "id": "child-lock",
        "question": "cómo trabo las puertas de atrás para que los chicos no las abran",
        "evidence": "Impide la apertura de las puertas traseras",
    },
    {
        "id": "trunk-emergency",
        "question": "cómo abro el baúl desde adentro si no anda la cerradura",
        "evidence": "mover la traba de apertura",
    },
    {
        "id": "open-hood",
        "question": "cómo se abre el capó",
        "evidence": "liberar la traba de cierre del capó",
    },
    {
        "id": "window-reset",
        "question": "desconecté la batería y ahora los vidrios no suben solos de un toque",
        "evidence": "funcionamiento automático de los levantavidrios",
    },
    {
        "id": "follow-me-home",
        "question": "cómo hago para que las luces queden prendidas un rato cuando me bajo",
        "evidence": ["Follow me home se activa", "Headlights Off Delay"],
    },
    {
        "id": "parking-lights",
        "question": "se pueden dejar las luces de posición prendidas con el auto cerrado y sin la llave?",
        "evidence": "presencia del vehículo de noche",
    },
    {
        "id": "trip-reset",
        "question": "cómo pongo en cero el consumo promedio",
        "evidence": "Para reiniciar los datos, pulsar",
    },
    {
        "id": "power-outlet",
        "question": "cuántos watts aguanta la toma de 12 voltios",
        "evidence": "180 W de potencia",
    },
    {
        "id": "service-reminder",
        "question": "me aparece un aviso de service en el tablero, quién lo borra?",
        "evidence": "puesta a cero (reset) de los contadores",
    },
    {
        "id": "steering-warning",
        "question": "se me prendió una luz de la dirección, qué hago?",
        "evidence": "AVERÍA EN LA DIRECCIÓN ASISTIDA",
    },
    {
        "id": "transmission-hot",
        "question": "me salta un aviso de temperatura de la caja automática",
        "evidence": [
            "ALTA TEMPERATURA DE LA TRANSMISIÓN",
            "TEMPERATURA EXCESIVA DEL ACEITE DEL CAMBIO",
        ],
    },
    {
        "id": "service-price",
        "question": "cuánto sale el service de los 20.000 km",
        "evidence": None,
    },
    {
        "id": "fuel-filter",
        "question": "cada cuánto se cambia el filtro de nafta",
        "evidence": "filtro de combustible. (*): en las revisiones",
    },
    # Written as an owner would ask, with Argentine words, before reading the
    # manual: the cases above borrowed its vocabulary. The evidence was found
    # afterwards.
    {
        "id": "owner-engine-light",
        "question": "se me prendió una luz amarilla con forma de motor en el tablero, puedo seguir andando?",
        "evidence": [
            "En estas condiciones, se puede seguir manejando",
            "Si la luz de aviso queda prendida o si se enciende durante la marcha",
        ],
    },
    # The manual gives the same pressure at half and full load.
    {
        "id": "owner-tire-pressure-load",
        "question": "cuánto aire le pongo a las gomas si salgo a la ruta con la familia y el baúl lleno?",
        "evidence": "Media carga Plena carga",
    },
    # The manual doesn't describe the clicking; jump-starting is the closest answer.
    # "MOTOR NO ARRANCA" turned up reviewing the top 5 after the first run: it
    # sends a flat battery to a jump-start too.
    {
        "id": "owner-no-start",
        "question": "el auto no arranca y hace tic tic tic, qué hago?",
        "evidence": [
            "ARRANQUE CON BATERÍA AUXILIAR",
            "En caso de batería descargada, es posible realizar un arranque de emergencia",
        ],
    },
    # The manual gives the octane rating, not "súper" or "premium".
    {
        "id": "owner-fuel-octane",
        "question": "qué nafta le cargo, súper o premium?",
        "evidence": [
            "octano (RON) no inferior a 95",
            "índice de octano no inferior a 95",
        ],
    },
    {
        "id": "owner-trailer",
        "question": "puedo enganchar un trailer? cuánto peso aguanta?",
        "evidence": [
            "Cargas remolcables (remolque sin freno) 400",
            "GANCHO DE REMOLQUE REMOLQUE DE ACOPLADOS",
        ],
    },
    {
        "id": "owner-spare-wheel",
        "question": "cómo saco la rueda de auxilio?",
        "evidence": [
            "Quitar la rueda de repuesto del interior del compartimiento de equipajes",
            "La rueda de repuesto está ubicada debajo de la alfombra",
        ],
    },
    {
        "id": "owner-tire-rotation",
        "question": "cada cuánto hay que rotar las cubiertas?",
        "evidence": "Cada 10.000 km, garantizar la rotación de los neumáticos",
    },
    # Peugeot 208. Some phrases keep the PDF's word breaks ("refri geración"),
    # since they must match the chunk's text.
    {
        "id": "208-tire-pressure",
        "manual": "peugeot-208-2024",
        "question": "qué presión llevan los neumáticos del 208",
        "evidence": "185/65 R15 88H 29 (1) 33 (2) 36 (3)",
    },
    {
        "id": "208-accessory-belt",
        "manual": "peugeot-208-2024",
        "question": "cada cuántos km se cambia la correa de accesorios",
        "evidence": "Sustitución de la correa de los órgaños auxiliares del motor*: 120.000 km",
    },
    {
        "id": "208-brake-fluid",
        "manual": "peugeot-208-2024",
        "question": "cada cuánto se cambia el líquido de frenos",
        "evidence": "Sustitución del fluido de los frenos: 2 años",
    },
    {
        "id": "208-severe-use",
        "manual": "peugeot-208-2024",
        "question": "uso el auto de remis, tengo que hacerle el service más seguido?",
        "evidence": "remolque, taxi, entregas puerta a puerta",
    },
    {
        "id": "208-oil-consumption",
        "manual": "peugeot-208-2024",
        "question": "cuánto aceite es normal que consuma el motor",
        "evidence": "aproximada mente 400 g",
    },
    {
        "id": "208-coolant-capacity",
        "manual": "peugeot-208-2024",
        "question": "cuántos litros de refrigerante lleva el motor",
        "evidence": "Capacidad del sistema de refri geración",
    },
    {
        "id": "208-fuel-tank",
        "manual": "peugeot-208-2024",
        "question": "de cuántos litros es el tanque",
        "evidence": "Capacidad del depósito: aproximadamente 47 litros",
    },
    {
        "id": "208-warranty",
        "manual": "peugeot-208-2024",
        "question": "cuánto dura la garantía del auto",
        "evidence": "treinta y seis (36) meses o 100.000 km",
    },
    # The heading: Android Auto's section mentions it too, without the ®.
    {
        "id": "208-bluetooth",
        "manual": "peugeot-208-2024",
        "question": "cómo conecto el celular por bluetooth",
        "evidence": "Vincular un teléfono Bluetooth®",
    },
    {
        "id": "208-engine-light",
        "manual": "peugeot-208-2024",
        "question": "se prendió y titila la luz de motor, qué hago",
        "evidence": "Fallo del sistema de gestión del motor",
    },
]

# Cases without "manual" are about the Cronos. Each manual is searched on its
# own, as search_manual does.
DEFAULT_MANUAL = "fiat-cronos-2021"


def load_chunks(manual: str) -> list[dict]:
    path = MANUALS_DIR / f"{manual}.chunks.json"
    return json.loads(path.read_text(encoding="utf-8"))


def is_relevant(chunk: dict, evidence: str | list[str]) -> bool:
    phrases = [evidence] if isinstance(evidence, str) else evidence
    text = f"{chunk['heading']} {chunk['text']}".lower()
    return any(phrase.lower() in text for phrase in phrases)


def tokenize(text: str) -> list[str]:
    """Lowercase words without accents, so "bujías" matches "bujias"."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", without_accents)


def bm25_retriever(chunks: list[dict]):
    """Return a search function: question -> chunk positions, best first."""
    documents = [Counter(tokenize(f"{c['heading']} {c['text']}")) for c in chunks]
    lengths = [sum(document.values()) for document in documents]
    average_length = sum(lengths) / len(documents)
    document_frequency = Counter(term for document in documents for term in document)
    total = len(documents)

    def search(question: str) -> list[int]:
        scores = [0.0] * total
        for term in set(tokenize(question)):
            containing = document_frequency.get(term)
            if not containing:
                continue
            rarity = math.log(1 + (total - containing + 0.5) / (containing + 0.5))
            for position, document in enumerate(documents):
                count = document[term]
                if count:
                    length_penalty = (
                        1 - BM25_B + BM25_B * lengths[position] / average_length
                    )
                    scores[position] += (
                        rarity
                        * count
                        * (BM25_K1 + 1)
                        / (count + BM25_K1 * length_penalty)
                    )
        return sorted(range(total), key=lambda position: scores[position], reverse=True)

    return search


def voyage_retriever(chunks: list[dict], model: str):
    """Return a search function backed by Voyage embeddings (cached on disk)."""
    import numpy as np
    import voyageai

    client = voyageai.Client()  # reads VOYAGE_API_KEY
    texts = [f"{chunk['heading']}. {chunk['text']}" for chunk in chunks]
    cache_key = hashlib.sha256("\n".join([model, *texts]).encode()).hexdigest()[:16]
    cache_path = MANUALS_DIR / f"embeddings-{cache_key}.npy"
    if cache_path.exists():
        vectors = np.load(cache_path)
    else:
        embeddings, tokens = [], 0
        for start in range(0, len(texts), EMBEDDING_BATCH):
            batch = texts[start : start + EMBEDDING_BATCH]
            result = client.embed(batch, model=model, input_type="document")
            embeddings += result.embeddings
            tokens += result.total_tokens
        vectors = np.array(embeddings)
        np.save(cache_path, vectors)
        print(f"{model}: embedded {len(texts)} chunks, {tokens} tokens")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    @cache
    def embed_query(question: str):
        result = client.embed([question], model=model, input_type="query")
        return np.array(result.embeddings[0])

    def search(question: str) -> list[int]:
        scores = vectors @ embed_query(question)
        return [int(position) for position in scores.argsort()[::-1]]

    return search


def hybrid_retriever(*searches):
    """Reciprocal Rank Fusion: combine rankings by position, ignoring raw scores."""

    def search(question: str) -> list[int]:
        scores = Counter()
        for ranked in (search_one(question) for search_one in searches):
            for position, index in enumerate(ranked[:RRF_DEPTH], start=1):
                scores[index] += 1 / (RRF_K + position)
        return [index for index, _ in scores.most_common()]

    return search


def reranked_retriever(first_stage, chunks: list[dict], model: str):
    """Reorder the first stage's top candidates with a Voyage reranker."""
    import voyageai

    client = voyageai.Client()
    texts = [f"{chunk['heading']}. {chunk['text']}" for chunk in chunks]
    usage = {"tokens": 0}

    def search(question: str) -> list[int]:
        ranked = first_stage(question)
        candidates = ranked[:RERANK_CANDIDATES]
        result = client.rerank(question, [texts[i] for i in candidates], model=model)
        usage["tokens"] += result.total_tokens
        reordered = [candidates[item.index] for item in result.results]
        return reordered + ranked[RERANK_CANDIDATES:]

    search.usage = usage
    return search


def evaluate(name: str, search, chunks: list[dict], cases: list[dict]) -> None:
    ranks = []
    print(f"== {name} ==")
    for case in cases:
        if case["evidence"] is None:  # needs a "nothing found" rule, measured later
            continue
        ranked = search(case["question"])
        rank = next(
            (
                position
                for position, index in enumerate(ranked, start=1)
                if is_relevant(chunks[index], case["evidence"])
            ),
            None,
        )
        ranks.append(rank)
        label = f"rank {rank:>3}" if rank else "NO CHUNK"
        print(f"  {label}  {case['id']}")
    found = [rank for rank in ranks if rank]
    recalls = "   ".join(
        f"recall@{k}: {sum(1 for rank in found if rank <= k) / len(ranks):.2f}"
        for k in RECALL_AT
    )
    mrr = sum(1 / rank for rank in found) / len(ranks)
    print(f"  {recalls}   MRR: {mrr:.2f}   ({len(ranks)} cases)")


if __name__ == "__main__":
    for manual in dict.fromkeys(case.get("manual", DEFAULT_MANUAL) for case in CASES):
        print(f"\n######## {manual}")
        cases = [c for c in CASES if c.get("manual", DEFAULT_MANUAL) == manual]
        chunks = load_chunks(manual)
        bm25 = bm25_retriever(chunks)
        voyage = voyage_retriever(chunks, "voyage-4")
        evaluate("BM25", bm25, chunks, cases)
        evaluate("voyage-4", voyage, chunks, cases)
        evaluate("hybrid (RRF)", hybrid_retriever(bm25, voyage), chunks, cases)
        reranked = reranked_retriever(voyage, chunks, "rerank-3")
        evaluate("voyage-4 + rerank-3", reranked, chunks, cases)
        print(f"  rerank tokens: {reranked.usage['tokens']}")

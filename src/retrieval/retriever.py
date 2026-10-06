from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Callable, Sequence

import asyncpg
from requests import RequestException

from config.logger import setup_logger
from config.settings import DB_PARAMS, VECTOR_OPERATOR
from ingestion.embedding.embedder import Embedder
from ingestion.metadata.metadata_store import to_pgvector

logger = setup_logger(__name__)
embedder = Embedder()

# Words that pin a question to one catalogue category (whole-word match, lower case).
CATEGORY_TERMS = {
    "cars": {"car", "cars", "suv", "suvs", "sedan", "sedans", "hatchback", "hatchbacks"},
    "motorcycles": {"motorcycle", "motorcycles", "motorbike", "motorbikes", "bike", "bikes"},
    "scooters": {"scooter", "scooters", "scooty"},
    "spare_parts": {"spare", "spares", "part", "parts"},
}


def categories_in(query: str) -> list[str]:
    """Categories named in the question. "brake parts for my car" means spare parts, not cars."""
    words = set(re.findall(r"[a-z]+", query.lower()))
    found = [category for category, terms in CATEGORY_TERMS.items() if words & terms]
    # ponytail: keyword rules; swap for a classifier if questions get past simple category words.
    return ["spare_parts"] if "spare_parts" in found else found


def get_embedding(text: str) -> list[float]:
    """Embed a user query with the Ollama model named in EMBEDDING_MODEL."""
    if not text or not text.strip():
        raise ValueError("A retrieval query must not be empty")
    logger.info("Retrieval embedding started | chars=%d", len(text.strip()))
    return embedder.embed_text(text.strip(), timeout=30)


async def search_similar(
    query_embedding: Sequence[float],
    category: str | Sequence[str] | None = None,
    top_k: int = 5,
    modality: str = "text",
) -> list[asyncpg.Record]:
    """Nearest rows, restricted to ``category`` (one name or several) when given."""
    vector = to_pgvector(query_embedding)
    if vector is None:
        return []
    if modality not in {"text", "image"}:
        raise ValueError("modality must be 'text' or 'image'")

    limit = max(1, min(int(top_k), 50))
    if modality == "text":
        columns = "chunk_id, category, text, metadata"
        table = "brochure_chunks"
        vector_column = "text_embedding"
        category_column = "category"
    else:
        columns = "i.image_id, i.chunk_id, c.category, c.text, i.metadata"
        table = "brochure_images i JOIN brochure_chunks c ON c.chunk_id = i.chunk_id"
        vector_column = "i.image_embedding"
        category_column = "c.category"

    query = f"SELECT {columns} FROM {table}"
    parameters: list[Any] = [vector, limit]
    categories = [category] if isinstance(category, str) else list(category or [])
    if categories:
        query += f" WHERE {category_column} = ANY($3::text[])"
        parameters.append(categories)
    query += f" ORDER BY {vector_column} {VECTOR_OPERATOR} $1::vector LIMIT $2"

    logger.info("Vector search started | modality=%s | category=%s | top_k=%d", modality, category, limit)
    connection = await asyncpg.connect(timeout=10, command_timeout=20, **DB_PARAMS)
    try:
        rows = list(await connection.fetch(query, *parameters))
        logger.info("Vector search completed | results=%d", len(rows))
        return rows
    finally:
        await connection.close()


def generate_answer(query: str, context: str) -> str:
    if not context.strip():
        return "I could not find indexed automotive information matching that question."
    prompt = (
        "You are an automotive parts assistant. Answer only from the supplied context. "
        "When the user requests products, include a concise Markdown table with part name, price, "
        "and vehicle compatibility when those fields are present. Do not say the question is missing "
        "when product context is supplied. If the context is incomplete, state exactly what is missing.\n\n"
        f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
    )
    logger.info("Answer generation started | context_chars=%d", len(context))
    return embedder.generate_text(prompt, timeout=30)


def parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return dict(value) if isinstance(value, dict) else {}


def product_identity(row: asyncpg.Record) -> str:
    metadata = parse_metadata(row["metadata"])
    return str(
        metadata.get("product_url")
        or metadata.get("part_name")
        or metadata.get("model")
        or metadata.get("document_id")
        or str(row["text"] or "").strip().lower()
    )


def unique_by(rows: Sequence[asyncpg.Record], key: Callable[[Any], str], limit: int) -> list[asyncpg.Record]:
    """First row per key, in rank order, at most ``limit`` rows."""
    first: dict[str, asyncpg.Record] = {}
    for row in rows:
        first.setdefault(key(row), row)
    return list(first.values())[:limit]


def wants_table(query: str) -> bool:
    normalized_query = query.lower()
    return any(term in normalized_query for term in ("table", "tabular", "columns"))


def format_product_answer(rows: Sequence[asyncpg.Record], as_table: bool) -> str | None:
    products = []
    for row in rows:
        metadata = parse_metadata(row["metadata"])
        part_name = metadata.get("part_name") or metadata.get("model")
        if not part_name:
            continue
        vehicle_type = metadata.get("vehicle_type") or metadata.get("vehicle_types") or "Not indexed"
        if isinstance(vehicle_type, list):
            vehicle_type = ", ".join(str(item) for item in vehicle_type)
        products.append((str(part_name), str(metadata.get("price") or "Not indexed"), str(vehicle_type)))

    if not products:
        return None

    def cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    if as_table:
        lines = ["| Part | Price | Vehicle type |", "| --- | --- | --- |"]
        lines.extend(f"| {cell(part)} | {cell(price)} | {cell(vehicle)} |" for part, price, vehicle in products)
        return "Here are the unique matching products from the indexed catalog.\n\n" + "\n".join(lines)

    lines = [f"- **{part}**: {price}. Vehicle type: {vehicle}." for part, price, vehicle in products]
    return "\n".join(lines)


def fallback_answer(query: str, rows: Sequence[asyncpg.Record]) -> str:
    product_answer = format_product_answer(rows, as_table=wants_table(query))
    if product_answer:
        return product_answer
    excerpts = [str(row["text"])[:400] for row in rows]
    return "The local answer model timed out. Retrieved source excerpts:\n\n" + "\n\n".join(excerpts)


async def search_in_category(
    query: str, category: str | None, top_k: int, modality: str = "text"
) -> list[asyncpg.Record]:
    """Vector search that never mixes categories: the selected one, else the ones the question names,
    else the category of the single best match. Fewer than ``top_k`` rows beats padding with other vehicles."""
    query_embedding = await asyncio.wait_for(asyncio.to_thread(get_embedding, query), timeout=35)
    categories = [category] if category else categories_in(query)
    if not categories:
        best = await search_similar(query_embedding, top_k=1, modality=modality)
        categories = [best[0]["category"]] if best else []
    logger.info("Category scope | selected=%s | used=%s", category, categories)
    return await search_similar(query_embedding, categories, min(max(int(top_k) * 4, int(top_k)), 50), modality)


async def retrieve_with_sources(
    query: str, category: str | None = None, top_k: int = 5
) -> tuple[str, list[asyncpg.Record]]:
    started_at = time.monotonic()
    logger.info("Retrieval request started | query=%r | category=%s", query, category)
    rows = await search_in_category(query, category, top_k)
    unique_rows = unique_by(rows, product_identity, top_k)

    context = "\n\n".join(str(row["text"]) for row in unique_rows)
    answer = format_product_answer(unique_rows, as_table=wants_table(query))
    if answer is None:
        try:
            answer = await asyncio.wait_for(asyncio.to_thread(generate_answer, query, context), timeout=35)
        except (RequestException, TimeoutError, asyncio.TimeoutError):
            logger.warning("Answer generation timed out; returning retrieved sources")
            answer = fallback_answer(query, unique_rows)
    logger.info("Retrieval request completed | sources=%d | duration=%.2fs", len(unique_rows), time.monotonic() - started_at)
    return answer, unique_rows


async def search_images_for_query(
    query: str, category: str | None = None, top_k: int = 5
) -> list[asyncpg.Record]:
    rows = await search_in_category(query, category, top_k, modality="image")
    return unique_by(rows, lambda row: str(row["chunk_id"]), top_k)


async def retrieve_pipeline(query: str, category: str | None = None, top_k: int = 5) -> str:
    answer, _rows = await retrieve_with_sources(query, category=category, top_k=top_k)
    return answer

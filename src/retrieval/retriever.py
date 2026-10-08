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

logger = setup_logger(__name__)
embedder = Embedder()

CATEGORY_TERMS = {
    "cars": {"car", "cars", "suv", "suvs", "sedan", "sedans", "hatchback", "hatchbacks"},
    "motorcycles": {"motorcycle", "motorcycles", "motorbike", "motorbikes", "bike", "bikes"},
    "scooters": {"scooter", "scooters", "scooty"},
    "spare_parts": {"spare", "spares", "part", "parts"},
}


def categories_in(query: str) -> list[str]:
    """Categories named in the question."""
    words = set(re.findall(r"[a-z]+", query.lower()))
    found = [cat for cat, terms in CATEGORY_TERMS.items() if words & terms]
    return ["spare_parts"] if "spare_parts" in found else found


def get_embedding(text: str) -> list[float]:
    """Embed a user query with the Ollama model named in EMBEDDING_MODEL."""
    if not text or not text.strip():
        raise ValueError("A retrieval query must not be empty")
    logger.info("Retrieval embedding started | chars=%d", len(text.strip()))
    return embedder.embed_text(text.strip(), timeout=30)


async def search_similar(
    query_embedding: Sequence[float],
    category: str | None = None,
    top_k: int = 5,
    modality: str = "text",
) -> list[asyncpg.Record]:
    if not query_embedding:
        return []
    if modality not in {"text", "image"}:
        raise ValueError("modality must be 'text' or 'image'")

    vector = "[" + ",".join(str(value) for value in query_embedding) + "]"
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
    if category:
        query += f" WHERE {category_column} = $3"
        parameters.append(category)
    query += f" ORDER BY {vector_column} {VECTOR_OPERATOR} $1::vector LIMIT $2"

    logger.info("Vector search started | modality=%s | category=%s | top_k=%d", modality, category, limit)
    connection = await asyncpg.connect(timeout=10, command_timeout=20, **DB_PARAMS)
    try:
        rows = list(await connection.fetch(query, *parameters))
        logger.info("Vector search completed | results=%d", len(rows))
        return rows
    finally:
        await connection.close()


async def search_keywords(
    query: str, categories: list[str] | None = None, top_k: int = 5
) -> list[asyncpg.Record]:
    """Full-text search via PostgreSQL tsvector + ts_rank (BM25-style keyword ranking)."""
    if not query or not query.strip():
        return []
    query_terms = " & ".join(query.lower().split()[:5])
    limit = max(1, min(int(top_k), 50))
    connection = await asyncpg.connect(timeout=10, command_timeout=20, **DB_PARAMS)
    try:
        category_filter = ""
        params: list[Any] = [query_terms, limit]
        if categories:
            category_filter = " AND category = ANY($3::text[])"
            params.append(categories)
        sql = f"""
            SELECT chunk_id, category, text, metadata,
                   ts_rank(to_tsvector('english', text), to_tsquery('english', $1)) AS rank
            FROM brochure_chunks
            WHERE to_tsvector('english', text) @@ to_tsquery('english', $1) {category_filter}
            ORDER BY rank DESC LIMIT $2
        """
        rows = list(await connection.fetch(sql, *params))
        logger.info("Keyword search completed | results=%d", len(rows))
        return rows
    finally:
        await connection.close()


def rrf_fusion(vector_rows: Sequence[asyncpg.Record], keyword_rows: Sequence[asyncpg.Record]) -> list[asyncpg.Record]:
    """Reciprocal Rank Fusion: combine vector + keyword results by chunk_id."""
    scores: dict[str, tuple[float, asyncpg.Record]] = {}
    for i, row in enumerate(vector_rows, start=1):
        cid = str(row["chunk_id"])
        score = scores.get(cid, (0.0, row))[0] + 1.0 / i
        scores[cid] = (score, row)
    for i, row in enumerate(keyword_rows, start=1):
        cid = str(row["chunk_id"])
        score = scores.get(cid, (0.0, row))[0] + 1.0 / i
        scores[cid] = (score, row)
    fused = sorted(scores.items(), key=lambda x: -x[1][0])
    return [row for _, (_, row) in fused]


def unique_by(rows: Sequence[asyncpg.Record], key: Callable[[Any], str], limit: int) -> list[asyncpg.Record]:
    """First row per key, in rank order, at most ``limit`` rows."""
    first: dict[str, asyncpg.Record] = {}
    for row in rows:
        first.setdefault(key(row), row)
    return list(first.values())[:limit]


def generate_answer(query: str, context: str) -> str:
    if not context.strip():
        return "I could not find indexed automotive information matching that question."
    prompt = (
        "You are an expert automotive knowledge assistant specializing in cars, motorcycles, scooters, and spare parts. "
        "Answer ONLY about the specific product(s) mentioned in the question. Ignore unrelated products in the context.\n"
        "Rules:\n"
        "1. Identify the main product(s) in the question (e.g., 'Altroz', 'Hero bike'). Answer ONLY about those.\n"
        "2. If asked about colors/variants/specifications, extract and list them explicitly in a bullet list or table.\n"
        "3. For product comparisons, show side-by-side specs in a Markdown table: Product | Price | Key Features | Colors.\n"
        "4. If a field is missing (e.g., no colors indexed), state: 'Colors not indexed for this product'.\n"
        "5. Be specific and avoid generic repeated data. Answer directly in 2–4 sentences + table/list.\n\n"
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
    """Vector + keyword hybrid search, category-scoped, fused with RRF."""
    query_embedding = await asyncio.wait_for(asyncio.to_thread(get_embedding, query), timeout=35)
    categories = [category] if category else categories_in(query)
    if not categories:
        best = await search_similar(query_embedding, top_k=1, modality=modality)
        categories = [best[0]["category"]] if best else []
    logger.info("Category scope | selected=%s | used=%s", category, categories)

    candidate_count = min(max(int(top_k) * 4, int(top_k)), 50)
    tasks = [search_similar(query_embedding, categories[0] if len(categories) == 1 else None, candidate_count, modality)]
    if modality == "text":
        tasks.append(search_keywords(query, categories, candidate_count))

    results = await asyncio.gather(*tasks)
    vector_rows = results[0]
    keyword_rows = results[1] if len(results) > 1 else []

    if modality == "text" and keyword_rows:
        rows = rrf_fusion(vector_rows, keyword_rows)
    else:
        rows = vector_rows
    return unique_by(rows, product_identity, top_k)


async def retrieve_with_sources(
    query: str, category: str | None = None, top_k: int = 5
) -> tuple[str, list[asyncpg.Record]]:
    started_at = time.monotonic()
    logger.info("Retrieval request started | query=%r | category=%s", query, category)
    rows = await search_in_category(query, category, top_k * 2)
    unique_rows = unique_by(rows, product_identity, top_k * 2)

    if len(unique_rows) > 1:
        primary_product = product_identity(unique_rows[0])
        same_product = [r for r in unique_rows if product_identity(r) == primary_product]
        if len(same_product) >= top_k:
            unique_rows = same_product[:top_k]
            logger.info("Filtered to single product | product=%s | results=%d", primary_product, len(unique_rows))
        else:
            unique_rows = unique_rows[:top_k]
    else:
        unique_rows = unique_rows[:top_k]

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

from __future__ import annotations

import asyncio
import difflib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
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


async def _latest_metadata_version() -> str | None:
    connection = await asyncpg.connect(timeout=10, command_timeout=10, **DB_PARAMS)
    try:
        return await connection.fetchval("SELECT MAX(metadata_version) FROM brochure_chunks")
    finally:
        await connection.close()


async def search_similar(
    query_embedding: Sequence[float],
    category: str | None = None,
    top_k: int = 5,
    metadata_version: str | None = None,
) -> list[asyncpg.Record]:
    if not query_embedding:
        return []

    vector = "[" + ",".join(str(value) for value in query_embedding) + "]"
    limit = max(1, min(int(top_k), 50))

    conditions: list[str] = []
    parameters: list[Any] = [vector, limit]
    idx = 3
    if category:
        conditions.append(f"category = ${idx}")
        parameters.append(category)
        idx += 1
    if metadata_version:
        conditions.append(f"metadata_version = ${idx}")
        parameters.append(metadata_version)
        idx += 1
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    query = f"SELECT chunk_id, category, text, metadata FROM brochure_chunks{where} ORDER BY text_embedding {VECTOR_OPERATOR} $1::vector LIMIT $2"

    logger.info("Vector search started | category=%s | version=%s | top_k=%d", category, metadata_version, limit)
    connection = await asyncpg.connect(timeout=10, command_timeout=20, **DB_PARAMS)
    try:
        rows = list(await connection.fetch(query, *parameters))
        logger.info("Vector search completed | results=%d", len(rows))
        return rows
    finally:
        await connection.close()


async def search_keywords(
    query: str, categories: list[str] | None = None, top_k: int = 5,
    metadata_version: str | None = None,
) -> list[asyncpg.Record]:
    """Full-text search via PostgreSQL tsvector + ts_rank (BM25-style keyword ranking)."""
    if not query or not query.strip():
        return []
    query_terms = " | ".join(re.findall(r"[a-z0-9]+", query.lower())[:8])
    if not query_terms:
        return []
    limit = max(1, min(int(top_k), 50))
    connection = await asyncpg.connect(timeout=10, command_timeout=20, **DB_PARAMS)
    try:
        extra = ""
        params: list[Any] = [query_terms, limit]
        idx = 3
        if categories:
            extra += f" AND category = ANY(${idx}::text[])"
            params.append(categories)
            idx += 1
        if metadata_version:
            extra += f" AND metadata_version = ${idx}"
            params.append(metadata_version)
            idx += 1
        sql = f"""
            SELECT chunk_id, category, text, metadata,
                   ts_rank(to_tsvector('english', text), to_tsquery('english', $1)) AS rank
            FROM brochure_chunks
            WHERE to_tsvector('english', text) @@ to_tsquery('english', $1){extra}
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


SUPPORTED_LANGUAGES = {"English": "", "German": "German", "Hindi": "Hindi", "Telugu": "Telugu"}


def generate_answer(query: str, context: str, language: str = "English") -> str:
    if not context.strip():
        return "I could not find indexed automotive information matching that question."
    lang_line = f"You MUST respond entirely in {SUPPORTED_LANGUAGES[language]}. " if language != "English" else ""
    prompt = (
        f"{lang_line}"
        "You are an expert automotive knowledge assistant specializing in cars, motorcycles, scooters, and spare parts. "
        "Answer only from the supplied context; never use outside knowledge. "
        "Answer ONLY about the specific product(s) mentioned in the question. Ignore unrelated products in the context.\n"
        "Rules:\n"
        "1. Identify the main product(s) in the question (e.g., 'Altroz', 'Hero bike'). Answer ONLY about those.\n"
        "2. If asked about colors/variants/specifications, extract and list them explicitly in a bullet list or table.\n"
        "3. For product comparisons, show side-by-side specs in a Markdown table: Product | Price | Key Features | Colors.\n"
        "4. If a field is missing (e.g., no colors indexed), state: 'Colors not indexed for this product'.\n"
        "5. Be specific and avoid generic repeated data. Answer directly in 2–4 sentences + table/list.\n\n"
        f"Question: {query}\n\nContext:\n{context}\n\n"
        f"{'Respond entirely in ' + SUPPORTED_LANGUAGES[language] + '.\\n' if language != 'English' else ''}"
        "Answer:"
    )
    logger.info("Answer generation started | language=%s | model=%s | context_chars=%d", language, embedder.gen_model, len(context))
    return embedder.generate_text(prompt, timeout=60, max_tokens=300)


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


GENERIC_NAME_WORDS = {"new", "the", "all", "tata", "hero", "suv", "kit", "ev", "ice", "en", "in"}
PRICE_PATTERN = re.compile(r"price|cost|mrp|how much|rupee|lakh|emi|₹|rs\.?\s*\d", re.IGNORECASE)


def wants_price(query: str) -> bool:
    return bool(PRICE_PATTERN.search(query))


def wants_comparison(query: str) -> bool:
    return bool(re.search(r"\b(compare|comparison|versus|vs|difference)\b", query, re.IGNORECASE))


COMPARE_SPLIT = re.compile(r"\s+(?:and|vs\.?|versus|with|to)\s+|,\s*", re.IGNORECASE)


def unindexed_products(query: str, rows: Sequence[asyncpg.Record]) -> list[str]:
    """Products named in a comparison question that no retrieved row mentions."""
    if not wants_comparison(query):
        return []
    body = re.sub(r"\b(compare|comparison|between|of|the|difference)\b", " ", query, flags=re.IGNORECASE)
    context_words = set()
    for row in rows:
        name = product_name(parse_metadata(row["metadata"]))
        context_words |= set(re.findall(r"[a-z0-9]+", name.lower())) - GENERIC_NAME_WORDS
    missing = []
    for part in COMPARE_SPLIT.split(body):
        words = set(re.findall(r"[a-z0-9]+", part.lower())) - GENERIC_NAME_WORDS
        if words and not words & context_words:
            missing.append(part.strip().strip("?.!"))
    return missing


def product_name(metadata: dict[str, Any]) -> str:
    name = metadata.get("part_name") or metadata.get("model")
    if name:
        return str(name)
    slug = str(metadata.get("product_url") or metadata.get("source_url") or "").rstrip("/").rsplit("/", 1)[-1]
    return slug.removesuffix(".html").replace("-", " ").title()


def named_products(query: str, rows: Sequence[asyncpg.Record]) -> list[asyncpg.Record]:
    """Rows whose product name the query names in full ('Glamour X' -> glamour-x, not glamour-x-abs).
    Falls back to the best partial overlap when no name is fully mentioned."""
    names = [(row, set(re.findall(r"[a-z0-9]+", product_name(parse_metadata(row["metadata"])).lower())) - GENERIC_NAME_WORDS)
             for row in rows]
    vocabulary = set().union(*(words for _, words in names))
    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    for word in list(query_words):
        if len(word) >= 5:
            query_words |= set(difflib.get_close_matches(word, vocabulary, n=3, cutoff=0.85))
    full: list[tuple[int, asyncpg.Record]] = []
    partial: list[tuple[int, asyncpg.Record]] = []
    for row, name_words in names:
        overlap = len(name_words & query_words)
        if not overlap:
            continue
        if overlap == len(name_words):
            full.append((len(name_words), row))
        else:
            partial.append((overlap, row))
    if full:
        longest = max(size for size, _ in full)
        return [row for size, row in full if size == longest]
    if partial:
        best = max(score for score, _ in partial)
        return [row for score, row in partial if score == best]
    return []


COLOUR_PATTERN = re.compile(r"colou?rs?|variants?", re.IGNORECASE)
SPEC_PATTERN = re.compile(r"engine|mileage|seating|boot|airbag|safety|features?|dimensions?|specifications?", re.IGNORECASE)


@dataclass(frozen=True)
class QueryIntent:
    intents: frozenset[str]
    named_rows: tuple[asyncpg.Record, ...]


def understand_query(query: str, rows: Sequence[asyncpg.Record]) -> QueryIntent:
    checks = {
        "comparison": wants_comparison(query),
        "price": wants_price(query),
        "table": wants_table(query),
        "colour": bool(COLOUR_PATTERN.search(query)),
        "spec": bool(SPEC_PATTERN.search(query)),
    }
    intents = frozenset(name for name, hit in checks.items() if hit) or frozenset({"general"})
    return QueryIntent(intents, tuple(named_products(query, rows)))


STOP_WORDS = {
    "which", "what", "show", "tell", "have", "has", "with", "does", "list", "about", "the", "and", "for", "are",
    "any", "give", "can", "you", "that", "this", "there", "who", "how", "many", "much", "is", "of", "in", "on", "a",
    "an", "to", "me", "all", "from", "its", "describe", "feature", "features", "details", "detail", "specification",
    "specifications", "information", "info", "model", "models", "vehicle", "vehicles", "product", "products",
}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
PRICE_AMOUNT = re.compile(r"₹\s?[\d,.]+(?:\s*Lakh)?", re.IGNORECASE)


def compact_price(value: Any) -> str:
    text = str(value or "")
    match = PRICE_AMOUNT.search(text)
    return match.group(0) if match else (text[:40] or "Not indexed")


def shown_price(rows: Sequence[asyncpg.Record], row: asyncpg.Record) -> str:
    price = compact_price(parse_metadata(row["metadata"]).get("price"))
    counts = Counter(compact_price(parse_metadata(r["metadata"]).get("price")) for r in rows)
    if counts[price] > 1 and price != "Not indexed":
        return "Not reliable (same price on several pages)"
    return price


def feature_terms(query: str) -> set[str]:
    category_words = set().union(*CATEGORY_TERMS.values())
    words = set(re.findall(r"[a-z]+", query.lower()))
    return {word for word in words - STOP_WORDS - category_words if len(word) >= 3}


def relevant_snippets(text: str, terms: set[str], limit: int) -> list[str]:
    snippets = []
    for sentence in SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        lowered = sentence.lower()
        positions = [lowered.find(term) for term in terms if term in lowered]
        if positions:
            start = sentence.rfind(" ", 0, max(0, min(positions) - 80)) + 1
            snippets.append(sentence[start:start + 200])
        if len(snippets) == limit:
            break
    return snippets


def feature_matches(query: str, rows: Sequence[asyncpg.Record]) -> list[tuple[asyncpg.Record, str]]:
    terms = feature_terms(query)
    if not terms:
        return []
    matches = []
    for row in rows:
        snippets = relevant_snippets(str(row["text"] or ""), terms, 1)
        if snippets:
            matches.append((row, snippets[0]))
    return matches


def feature_list(matches: Sequence[tuple[asyncpg.Record, str]]) -> str:
    lines = []
    for row, snippet in matches:
        metadata = parse_metadata(row["metadata"])
        name = product_name(metadata) or "Not indexed"
        link = metadata.get("product_url") or metadata.get("source_url")
        product = f"[{name}]({link})" if link else name
        lines.append(f"- **{product}**: {snippet}")
    return "Products whose indexed details mention it:\n\n" + "\n".join(lines)


def product_context(query: str, rows: Sequence[asyncpg.Record]) -> str:
    terms = feature_terms(query)
    blocks = []
    for row in rows:
        metadata = parse_metadata(row["metadata"])
        name = product_name(metadata) or "Unknown product"
        lines = [f"Product: {name}", f"Price: {compact_price(metadata.get('price'))}"]
        colours = ", ".join(str(c) for c in metadata.get("color_variants") or [])
        if colours:
            lines.append(f"Colours: {colours}")
        features = [str(f)[:80] for f in (metadata.get("features") or [])[:5]]
        if features:
            lines.append("Features: " + "; ".join(features))
        lines.extend(relevant_snippets(str(row["text"] or ""), terms, 2))
        blocks.append("\n".join(lines)[:800])
    return "\n\n".join(blocks)


def table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def feature_table(rows: Sequence[asyncpg.Record]) -> str:
    lines = ["| Product | Price | Colours | Key features |", "| --- | --- | --- | --- |"]
    for row in rows:
        metadata = parse_metadata(row["metadata"])
        name = product_name(metadata) or "Not indexed"
        link = metadata.get("product_url") or metadata.get("source_url")
        product = f"[{name}]({link})" if link else name
        price = shown_price(rows, row)
        colours = ", ".join(str(c) for c in metadata.get("color_variants") or []) or "Not indexed"
        features = "; ".join(str(f)[:80] for f in (metadata.get("features") or [])[:4]) or "Not indexed"
        lines.append(f"| {table_cell(product)} | {table_cell(price)} | {table_cell(colours)} | {table_cell(features)} |")
    return "Here is the side-by-side comparison from the indexed catalog.\n\n" + "\n".join(lines)


def format_product_answer(query: str, rows: Sequence[asyncpg.Record], as_table: bool) -> str | None:
    show_price = wants_price(query)
    products = []
    for row in rows:
        metadata = parse_metadata(row["metadata"])
        part_name = product_name(metadata)
        if not part_name:
            continue
        vehicle_type = metadata.get("vehicle_type") or metadata.get("vehicle_types") or "Not indexed"
        if isinstance(vehicle_type, list):
            vehicle_type = ", ".join(str(item) for item in vehicle_type)
        products.append((
            str(part_name),
            shown_price(rows, row),
            str(vehicle_type),
            str(metadata.get("description") or "Not indexed"),
        ))

    if not products:
        return None

    cell = table_cell

    if as_table and show_price:
        lines = ["| Part | Price | Vehicle type |", "| --- | --- | --- |"]
        lines.extend(f"| {cell(part)} | {cell(price)} | {cell(vehicle)} |" for part, price, vehicle, _ in products)
        return "Here are the unique matching products from the indexed catalog.\n\n" + "\n".join(lines)
    if as_table:
        lines = ["| Part | Description |", "| --- | --- |"]
        lines.extend(f"| {cell(part)} | {cell(description)} |" for part, _, _, description in products)
        return "Here are the unique matching products from the indexed catalog.\n\n" + "\n".join(lines)
    if show_price:
        return "\n".join(f"- **{part}**: {price}. Vehicle type: {vehicle}." for part, price, vehicle, _ in products)
    return "\n".join(f"- **{part}**: {description}" for part, _, _, description in products)


def fallback_answer(query: str, rows: Sequence[asyncpg.Record]) -> str:
    product_answer = format_product_answer(query, rows, as_table=wants_table(query))
    if product_answer:
        return product_answer
    excerpts = [str(row["text"])[:400] for row in rows]
    return "The local answer model timed out. Retrieved source excerpts:\n\n" + "\n\n".join(excerpts)


def _has_useful_metadata(row: asyncpg.Record) -> bool:
    meta = parse_metadata(row.get("metadata", {}))
    return bool(meta.get("model") or meta.get("part_name") or meta.get("brand"))


async def _search_with_version(
    query_embedding: Sequence[float], query: str, categories: list[str],
    candidate_count: int, metadata_version: str | None,
) -> list[asyncpg.Record]:
    cat = categories[0] if len(categories) == 1 else None
    vector_rows, keyword_rows = await asyncio.gather(
        search_similar(query_embedding, cat, candidate_count, metadata_version=metadata_version),
        search_keywords(query, categories, candidate_count, metadata_version),
    )
    return rrf_fusion(vector_rows, keyword_rows) if keyword_rows else list(vector_rows)


async def search_in_category(query: str, category: str | None, top_k: int) -> list[asyncpg.Record]:
    """Vector + keyword hybrid search, category-scoped, fused with RRF.
    Prefers latest metadata_version; falls back to any version if no results."""
    query_embedding = await asyncio.wait_for(asyncio.to_thread(get_embedding, query), timeout=35)
    categories = [category] if category else categories_in(query)
    if not categories:
        best = await search_similar(query_embedding, top_k=1)
        categories = [best[0]["category"]] if best else []
    logger.info("Category scope | selected=%s | used=%s", category, categories)

    candidate_count = min(max(int(top_k) * 4, int(top_k)), 50)
    latest_version = await _latest_metadata_version()
    rows = await _search_with_version(query_embedding, query, categories, candidate_count, latest_version)
    useful = [r for r in rows if _has_useful_metadata(r)]
    if not useful and latest_version:
        logger.info("No results for version=%s, falling back to any version", latest_version)
        rows = await _search_with_version(query_embedding, query, categories, candidate_count, None)
        useful = [r for r in rows if _has_useful_metadata(r)]
    # ponytail: if even fallback has no useful metadata, return raw rows rather than nothing
    return unique_by(useful or rows, product_identity, top_k)


async def retrieve_with_sources(
    query: str, category: str | None = None, top_k: int = 5, language: str = "English"
) -> tuple[str, list[asyncpg.Record]]:
    started_at = time.monotonic()
    logger.info("Retrieval request started | query=%r | category=%s", query, category)
    rows = await search_in_category(query, category, top_k * 2)
    unique_rows = unique_by(rows, product_identity, top_k * 2)
    # ponytail: named products are matched within the top-2k pool, not the whole catalog; widen the pool if a named product is missed
    understanding = understand_query(query, unique_rows)
    logger.info("Query understanding | intents=%s | named_products=%d", sorted(understanding.intents), len(understanding.named_rows))
    unique_rows = (list(understanding.named_rows) or unique_rows)[:top_k]

    context = product_context(query, unique_rows)
    missing = unindexed_products(query, unique_rows)
    matches = feature_matches(query, unique_rows) if not understanding.named_rows else []
    if language == "English" and missing:
        note = "\n".join(f"**{name}**: not in the indexed catalog." for name in missing)
        answer = note + "\n\n" + (feature_table(unique_rows) if unique_rows else "")
    elif language == "English" and unique_rows and "comparison" in understanding.intents and understanding.intents & {"spec", "colour"}:
        answer = feature_table(unique_rows)
    elif language == "English" and understanding.intents & {"table", "price"} and "comparison" not in understanding.intents:
        answer = format_product_answer(query, unique_rows, as_table=wants_table(query))
    elif language == "English" and matches and "comparison" not in understanding.intents:
        answer = feature_list(matches)
    else:
        answer = None
    if answer is None:
        try:
            answer = await asyncio.wait_for(asyncio.to_thread(generate_answer, query, context, language), timeout=65)
        except (RequestException, TimeoutError, asyncio.TimeoutError):
            logger.warning("Answer generation failed | language=%s", language)
            answer = fallback_answer(query, unique_rows)
            if language != "English":
                answer = f"⚠ Generation in {language} failed (model timeout). Showing English fallback.\n\n{answer}"
    logger.info("Retrieval request completed | sources=%d | duration=%.2fs", len(unique_rows), time.monotonic() - started_at)
    return answer, unique_rows


async def retrieve_pipeline(query: str, category: str | None = None, top_k: int = 5, language: str = "English") -> str:
    answer, _rows = await retrieve_with_sources(query, category=category, top_k=top_k, language=language)
    return answer

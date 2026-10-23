from __future__ import annotations

import asyncio
import difflib
import json
import re
import textwrap
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Sequence
import asyncpg
from requests import RequestException
from config.logger import setup_logger
from config.settings import DB_PARAMS, MULTILINGUAL_MODEL, VECTOR_OPERATOR
from ingestion.embedding.embedder import Embedder
from ingestion.metadata.metadata_store import parse_version

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
    vocabulary = set().union(*CATEGORY_TERMS.values())
    for word in words - vocabulary:
        if len(word) >= 5:
            # 0.85: "motocycles"(0.95) and "scootors"(0.88) match; "share" vs "spare"(0.80) must not
            words |= set(difflib.get_close_matches(word, vocabulary, n=1, cutoff=0.85))
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
        versions = [r["metadata_version"] for r in await connection.fetch("SELECT DISTINCT metadata_version FROM brochure_chunks")]
        return max(versions, key=parse_version, default=None)
    finally:
        await connection.close()


async def search_similar(
    query_embedding: Sequence[float],
    category: str | Sequence[str] | None = None,
    top_k: int = 5,
    metadata_version: str | None = None,
) -> list[asyncpg.Record]:
    if not query_embedding:
        return []

    vector = "[" + ",".join(str(value) for value in query_embedding) + "]"
    limit = max(1, min(int(top_k), 50))
    categories = [category] if isinstance(category, str) else list(category or [])

    conditions: list[str] = []
    parameters: list[Any] = [vector, limit]
    idx = 3
    if categories:
        conditions.append(f"category = ANY(${idx}::text[])")
        parameters.append(categories)
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


SUPPORTED_LANGUAGES = ("English", "German", "Hindi", "Telugu")


TRANSLATE_TIMEOUT = 240


LINK_PATTERN = re.compile(r"\[[^\]]+\]\([^)]+\)")


def _ollama_translate(prompt: str, max_tokens: int) -> str:
    result = embedder.generate_text(prompt, timeout=TRANSLATE_TIMEOUT, model=MULTILINGUAL_MODEL or None, max_tokens=max_tokens)
    return re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()


def needs_translation(query: str) -> bool:
    """True when a question is not English: non-ASCII script, or a sentence with almost no English function words."""
    if not query.isascii():
        return True
    words = re.findall(r"[a-z]+", query.lower())
    return len(words) >= 3 and sum(word in STOP_WORDS for word in words) < 2


def translate_query(query: str) -> str:
    prompt = (
        "Translate this question into English. Reply with only the English question on one line. "
        "Keep product names unchanged. Do not answer the question.\n\n"
        f"Question: {query}\nEnglish:"
    )
    logger.info("Query translation started | model=%s", MULTILINGUAL_MODEL)
    lines = _ollama_translate(prompt, 60).splitlines()
    return lines[0].strip() if lines and lines[0].strip() else query


def translate_answer(text: str, target: str) -> str:
    """Translate a finished English answer. Facts are decided in English first; product links are protected."""
    links = LINK_PATTERN.findall(text)
    for index, link in enumerate(links):
        text = text.replace(link, f"@@{index}@@", 1)
    prompt = (
        f"Translate the text below into {target}. Output only the translation, nothing else. "
        "Keep the Markdown structure (tables, bullets, bold). "
        "Keep numbers, prices and every @@N@@ marker exactly as written.\n\n"
        f"{text}\n\nTranslation:"
    )
    logger.info("Answer translation started | target=%s | model=%s | chars=%d", target, MULTILINGUAL_MODEL, len(text))
    translated = _ollama_translate(prompt, 700)
    for index, link in enumerate(links):
        if f"@@{index}@@" not in translated:
            raise ValueError("translation dropped a product link")
        translated = translated.replace(f"@@{index}@@", link)
    return translated


PRICE_PHRASE = re.compile(
    r"(?:\s*(?:starting|priced)\s+(?:from|at))?\s*₹\s?[\d,.]+\s*(?:lakh|l)?\b\s*\*?\s*(?:\(ex-showroom\))?", re.IGNORECASE
)


CALL_TO_ACTION = re.compile(r"\bbook\b|explore features|\btoday\b", re.IGNORECASE)


def clean_description(metadata: dict[str, Any]) -> str:
    """Page description with price phrases removed, so overview text never quotes a price."""
    return re.sub(r"\s{2,}", " ", PRICE_PHRASE.sub(" ", str(metadata.get("description") or ""))).strip()


def product_paragraph(row: asyncpg.Record, price_rows: Sequence[asyncpg.Record]) -> str:
    """A paragraph from this product's own stored fields only; "" when the page stored nothing usable (Hero rows)."""
    metadata = parse_metadata(row["metadata"])
    name = product_name(metadata)
    description = clean_description(metadata)
    sentences = [s.strip().rstrip(".!") + "." for s in SENTENCE_SPLIT.split(description)
                 if len(s.split()) >= 4 and not CALL_TO_ACTION.search(s)][:2]
    features: list[str] = []
    for raw in metadata.get("features") or []:
        feature = textwrap.shorten(str(raw).strip().rstrip("."), 120, placeholder="…")
        lowered = feature.lower().rstrip("…")
        if len(feature.split()) >= 3 and not any(lowered in kept.lower() or kept.lower().rstrip("…") in lowered for kept in features):
            features.append(feature)
    features = features[:4]
    if features:
        listed = "; ".join(features)
        sentences.append(f"Key features include {listed}{'' if listed.endswith('…') else '.'}")
    colours = [str(c) for c in metadata.get("color_variants") or []]
    if colours:
        sentences.append(f"It is listed in {', '.join(colours)}.")
    price = shown_price(price_rows, row)
    if "₹" in price:
        sentences.append(f"The indexed starting price is {price}.")
    if not sentences:
        return ""
    link = metadata.get("product_url") or metadata.get("source_url")
    if link:
        sentences.append(f"Full details: [{name}]({link}).")
    return f"**{name}**: " + " ".join(sentences)


def catalog_overview(rows: Sequence[asyncpg.Record]) -> str:
    """Overview for a general question, built only from indexed page descriptions: no model call, nothing invented, no prices.
    One intro sentence per category + one sentence per product + one closing line (~7 sentences for 5 products)."""
    by_category: dict[str, list[str]] = {}
    lines = []
    for row in rows:
        metadata = parse_metadata(row["metadata"])
        name = product_name(metadata)
        by_category.setdefault(str(row["category"]), []).append(name)
        description = clean_description(metadata)
        first =SENTENCE_SPLIT.split(description)[0].strip().rstrip(".!") if description else ""
        if first:
            lines.append(f"**{name}**: {first}.")
    intro = " ".join(
        f"The indexed catalog has {len(names)} {category.replace('_', ' ')}: {', '.join(names)}." for category, names in by_category.items()
    )
    outro = "Ask about a specific model or category to see its price, colours, features and images."
    return "\n\n".join([intro, *lines, outro])


def generate_answer(query: str, context: str) -> str:
    if not context.strip():
        return "I could not find indexed automotive information matching that question."
    prompt = (
        "You are an expert automotive knowledge assistant specializing in cars, motorcycles, scooters, and spare parts. "
        "Answer only from the supplied context; never use outside knowledge. "
        "Answer ONLY about the specific product(s) mentioned in the question. Ignore unrelated products in the context.\n"
        "Rules:\n"
        "1. Identify the main product(s) in the question (e.g., 'Altroz', 'Hero bike'). Answer ONLY about those.\n"
        "2. If asked about colors/variants/specifications, extract and list them explicitly in a bullet list or table.\n"
        "3. For product comparisons, show side-by-side specs in a Markdown table: Product | Price | Key Features | Colors.\n"
        "4. If a field is missing (e.g., no colors indexed), state: 'Colors not indexed for this product'.\n"
        "5. Be specific and avoid generic repeated data. Answer in one or two full paragraphs (5–10 sentences) "
        "using only facts from the context, plus a table/list when it helps.\n\n"
        f"Question: {query}\n\nContext:\n{context}\n\n"
        "Answer:"
    )
    logger.info("Answer generation started | model=%s | context_chars=%d", embedder.gen_model, len(context))
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
            label = " ".join(w for w in part.split() if w.lower().strip("?.!,") not in STOP_WORDS)
            missing.append(label.strip("?.!, ") or part.strip())
    return missing


def product_name(metadata: dict[str, Any]) -> str:
    name = metadata.get("part_name") or metadata.get("model")
    if name:
        return str(name)
    slug = str(metadata.get("product_url") or metadata.get("source_url") or "").rstrip("/").rsplit("/", 1)[-1]
    return slug.removesuffix(".html").replace("-", " ").title()


def own_image(metadata: dict[str, Any]) -> str | None:
    """First stored image whose filename matches this product (model slug or part number), so site-wide
    pictures scraped onto every page (xoom-160 on every Hero row) are never shown as this product."""
    name = re.sub(r"[^a-z0-9]", "", product_name(metadata).lower())
    for image in metadata.get("images") or []:
        path = str(image.get("local_path") or "")
        stem = PureWindowsPath(path).stem.lower()  # splits on both / and \
        tokens = {t for t in re.findall(r"[a-z0-9]+", stem) if len(t) >= 3} - GENERIC_NAME_WORDS
        if path and any(t in name for t in tokens) and Path(path).exists():
            return path
    return None


def name_words(row: asyncpg.Record) -> set[str]:
    """Distinctive words of a product name; empty for landing pages such as the Hero home page ("En In")."""
    words = set(re.findall(r"[a-z0-9]+", product_name(parse_metadata(row["metadata"])).lower()))
    return words - GENERIC_NAME_WORDS - STOP_WORDS - {"for", "or"}


def named_products(query: str, rows: Sequence[asyncpg.Record]) -> list[asyncpg.Record]:
    """Rows whose product name the query names in full ('Glamour X' -> glamour-x, not glamour-x-abs).
    Falls back to the best partial overlap when no name is fully mentioned."""
    names = [(row, name_words(row)) for row in rows]
    vocabulary = set().union(*(words for _, words in names))
    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    # only unknown words are typo candidates: "price" must never fuzz into "prime"
    for word in query_words - AUTOMOTIVE_TERMS - STOP_WORDS:
        if len(word) >= 5:
            query_words |= set(difflib.get_close_matches(word, vocabulary, n=3, cutoff=0.85))
    full: list[tuple[set[str], asyncpg.Record]] = []
    partial: list[tuple[int, asyncpg.Record]] = []
    for row, words in names:
        overlap = len(words & query_words)
        if not overlap:
            continue
        if overlap == len(words):
            full.append((words, row))
        else:
            partial.append((overlap, row))
    if full:
        # drop a name only when a longer matched name contains it ("Glamour X" inside "Glamour X ABS")
        return [row for words, row in full if not any(words < other for other, _ in full)]
    if partial:
        best = max(score for score, _ in partial)
        return [row for score, row in partial if score == best]
    return []


AUTOMOTIVE_TERMS = set().union(*CATEGORY_TERMS.values()) | {
    "automobile", "automotive", "tata", "hero", "price", "prices", "cost", "costs", "lakh", "emi", "showroom",
    "mileage", "engine", "engines", "cc", "bhp", "torque", "fuel", "petrol", "diesel", "electric", "ev", "cng",
    "hybrid", "battery", "charging", "colour", "colours", "color", "colors", "variant", "variants", "sunroof",
    "airbag", "airbags", "safety", "ncap", "abs", "brake", "brakes", "tyre", "tyres", "tire", "tires", "wheel",
    "wheels", "alloy", "alloys", "seat", "seats", "seating", "boot", "transmission", "gear", "gears", "clutch",
    "interior", "exterior", "infotainment", "adas", "cabin", "headlamp", "headlamps", "feature", "features",
    "specification", "specifications", "spec", "specs", "accessory", "accessories", "cable", "piston", "chain",
    "filter", "kicklever", "helmet", "dealer", "warranty", "drive", "driving", "ride", "riding",
}
# Fixed text, not model output: an off-topic question must never reach the LLM, in any language.
OUT_OF_SCOPE = {
    "English": "I can only answer questions about the cars, motorcycles, scooters and spare parts in the indexed catalog.",
    "German": "Ich kann nur Fragen zu den Autos, Motorrädern, Rollern und Ersatzteilen im indexierten Katalog beantworten.",
    "Hindi": "मैं केवल इंडेक्स किए गए कैटलॉग की कारों, मोटरसाइकिलों, स्कूटरों और स्पेयर पार्ट्स के बारे में प्रश्नों का उत्तर दे सकता हूँ।",
    "Telugu": "నేను ఇండెక్స్ చేసిన కేటలాగ్‌లోని కార్లు, మోటార్‌సైకిళ్లు, స్కూటర్లు మరియు స్పేర్ పార్ట్స్ గురించిన ప్రశ్నలకు మాత్రమే సమాధానం ఇవ్వగలను.",
}


def is_automotive(query: str, named_rows: Sequence[asyncpg.Record]) -> bool:
    """In scope when the (English) question names an indexed product or uses an automotive term."""
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    return bool(named_rows) or bool(words & AUTOMOTIVE_TERMS) or bool(categories_in(query))


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


OVERVIEW_WORDS = {"explain", "overview", "summary", "summarize", "automotive", "automobile", "company", "brand", "brands", "general", "catalog", "catalogue"}


def feature_terms(query: str) -> set[str]:
    """Words that name a feature to look for. Brand and overview words are excluded: every Tata page says "Brand: tata"."""
    category_words = set().union(*CATEGORY_TERMS.values())
    words = set(re.findall(r"[a-z]+", query.lower()))
    excluded = STOP_WORDS | category_words | GENERIC_NAME_WORDS | OVERVIEW_WORDS
    return {word for word in words - excluded if len(word) >= 3}


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


def product_context(query: str, rows: Sequence[asyncpg.Record], price_rows: Sequence[asyncpg.Record]) -> str:
    terms = feature_terms(query)
    blocks = []
    for row in rows:
        metadata = parse_metadata(row["metadata"])
        name = product_name(metadata) or "Unknown product"
        lines = [f"Product: {name}", f"Price: {shown_price(price_rows, row)}"]
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


def feature_table(rows: Sequence[asyncpg.Record], price_rows: Sequence[asyncpg.Record]) -> str:
    lines = ["| Product | Price | Colours | Key features |", "| --- | --- | --- | --- |"]
    for row in rows:
        metadata = parse_metadata(row["metadata"])
        name = product_name(metadata) or "Not indexed"
        link = metadata.get("product_url") or metadata.get("source_url")
        product = f"[{name}]({link})" if link else name
        price = shown_price(price_rows, row)
        colours = ", ".join(str(c) for c in metadata.get("color_variants") or []) or "Not indexed"
        features = "; ".join(str(f)[:80] for f in (metadata.get("features") or [])[:4]) or "Not indexed"
        lines.append(f"| {table_cell(product)} | {table_cell(price)} | {table_cell(colours)} | {table_cell(features)} |")
    return "Here is the side-by-side comparison from the indexed catalog.\n\n" + "\n".join(lines)


def format_product_answer(
    query: str, rows: Sequence[asyncpg.Record], as_table: bool, price_rows: Sequence[asyncpg.Record]
) -> str | None:
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
            shown_price(price_rows, row),
            str(vehicle_type),
            str(metadata.get("description") or "Not indexed"),
            metadata.get("product_url") or metadata.get("source_url"),
        ))

    if not products:
        return None

    if show_price:
        priced = [p for p in products if "₹" in p[1]]
        unpriced = [p for p in products if "₹" not in p[1]]
        parts = []
        if priced and as_table:
            lines = ["| Part | Price | Vehicle type |", "| --- | --- | --- |"]
            lines.extend(f"| {table_cell(part)} | {table_cell(price)} | {table_cell(vehicle)} |" for part, price, vehicle, _, _ in priced)
            parts.append("Here are the unique matching products from the indexed catalog.\n\n" + "\n".join(lines))
        elif priced:
            parts.append("\n".join(f"- **{part}**: {price}. Vehicle type: {vehicle}." for part, price, vehicle, _, _ in priced))
        if unpriced:
            links = ", ".join(f"[{part}]({url})" if url else part for part, _, _, _, url in unpriced)
            names = ", ".join(part for part, *_ in unpriced)
            parts.append(f"No reliable price is indexed for {names}.\nSee the product page for the current price and details: {links}")
        return "\n\n".join(parts)
    if as_table:
        lines = ["| Part | Description |", "| --- | --- |"]
        lines.extend(f"| {table_cell(part)} | {table_cell(description)} |" for part, _, _, description, _ in products)
        return "Here are the unique matching products from the indexed catalog.\n\n" + "\n".join(lines)
    return "\n".join(f"- **{part}**: {description}" for part, _, _, description, _ in products)


def fallback_answer(query: str, rows: Sequence[asyncpg.Record]) -> str:
    product_answer = format_product_answer(query, rows, as_table=wants_table(query), price_rows=rows)
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
    vector_rows, keyword_rows = await asyncio.gather(
        search_similar(query_embedding, categories, candidate_count, metadata_version=metadata_version),
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
    logger.info("Retrieval request started | query=%r | category=%s | language=%s", query, category, language)
    if needs_translation(query):
        try:
            query = await asyncio.wait_for(asyncio.to_thread(translate_query, query), timeout=TRANSLATE_TIMEOUT + 5)
            logger.info("Query translated | english=%r", query)
        except (RequestException, TimeoutError, asyncio.TimeoutError):
            logger.warning("Query translation failed; searching with the original text")
    rows = await search_in_category(query, category, top_k * 2)
    unique_rows = [row for row in unique_by(rows, product_identity, top_k * 2) if name_words(row)]
    # ponytail: named products are matched within the top-2k pool, not the whole catalog; widen the pool if a named product is missed
    understanding = understand_query(query, unique_rows)
    logger.info("Query understanding | intents=%s | named_products=%d", sorted(understanding.intents), len(understanding.named_rows))
    if not is_automotive(query, understanding.named_rows):
        logger.info("Out of scope | query=%r", query)
        return OUT_OF_SCOPE.get(language, OUT_OF_SCOPE["English"]), []
    specific = understanding.named_rows or categories_in(query) or feature_terms(query) or understanding.intents - {"general"}
    if not specific:
        logger.info("Overview question | query=%r", query)
        return await localize(catalog_overview(unique_rows[:top_k]), language), []
    pool = unique_rows  # prices are judged against the whole pool: one narrowed row cannot reveal a shared price
    unique_rows = (list(understanding.named_rows) or unique_rows)[:top_k]

    context = product_context(query, unique_rows, pool)
    missing = unindexed_products(query, unique_rows)
    matches = feature_matches(query, unique_rows) if not understanding.named_rows else []
    if missing:
        note = "\n".join(f"**{name}**: not in the indexed catalog." for name in missing)
        answer = note + "\n\n" + (feature_table(unique_rows, pool) if unique_rows else "")
    elif unique_rows and "comparison" in understanding.intents and understanding.intents & {"spec", "colour"}:
        answer = feature_table(unique_rows, pool)
    elif understanding.intents & {"table", "price"} and "comparison" not in understanding.intents:
        answer = format_product_answer(query, unique_rows, as_table=wants_table(query), price_rows=pool)
    elif matches and "comparison" not in understanding.intents:
        answer = feature_list(matches)
        unique_rows = [row for row, _ in matches]  # details, sources and cards cover only the listed products
    else:
        answer = None
    if answer is not None:
        details = [p for p in (product_paragraph(row, pool) for row in unique_rows) if p]
        if details:
            answer += "\n\n**Details**\n\n" + "\n\n".join(details)
    if answer is None:
        try:
            answer = await asyncio.wait_for(asyncio.to_thread(generate_answer, query, context), timeout=65)
        except (RequestException, TimeoutError, asyncio.TimeoutError):
            logger.warning("Answer generation failed")
            answer = fallback_answer(query, unique_rows)
    answer = await localize(answer, language)
    logger.info("Retrieval request completed | sources=%d | duration=%.2fs", len(unique_rows), time.monotonic() - started_at)
    return answer, unique_rows


async def localize(answer: str, language: str) -> str:
    if language == "English":
        return answer
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(translate_answer, answer, language), timeout=TRANSLATE_TIMEOUT + 5
        )
    except (RequestException, TimeoutError, asyncio.TimeoutError, ValueError) as exc:
        logger.warning("Answer translation failed | language=%s | %s", language, exc)
        return f"⚠ Translation to {language} failed. Showing the English answer.\n\n{answer}"



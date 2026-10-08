# Claude Code Project Guidance

**Ponytail Mode:** ACTIVE (full level)  
**Last updated:** 2026-10-08

## Working Style

- **Lazy first:** stdlib > native platform > existing deps > install new > code. Read fully before solving.
- **No over-engineering:** no abstract base classes with one impl, no config for unchanging values, no speculative features.
- **Boring beats clever:** the simplest correct solution that works is the right one.
- **One line if possible:** `@lru_cache(maxsize=1000)` instead of a custom cache class.

## Code Conventions

- **No comments** unless the WHY is non-obvious (hidden constraint, subtle invariant, workaround for a bug).
- **Short diffs:** prefer Edit over Write scripts. If a script must rewrite, show the unified diff after.
- **Show diffs:** After any file edit, display what changed or ask for review before moving on. The IDE may not show changes from scripts.

## Repository Shape

- **Offline/Online split:** ingestion (Playwright, vision, embeddings) is decoupled from retrieval (serving, LLM).
- **Four loaders:** CarsLoader (bespoke discovery), GenericVehicleLoader (motorcycles/scooters), SparePartsLoader, all inherit BaseLoader.
- **Shared Embedder:** one Ollama client across loaders, not three hand-rolled ones.
- **Category-scoped retrieval:** `search_in_category()` resolves to a single category before vector/keyword search, enforced in SQL `WHERE category = ANY(...)`.
- **Dual-path search:** BM25 keyword (PostgreSQL tsvector) + vector (pgvector) fused via Reciprocal Rank Fusion (RRF).

## Key Files

- **src/ingestion/loader/base_loader.py** — shared Playwright lifecycle, image download, Ollama preflight, `download_and_extract()` crawl loop
- **src/ingestion/loader/cars_loader.py** — bespoke tata.cars discovery (/{model}/{energy}.html), per-model image filtering
- **src/ingestion/embedding/embedder.py** — thin Ollama client (embed_text, generate_text, describe_image, embed_image); vision_max_tokens=120 to cap caption length
- **src/retrieval/retriever.py** — `search_in_category()` orchestrates vector + keyword search in parallel; `rrf_fusion()` combines results; `unique_by()` dedupes by product identity
- **src/config/settings.py** — single source for DB_PARAMS, VECTOR_OPERATOR, EMBEDDING_DIM, VISION_MAX_TOKENS
- **README.md** — production-grade: architecture diagrams, feature status table, example prompts, design trade-offs

## Decisions / Trade-offs

- **Text + image embeddings in same space:** both use `nomic-embed-text`, no separate CLIP visual encoder — simpler, adequate for product retrieval.
- **Deterministic document_id:** sha1(category | product_url or brand+model) means re-running a loader updates its existing row, never duplicates.
- **One chunk per product:** no sub-document chunking; each `brochure_chunks` row is one product with full text/metadata.
- **Vision optional per category:** `{CATEGORY}_ENABLE_VISION=false` for CPU-only Ollama; text-only is much faster.
- **HNSW index over IVFFlat:** HNSW trades slightly more memory for faster/more accurate recalls; `MetadataStore._ensure_indexes()` builds it on startup, drops legacy IVFFlat.

## When Ponytail Doesn't Apply

Never lazy about:
- Input validation at trust boundaries (user queries, external API responses)
- Error handling that prevents data loss
- Security measures (SQL injection, prompt injection, credential leakage)
- Accessibility basics
- Explicitly requested features (user says "build the full version" → build it)

## Testing

- **Smoke tests, not suites:** one small `test_*.py` per module, or a `demo()`/`__main__` self-check for non-trivial logic.
- **Integration > unit mocks:** real database for metadata_store tests, real Ollama for embedder tests.
- **No pytest fixtures** unless the test suite grows beyond ~10 cases; keep it simple.

---

**If this guidance becomes stale,** update it — decisions rot if they're not defended by a reason that still holds.

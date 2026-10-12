# Skills & Patterns

## RAG Pipeline

- **Ingestion:** Playwright scraping → normalization (`NormalizedDocument`) → Ollama embedding/vision → PostgreSQL + pgvector upsert
- **Retrieval:** query embedding → category-scoped vector search (HNSW) + BM25 keyword search (tsvector) → Reciprocal Rank Fusion → answer generation
- **One chunk per product:** no sub-document chunking; deterministic `document_id` (sha1) for idempotent re-ingestion
- **Four loaders:** `CarsLoader` (bespoke), `GenericVehicleLoader` (motorcycles/scooters), `SparePartsLoader` — all inherit `BaseLoader`

## Database

- **PostgreSQL + pgvector:** `brochure_chunks` (text + VECTOR(768)), `brochure_images` (image embeddings), `api_users` (JWT auth)
- **HNSW indexes** over IVFFlat — faster/more accurate; auto-built on startup via `MetadataStore._ensure_indexes()`
- **Async queries** with `asyncpg` throughout retrieval and API layers
- **Category-scoped search:** SQL `WHERE category = ANY(...)` — never mixes categories

## LLM Integration (Ollama, local)

| Model | Role |
|---|---|
| `nomic-embed-text:latest` | Text + image-caption embeddings (768-dim) |
| `llava:latest` | Image captioning (vision); re-encodes AVIF/WebP → JPEG first |
| `qwen2.5:1.5b` | Answer generation (instruction-tuned, multilingual) |

- **Shared `Embedder` class** (`src/ingestion/embedding/embedder.py`): one Ollama client for embed, generate, vision
- **`generate_text`** supports `model` override and `max_tokens` (`num_predict`) params

## Multilingual Response Output

- **Languages:** English, German, Hindi, Telugu (sidebar selector in Streamlit UI)
- **Method:** prompt engineering — language instruction at start AND end of prompt (1.5B models lose mid-prompt instructions in long English context)
- **Model:** `qwen2.5:1.5b` (instruction-tuned general model, NOT `qwen2.5-coder:1.5b-base` which ignores language instructions)
- **Fallback:** if generation fails for non-English, shows English fallback with warning
- **No translation API or library** — zero extra deps/cost

## REST API

- **Framework:** FastAPI with Swagger UI at `/docs`
- **Auth:** JWT (bcrypt password hashing, 24h expiry); signup/login at `/api/v1/signup`, `/api/v1/login`
- **Structure:** versioned under `src/app/api/v1/` with `APIRouter` + tags (auth, products)
- **Endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/signup` | Create account → JWT |
| `POST` | `/api/v1/login` | Authenticate → JWT |
| `GET` | `/api/v1/cars` | List cars |
| `GET` | `/api/v1/motorcycles` | List motorcycles |
| `GET` | `/api/v1/scooters` | List scooters |
| `GET` | `/api/v1/spare-parts` | List spare parts |
| `GET` | `/api/v1/products?category=X` | Unified product endpoint |
| `GET` | `/api/v1/search?q=brake` | Full-text search (optional category filter) |

- **Auth dependency** applied at router level (`dependencies=[Depends(get_current_user)]`), not per-endpoint
- **Shared deps** in `api/deps.py`: DB connection, JWT helpers, Pydantic models

## Streamlit UI

- **Sidebar controls:** search type (Text/Image), category, response language, result count
- **Image grid:** image search results in 3-column layout with product captions
- **Text search:** answer + expandable source details with metadata and images

## Key Files

| File | Purpose |
|---|---|
| `main.py` | Launches Streamlit (8501) + FastAPI (8000) together |
| `src/ingestion/pipeline.py` | Offline ingestion orchestrator |
| `src/ingestion/loader/base_loader.py` | Shared Playwright lifecycle, image download, Ollama preflight |
| `src/ingestion/embedding/embedder.py` | Ollama client (embed, generate, vision) |
| `src/retrieval/retriever.py` | Hybrid search, RRF fusion, multilingual generation |
| `src/app/ui_streamlit.py` | Streamlit front end |
| `src/api/__init__.py` | FastAPI app factory, startup |
| `src/api/deps.py` | JWT auth, DB conn, Pydantic models |
| `src/api/v1/auth.py` | Signup/login routes |
| `src/api/v1/routes.py` | Product and search routes |
| `src/config/settings.py` | Env-based config: DB_PARAMS, model names, vector settings |

## Running

```bash
python main.py                                    # both services
cd src && streamlit run app/ui_streamlit.py    # UI only
cd src && uvicorn app.api:app --port 8000      # API only
python -m ingestion.pipeline --categories cars     # ingestion
```

## Design Decisions

- **Text + image in same embedding space** — both via `nomic-embed-text`, no CLIP encoder
- **JWT over OAuth** — no external identity provider needed for a local project
- **Prompt engineering over translation API** — cheaper, simpler, good enough for 4 languages
- **APIRouter over bare @app** — versioned routes, grouped tags, router-level auth
- **HNSW over IVFFlat** — better recall, auto-migrated on startup

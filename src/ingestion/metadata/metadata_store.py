from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Sequence
from typing import Any

import asyncpg

from config.logger import setup_logger
from config.settings import DB_PARAMS, EMBEDDING_DIM, VECTOR_OPCLASS
from ingestion.normalization.document_normalizer import METADATA_VERSION, NormalizedDocument, build_metadata

logger = setup_logger(__name__)

CATEGORIES = ("cars", "motorcycles", "scooters", "spare_parts")


def to_pgvector(embedding: Sequence[float] | None, *, expected_dim: int | None = EMBEDDING_DIM) -> str | None:
    """Python list -> pgvector literal; ``None`` for empty. A wrong size raises instead of failing in SQL."""
    if embedding is None or len(embedding) == 0:
        return None
    if expected_dim and len(embedding) != expected_dim:
        raise ValueError(f"embedding has {len(embedding)} dimensions but the table expects {expected_dim}")
    return "[" + ",".join(f"{float(x):.8g}" for x in embedding) + "]"
 
 
def _connection_params() -> dict[str, Any]:
    needed = (("user", "POSTGRES_USER"), ("database", "POSTGRES_DB"))
    missing = [env for key, env in needed if not DB_PARAMS.get(key)]
    if missing:
        raise RuntimeError(
            f"Missing {', '.join(missing)}. Define them in a .env file (searched: current directory, src/, "
            "project root) or as environment variables. See .env.example."
        )
    return dict(DB_PARAMS)
 
 
class MetadataStore:
    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None
 
    # -- schema -----------------------------------------------------------------------------------
    async def init_db(self) -> None:
        params = _connection_params()
        try:
            self.pool = await asyncpg.create_pool(**params, min_size=1, max_size=4)
        except (OSError, asyncpg.PostgresError) as exc:  # never echo the password
            raise RuntimeError(
                f"Cannot connect to PostgreSQL at {params['host']}:{params['port']} "
                f"(db={params['database']}, user={params['user']}): {type(exc).__name__}: {exc}"
            ) from exc
        async with self.pool.acquire() as conn:
            await self._ensure_extension(conn)
            await self._create_tables(conn)
            await self._check_dimensions(conn)
            await self._ensure_indexes(conn)
 
    @staticmethod
    async def _ensure_extension(conn: asyncpg.Connection) -> None:
        if await conn.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'"):
            return
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except asyncpg.PostgresError as exc:
            if getattr(exc, "sqlstate", None) == "42501":  # insufficient_privilege
                raise RuntimeError("pgvector is not enabled in this database and your user may not enable it. "
                                   "Run once as a superuser:  CREATE EXTENSION vector;") from exc
            if getattr(exc, "sqlstate", None) == "58P01":  # undefined_file
                raise RuntimeError("pgvector is not installed on the PostgreSQL server. Install it "
                                   "(https://github.com/pgvector/pgvector) and restart PostgreSQL.") from exc
            raise
 
    @staticmethod
    async def _create_tables(conn: asyncpg.Connection) -> None:
        categories = ",".join(f"'{c}'" for c in CATEGORIES)
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS brochure_chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT,
                run_id TEXT,
                chunk_index INT,
                text TEXT,
                content_hash TEXT,
                token_count INT,
                ingestion_timestamp TIMESTAMPTZ,
                metadata_version TEXT,
                category TEXT CHECK (category IN ({categories})),
                metadata JSONB,
                text_embedding VECTOR({EMBEDDING_DIM})
            );""")
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS brochure_images (
                image_id TEXT PRIMARY KEY,
                chunk_id TEXT REFERENCES brochure_chunks(chunk_id) ON DELETE CASCADE,
                image_index INT,
                image_embedding VECTOR({EMBEDDING_DIM}),
                metadata JSONB
            );""")
 
    @staticmethod
    async def _check_dimensions(conn: asyncpg.Connection) -> None:
        """A table created earlier with another size makes every insert fail; say so up front."""
        for table, column in (("brochure_chunks", "text_embedding"), ("brochure_images", "image_embedding")):
            dims = await conn.fetchval(
                "SELECT atttypmod FROM pg_attribute WHERE attrelid = $1::text::regclass AND attname = $2",
                table, column,
            )
            if dims and dims > 0 and dims != EMBEDDING_DIM:
                raise RuntimeError(
                    f"{table}.{column} is VECTOR({dims}) but EMBEDDING_DIM={EMBEDDING_DIM}. Set EMBEDDING_DIM to "
                    f"your embedding model's output size, or drop the tables and re-run."
                )
 
    @staticmethod
    async def _ensure_indexes(conn: asyncpg.Connection) -> None:
        await conn.execute("CREATE INDEX IF NOT EXISTS brochure_chunks_category_idx ON brochure_chunks (category);")
        await conn.execute("CREATE INDEX IF NOT EXISTS brochure_images_chunk_idx ON brochure_images (chunk_id);")
        # Legacy IVFFlat(lists=100) indexes were built on an EMPTY table: very low recall, and ANN-then-filter can
        # return fewer than LIMIT rows with a category filter.
        await conn.execute("DROP INDEX IF EXISTS brochure_chunks_text_embedding_idx;")
        await conn.execute("DROP INDEX IF EXISTS brochure_images_embedding_idx;")
        try:
            await conn.execute("CREATE INDEX IF NOT EXISTS brochure_chunks_text_hnsw_idx ON brochure_chunks "
                               f"USING hnsw (text_embedding {VECTOR_OPCLASS});")
            await conn.execute("CREATE INDEX IF NOT EXISTS brochure_images_hnsw_idx ON brochure_images "
                               f"USING hnsw (image_embedding {VECTOR_OPCLASS});")
        except asyncpg.PostgresError as exc:  # pgvector < 0.5: exact scans are still correct at this scale
            logger.warning("HNSW unavailable (%s); using exact scans", exc)
 
    async def reset(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("TRUNCATE brochure_images, brochure_chunks;")
        logger.warning("Knowledge base truncated (reset requested)")
 
    # -- writing ----------------------------------------------------------------------------------
    async def upsert_documents(self, docs: list[NormalizedDocument], run_id: str) -> tuple[int, int]:
        """Upsert one chunk (+ its image rows) per document. Each runs in its own savepoint: a bad row is
        logged and skipped. Returns (chunks stored, images stored)."""
        stored = image_count = 0
        now = datetime.datetime.now(datetime.timezone.utc)
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for doc in docs:
                    try:
                        async with conn.transaction():  # nested -> SAVEPOINT
                            image_count += await self._upsert_document(conn, doc, run_id, now)
                        stored += 1
                    except (asyncpg.PostgresError, ValueError) as exc:
                        logger.error("Insert failed | doc=%s | %s: %s", doc.document_id, type(exc).__name__, exc)
        return stored, image_count

    @staticmethod
    async def _upsert_document(
        conn: asyncpg.Connection, doc: NormalizedDocument, run_id: str, now: datetime.datetime
    ) -> int:
        chunk_id = f"{doc.category}_{doc.document_id}_0"
        content_hash = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
        token_count = len(doc.text.split())
        meta = {
            **build_metadata(doc),
            "document_id": doc.document_id, "chunk_id": chunk_id, "chunk_index": 0, "run_id": run_id,
            "content_hash": content_hash, "token_count": token_count, "ingestion_timestamp": now.isoformat(),
            "metadata_version": METADATA_VERSION, "category": doc.category,
        }
        await conn.execute(
            """
            INSERT INTO brochure_chunks
                (chunk_id, document_id, run_id, chunk_index, text, content_hash, token_count,
                 ingestion_timestamp, metadata_version, category, metadata, text_embedding)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            ON CONFLICT (chunk_id) DO UPDATE SET
                run_id = EXCLUDED.run_id, text = EXCLUDED.text, content_hash = EXCLUDED.content_hash,
                token_count = EXCLUDED.token_count, ingestion_timestamp = EXCLUDED.ingestion_timestamp,
                metadata_version = EXCLUDED.metadata_version, category = EXCLUDED.category,
                metadata = EXCLUDED.metadata, text_embedding = EXCLUDED.text_embedding
            """,
            chunk_id, doc.document_id, run_id, 0, doc.text, content_hash, token_count, now,
            METADATA_VERSION, doc.category, json.dumps(meta, default=str), to_pgvector(doc.text_embedding),
        )
        await conn.execute("DELETE FROM brochure_images WHERE chunk_id = $1", chunk_id)  # re-runs replace, not append
        for image in doc.images:
            try:
                vector = to_pgvector(image.embedding)
            except ValueError as exc:  # keep the image for display; it just is not searchable by vector
                logger.warning("Image embedding dropped | chunk=%s | %s", chunk_id, exc)
                vector = None
            image_json = {**image.public_dict(), "brand": doc.brand, "model": doc.model,
                          "product_url": doc.product_url, "category": doc.category}
            await conn.execute(
                """
                INSERT INTO brochure_images (image_id, chunk_id, image_index, metadata, image_embedding)
                VALUES ($1,$2,$3,$4,$5)
                """,
                f"{chunk_id}_img{image.index}", chunk_id, image.index,
                json.dumps({k: v for k, v in image_json.items() if v not in (None, "")}, default=str), vector,
            )
        return len(doc.images)

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

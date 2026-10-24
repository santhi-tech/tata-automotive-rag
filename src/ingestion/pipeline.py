"""Offline ingestion pipeline: load -> normalise -> embed(if missing) -> upsert.
Here chunking = one product → one chunk.
Here loader already produced embeddings for images, they are preserved. 
Usage (from ``src/``):
    python -m ingestion.pipeline                      # all categories
    python -m ingestion.pipeline --reset              # rebuild from scratch
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import sys
import uuid

from config.logger import setup_logger
from ingestion.embedding.embedder import Embedder
from ingestion.images import PersistentImageStore
from ingestion.loader.loader import LOADERS
from ingestion.metadata.metadata_store import MetadataStore
from ingestion.normalization.document_normalizer import NormalizedDocument, normalize_document

logger = setup_logger(__name__)


def _embed_missing(embedder: Embedder, doc: NormalizedDocument) -> None:
    """Blocking (HTTP to Ollama) - call via ``asyncio.to_thread``. Only embeds what loaders did not."""
    if not doc.text_embedding and doc.text:
        try:
            doc.text_embedding = embedder.embed_text(doc.text)
        except Exception:
            logger.exception("Text embedding failed | doc=%s", doc.document_id)
    for image in doc.images:
        if image.embedding:
            continue
        try:
            image.embedding = embedder.embed_image(image.local_path)
        except Exception:
            logger.exception("Image embedding failed | %s", image.relative_path)


async def run_ingestion(categories: list[str] | None = None, *, reset: bool = False):
    run_id = f"run_{datetime.datetime.now(datetime.timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    embedder = Embedder()
    store = MetadataStore()
    await store.init_db()

    # STRUCTURAL FIX: always ingest all categories to keep versions in sync
    # --categories is for logging/documentation only; partial ingestion breaks versioning
    if categories:
        logger.warning("[Ingestion] --categories specified but IGNORED; all categories will be ingested to maintain version sync")

    try:
        if reset:
            version = await store.reset()
        else:
            version = await store.next_metadata_version()
        logger.info("[Ingestion] metadata_version=%s (all categories in this version)", version)
        for category in LOADERS:  # Always all categories
            logger.info("[Ingestion] category=%s run=%s", category, run_id)
            raw_docs = await LOADERS[category]().download_and_extract()
            if not raw_docs:
                logger.warning("[Ingestion] no documents for category=%s", category)
                continue

            image_store = PersistentImageStore(category)
            accepted: list[NormalizedDocument] = []
            for raw in raw_docs:
                try:
                    doc = normalize_document(raw, category, image_store)
                except Exception:
                    logger.exception("Normalisation failed; document rejected")
                    continue
                await asyncio.to_thread(_embed_missing, embedder, doc)
                accepted.append(doc)

            chunks, images = await store.upsert_documents(accepted, run_id, version)
            logger.info("[Ingestion] stored category=%s chunks=%d images=%d", category, chunks, images)
        await store._save_version(version)
    finally:
        await store.close()

    logger.info("[Ingestion] run=%s complete", run_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the automotive ingestion pipeline")
    parser.add_argument("--categories", nargs="*", choices=list(LOADERS))
    parser.add_argument("--reset", action="store_true", help="TRUNCATE existing chunks/images first")
    args = parser.parse_args(argv)
    asyncio.run(run_ingestion(args.categories, reset=args.reset))
    return 0


if __name__ == "__main__":
    sys.exit(main())
import pytest
from ingestion.metadata.metadata_store import MetadataStore
from ingestion.normalization.document_normalizer import NormalizedDocument

@pytest.mark.asyncio
async def test_init_db():
    store = MetadataStore()
    await store.init_db()
    # If no exception, DB initialized successfully
    await store.close()

@pytest.mark.asyncio
async def test_upsert_documents():
    store = MetadataStore()
    await store.init_db()
    doc = NormalizedDocument(document_id="test_doc", category="cars", text="Test car chunk", text_embedding=[0.1] * 768)
    chunks, images = await store.upsert_documents([doc], run_id="test_run")
    await store.close()
    assert (chunks, images) == (1, 0)

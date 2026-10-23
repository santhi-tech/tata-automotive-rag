import pytest
from src.ingestion.pipeline import run_ingestion
from src.retrieval.retriever import retrieve_with_sources

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_e2e_pipeline():
    await run_ingestion()
    answer, _ = await retrieve_with_sources("Tell me about electric cars", category="cars", top_k=2)
    assert "electric" in answer.lower()

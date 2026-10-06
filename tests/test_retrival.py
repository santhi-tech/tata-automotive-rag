import pytest
from retrieval.retriever import get_embedding, search_similar

@pytest.mark.asyncio
async def test_search_similar():
    emb = get_embedding("Electric cars")
    rows = await search_similar(emb, category="cars", top_k=1)
    assert isinstance(rows, list)

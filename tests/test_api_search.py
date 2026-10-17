import pytest
from fastapi import HTTPException

from api.v1 import routes

ROWS = [
    {"category": "scooters", "text": "Hero in India", "metadata": {"product_url": "https://www.heromotocorp.com/en-in.html"}},
    {"category": "scooters", "text": "Destini India", "metadata": {"product_url": "https://www.heromotocorp.com/en-in/buy-now/destini-110.html"}},
]


@pytest.fixture(autouse=True)
def fake_keyword_search(monkeypatch):
    async def search(query, categories, top_k):
        return ROWS

    monkeypatch.setattr(routes, "search_keywords", search)


@pytest.mark.asyncio
async def test_search_refuses_non_automotive_questions():
    with pytest.raises(HTTPException) as error:
        await routes.search_products(q="What is the capital of India?", category=None, limit=10)
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_search_returns_products_without_landing_pages():
    result = await routes.search_products(q="destini scooter price", category=None, limit=10)
    assert [p.name for p in result.products] == ["Destini 110"]

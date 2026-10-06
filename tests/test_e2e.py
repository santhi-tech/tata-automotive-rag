import pytest
from src.ingestion.pipeline import run_ingestion
from src.retrieval.retriever import retrieve_pipeline

@pytest.mark.asyncio
async def test_e2e_pipeline():
    await run_ingestion()
    answer = await retrieve_pipeline("Tell me about electric cars", category="cars", top_k=2)
    assert "electric" in answer.lower()

# @pytest.mark.asyncio
# async def test_motorcycles_data():
#     # Run ingestion
#     await run_ingestion()

#     # Retrieve motorcycles
#     answer = await retrieve_pipeline("List Hero motorcycles", category="motorcycle", top_k=5)

#     # Basic checks
#     assert "hero" in answer.lower()
#     assert "motorcycle" in answer.lower()

#     # Check for price presence (₹ symbol or numeric)
#     assert "₹" in answer or any(char.isdigit() for char in answer)

#     # Check for image URLs
#     assert "http" in answer

#     # Check for colours (common words like Red, Blue, Black, Silver, etc.)
#     colours = ["red", "blue", "black", "silver", "white", "grey"]
#     assert any(c in answer.lower() for c in colours)

# @pytest.mark.asyncio
# async def test_engine_and_mileage():
#     await run_ingestion()
#     answer = await retrieve_pipeline("Tell me about Splendor Plus engine and mileage", category="motorcycle", top_k=2)
#     assert "engine" in answer.lower()
#     assert "mileage" in answer.lower()

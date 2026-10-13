import pytest
from retrieval.retriever import feature_list, feature_matches, feature_table, feature_terms, get_embedding, named_products, parse_metadata, product_context, retrieve_with_sources, search_similar, shown_price, understand_query, unindexed_products, wants_comparison


def test_named_products_keeps_only_mentioned_models():
    rows = [
        {"metadata": {"model": "Altroz"}, "text": "a"},
        {"metadata": {"model": "Nexon"}, "text": "n"},
        {"metadata": {"model": "New Tata Aeris"}, "text": "x"},
    ]
    assert [r["text"] for r in named_products("Compare the Altroz and the Punch", rows)] == ["a"]
    assert [r["text"] for r in named_products("tell me about the new aeris", rows)] == ["x"]
    assert named_products("suvs with a sunroof", rows) == []


def test_named_products_uses_url_name_for_motorcycles_and_full_name_match():
    rows = [
        {"metadata": {"product_url": "https://www.heromotocorp.com/en-in/buy-now/glamour-x.html"}, "text": "g"},
        {"metadata": {"product_url": "https://www.heromotocorp.com/en-in/buy-now/glamour-x-abs.html"}, "text": "a"},
        {"metadata": {"product_url": "https://www.heromotocorp.com/en-in/buy-now/destini-110.html"}, "text": "d"},
        {"metadata": {"product_url": "https://www.heromotocorp.com/en-in.html"}, "text": "h"},
    ]
    assert [r["text"] for r in named_products("Tell me about the Glamour X", rows)] == ["g"]
    assert [r["text"] for r in named_products("Tell me about the Glamour X ABS", rows)] == ["a"]
    assert [r["text"] for r in named_products("Tell me about the Destini 110", rows)] == ["d"]
    assert named_products("Which bike has the best mileage?", rows) == []


def test_named_products_tolerates_spelling_variants():
    rows = [
        {"metadata": {"product_url": "https://www.heromotocorp.com/en-in/buy-now/destini-110.html"}, "text": "d"},
        {"metadata": {"product_url": "https://www.heromotocorp.com/en-in/buy-now/glamour-x.html"}, "text": "g"},
    ]
    assert [r["text"] for r in named_products("price of the destiny scooter", rows)] == ["d"]


def test_shared_price_is_not_claimed_per_product():
    rows = [
        {"metadata": {"model": "Destini 110", "price": "₹ 1,72,524 Starting ex-showroom price"}, "text": "a"},
        {"metadata": {"model": "Destini Prime", "price": "₹ 1,72,524 Starting ex-showroom price"}, "text": "b"},
        {"metadata": {"model": "Nexon", "price": "Showroom price ₹7,39,990*"}, "text": "c"},
    ]
    assert "Not reliable" in shown_price(rows, rows[0])
    assert shown_price(rows, rows[2]) == "₹7,39,990"


def test_wants_comparison():
    assert wants_comparison("Compare the Altroz and the Punch")
    assert wants_comparison("Nexon vs Harrier")
    assert not wants_comparison("How many colours does the Altroz come in?")


def test_understand_query_intents_and_entities():
    rows = [{"metadata": {"model": "Altroz"}}, {"metadata": {"model": "Nexon"}}]
    assert understand_query("How many colours does the Altroz come in?", rows).intents == {"colour"}
    assert understand_query("Compare the Altroz and the Punch", rows).intents == {"comparison"}
    assert understand_query("Show the Harrier and the Nexon as a table with price", rows).intents == {"table", "price"}
    assert understand_query("What is the mileage and engine of the Nexon?", rows).intents == {"spec"}
    assert understand_query("Tell me about the Nexon", rows).intents == {"general"}
    named = understand_query("Compare the Altroz and the Punch", rows).named_rows
    assert [r["metadata"]["model"] for r in named] == ["Altroz"]


def test_feature_table_links_price_colours_and_features():
    rows = [{"metadata": {
        "model": "Altroz",
        "product_url": "https://tata.cars/altroz/ice.html",
        "price": "Showroom price in MUMBAI Starting from ₹5,99,990* 27 Variants available",
        "color_variants": ["Pristine White", "Royal Blue"],
        "features": ["6 airbags", "5 star BNCAP", "LED fog lamps", "ESP", "SOS calling"],
    }}]
    table = feature_table(rows)
    assert "[Altroz](https://tata.cars/altroz/ice.html)" in table
    assert "| ₹5,99,990 |" in table
    assert "Pristine White, Royal Blue" in table
    assert "6 airbags; 5 star BNCAP; LED fog lamps; ESP" in table
    assert "SOS calling" not in table


def test_feature_question_lists_only_products_that_mention_it():
    assert feature_terms("Which SUV has a sunroof?") == {"sunroof"}
    rows = [
        {"metadata": {"model": "Curvv Series X", "product_url": "https://tata.cars/curvv/ice.html"},
         "text": "Panoramic sunroof with mood lighting. Ventilated seats."},
        {"metadata": {"model": "Altroz"}, "text": "6 airbags as standard. Price starts at 5.99 Lakh."},
    ]
    matches = feature_matches("Which SUV has a sunroof?", rows)
    assert [parse_metadata(r["metadata"])["model"] for r, _ in matches] == ["Curvv Series X"]
    assert "[Curvv Series X](https://tata.cars/curvv/ice.html)" in feature_list(matches)


def test_product_context_stays_small_per_product():
    long_text = "Sunroof detail. " * 2000
    rows = [{"metadata": {"model": f"Model {i}", "features": ["x" * 500] * 10}, "text": long_text} for i in range(5)]
    assert all(len(block) <= 800 for block in product_context("Which SUV has a sunroof?", rows).split("\n\n") if block)


def test_unindexed_products_flags_models_missing_from_context():
    rows = [{"metadata": {"model": "Altroz"}}]
    assert unindexed_products("Compare the Altroz and the Punch", rows) == ["Punch"]
    assert unindexed_products("Compare the Altroz and the Nexon", rows) == ["Nexon"]
    assert unindexed_products("How many colours does the Altroz come in?", rows) == []


@pytest.mark.asyncio
async def test_search_similar():
    emb = get_embedding("Electric cars")
    rows = await search_similar(emb, category="cars", top_k=1)
    assert isinstance(rows, list)


@pytest.mark.asyncio
async def test_compare_question_returns_only_named_products():
    _answer, rows = await retrieve_with_sources("Compare the Altroz and the Punch", category="cars", top_k=5)
    models = {parse_metadata(r["metadata"]).get("model") for r in rows}
    assert models == {"Altroz"}, models

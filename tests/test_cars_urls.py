from ingestion.loader.cars_loader import CarsLoader

# Real URL shapes from tata.cars; __new__ skips the constructor (no Ollama client, no image dirs).
loader = CarsLoader.__new__(CarsLoader)


def score(url: str) -> int:
    normalized = loader._normalize_url(url)
    return loader._score_vehicle_url(normalized) if normalized else 0


def test_vehicle_pages_score_and_forms_or_other_areas_do_not():
    assert score("https://tata.cars/nexon/ev.html?utm=x#top") == 10
    assert score("/punch/ev.html") == 10  # relative link from the home page
    assert score("https://tata.cars/harrier/ev/overview.html") == 10
    assert score("https://tata.cars/sierra/ice/edition/dark.html") == 1
    assert score("https://tata.cars/nexon/ice/request-a-call-back.html") == 0
    assert score("https://tata.cars/service/booking.html") == 0
    assert score("https://tata.cars/blogs.html") == 0
    assert loader._normalize_url("https://evil.example/nexon/ev.html") is None


def test_model_images_drop_the_site_menu_but_never_empty_the_list():
    menu = ["https://cdn/aeris-dandeli-drizzle.jpg", "https://cdn/curvv-ev.jpg"]
    own = ["https://cdn/altroz-steering.jpg"]
    assert CarsLoader._model_images(menu + own, "https://tata.cars/altroz/ice.html") == own
    assert CarsLoader._model_images(menu, "https://tata.cars/altroz/ice.html") == menu

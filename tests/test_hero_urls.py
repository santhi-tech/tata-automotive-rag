from ingestion.loader.motorcycles_loader import MotorcyclesLoader
from ingestion.loader.scooters_loader import ScootersLoader

HERO = "https://www.heromotocorp.com"


def accepts(loader_class, path: str) -> bool:
    loader = loader_class.__new__(loader_class)  # skip the constructor: no Ollama client, no image dirs
    return loader._is_candidate_product_url(source_url=f"{HERO}/en-in/{loader_class.CATEGORY}.html", candidate_url=HERO + path)


def test_each_loader_keeps_only_its_own_category_pages():
    assert accepts(MotorcyclesLoader, "/en-in/motorcycles/glamour-x.html")
    assert accepts(MotorcyclesLoader, "/en-in/motorcycles/practical/splendor-plus.html")
    assert accepts(ScootersLoader, "/en-in/scooters/destini-110.html")
    assert not accepts(MotorcyclesLoader, "/en-in/scooters/destini-110.html")  # scooter from the site-wide menu
    assert not accepts(ScootersLoader, "/en-in/motorcycles/glamour-x.html")


def test_booking_forms_and_listing_pages_are_not_products():
    assert not accepts(MotorcyclesLoader, "/en-in/buy-now/glamour-x.html")  # category-neutral booking form
    assert not accepts(ScootersLoader, "/en-in/buy-now/destini-110.html")
    assert not accepts(MotorcyclesLoader, "/en-in/motorcycles/karizma-xmr/test-ride.html")
    assert not accepts(MotorcyclesLoader, "/en-in/motorcycles.html")
    assert not accepts(MotorcyclesLoader, "/en-in.html")

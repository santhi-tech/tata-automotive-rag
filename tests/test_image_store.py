from pathlib import Path

from src.ingestion.images import PersistentImageStore


def test_persisted_images_are_category_scoped_and_idempotent(tmp_path: Path):
    source = tmp_path / "downloaded-image.png"
    source.write_bytes(b"image-content")
    store = PersistentImageStore("scooters", root=tmp_path / "data")

    first_path = store.persist(source, "https://example.com/images/vida.png")
    source.unlink()
    second_path = store.persist(first_path, "https://example.com/images/vida.png")

    assert first_path == second_path
    assert first_path.is_file()
    assert first_path.parent == tmp_path / "data" / "scooters" / "images"
    assert store.metadata(first_path)["relative_path"].startswith("data/scooters/images/vida_")

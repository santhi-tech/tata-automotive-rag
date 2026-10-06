from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

class InvalidImageError(ValueError):
    """The file exists but is not a usable image (HTML error page, truncated, 1x1 tracker...)."""


class PersistentImageStore:
    """Owns durable image files for one ingestion category."""

    def __init__(self, category: str, root: str | Path | None = None):
        normalized_category = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in category.strip().lower()
        )
        if not normalized_category:
            raise ValueError("Image category must not be empty")
        repository_root = Path(__file__).resolve().parents[3]
        self.root = (Path(root) if root is not None else repository_root / "data").resolve()
        self.category = normalized_category
        self.directory = self.root / self.category / "images"
        self.directory.mkdir(parents=True, exist_ok=True)

    def destination_for(self, source_url: str | None, source_path: str | Path | None = None) -> Path:
        parsed = urlparse(source_url or "")
        source_name = Path(parsed.path).name or (Path(source_path).name if source_path else "image")
        stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in Path(source_name).stem)
        suffix = Path(source_name).suffix.lower()
        if suffix not in {".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}:
            suffix = ".jpg"
        identity = source_url or str(source_path) or source_name
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return self.directory / f"{stem[:80] or 'image'}_{digest}{suffix}"

    def persist(self, source_path: str | Path, source_url: str | None = None) -> Path:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Image not found: {source}")
        destination = self.destination_for(source_url, source)
        if source.resolve() == destination.resolve():
            return destination
        if not destination.exists():
            temporary_destination = destination.with_suffix(destination.suffix + ".part")
            shutil.copyfile(source, temporary_destination)
            os.replace(temporary_destination, destination)
        return destination

    def metadata(self, image_path: str | Path) -> dict[str, str]:
        path = Path(image_path).resolve()
        return {
            "local_path": str(path),
            "relative_path": path.relative_to(self.root.parent).as_posix(),
            "storage_category": self.category,
        }

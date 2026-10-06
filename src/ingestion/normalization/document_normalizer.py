"""Canonical Document Model + normalizer (anti-corruption layer between loaders and storage).

The four loaders emit four different shapes:

    motorcycles / scooters : top-level brand/price/model/features/source_url,
                             images=[{image_url, local_path, caption, ocr_text, embedding}]
    spare_parts            : {id, text, text_embedding, metadata:{brand, part_name, price,...},
                             images=[{image_url, local_path, ...}]}
    cars (PDF brochures)   : {id, text, images=[{file_path, caption, embedding, ...}]}  (no brand/price)

The normalization layer is a single point of truth for the canonical document model. 
 It is responsible for: the place that knows every loader dialect; everything downstream sees one ``NormalizedDocument``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from config.logger import setup_logger
from config.settings import EMBEDDING_DIM
from ingestion.images import InvalidImageError, PersistentImageStore
from ingestion.normalization.attributes import (
    canonical_brand,
    clean_features,
    clean_text,
    extract_color_variants,
    prettify_slug,
)

logger = setup_logger(__name__)

METADATA_VERSION = "2.0"
_SOURCE_TYPE = {"cars": "pdf_brochure", "motorcycles": "web_product_page", "scooters": "web_product_page", "spare_parts": "spare_part"}


@dataclass
class NormalizedImage:
    index: int
    relative_path: str  # canonical, relative to IMAGE_DATA_DIR, posix separators
    local_path: str  # absolute at ingestion time (informational)
    image_url: str | None = None
    caption: str = ""
    ocr_text: str = ""
    alt: str = ""
    embedding: list[float] = field(default_factory=list)

    def public_dict(self) -> dict[str, Any]:
        """What is stored in chunk metadata (no vectors)."""
        return {
            "relative_path": self.relative_path,
            "local_path": self.local_path,
            "image_url": self.image_url,
            "caption": self.caption,
            "ocr_text": self.ocr_text[:500],
            "alt": self.alt,
        }


@dataclass
class NormalizedDocument:
    document_id: str
    category: str
    text: str
    brand: str | None = None
    model: str | None = None
    price: str | None = None
    description: str | None = None
    features: list[str] = field(default_factory=list)
    color_variants: list[str] = field(default_factory=list)
    product_url: str | None = None
    source_type: str | None = None
    text_embedding: list[float] = field(default_factory=list)
    images: list[NormalizedImage] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)  # loader-specific passthrough (e.g. vehicle_type)


def _pick(*sources: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value not in (None, "", [], {}):
                return value
    return None


def _stable_document_id(category: str, product_url: str | None, brand: str | None, model: str | None, text: str) -> str:
    """Deterministic: re-ingesting the same product yields the same id (the old code used uuid4,
    so every run duplicated every motorcycle and scooter)."""
    identity = product_url or "|".join(filter(None, [brand, model])) or text[:200]
    return "doc_" + hashlib.sha1(f"{category}|{identity}".encode("utf-8")).hexdigest()[:16]


def _normalize_images(raw: dict[str, Any], store: PersistentImageStore) -> list[NormalizedImage]:
    images: list[NormalizedImage] = []
    seen_paths: set[str] = set()
    for position, candidate in enumerate(c for c in (raw.get("images") or []) if isinstance(c, dict)):
        url = candidate.get("image_url") or candidate.get("url")  # loaders disagree: image_url vs url
        path = candidate.get("local_path") or candidate.get("file_path") or candidate.get("path")  # cars: file_path
        if not path:
            logger.warning("image[%d] has no file path (url=%s)", position, url)
            continue
        try:
            stored = store.persist(path, url)
        except (FileNotFoundError, InvalidImageError) as exc:
            logger.warning("image[%d] dropped: %s", position, exc)
            continue
        meta = store.metadata(stored)
        if meta["relative_path"] in seen_paths:
            continue
        seen_paths.add(meta["relative_path"])

        embedding = [float(x) for x in (candidate.get("embedding") or [])]
        if embedding and len(embedding) != EMBEDDING_DIM:
            # A wrong-sized vector would abort the whole category INSERT transaction in pgvector.
            logger.warning("image[%d] embedding has %d dims, expected %d; will re-embed", position, len(embedding), EMBEDDING_DIM)
            embedding = []
        images.append(
            NormalizedImage(
                index=len(images),
                relative_path=meta["relative_path"],
                local_path=meta["local_path"],
                image_url=url,
                caption=clean_text(candidate.get("caption")) or "",
                ocr_text=clean_text(candidate.get("ocr_text")) or "",
                alt=clean_text(candidate.get("alt")) or "",
                embedding=embedding,
            )
        )
    return images


def normalize_document(raw: dict[str, Any], category: str, store: PersistentImageStore) -> NormalizedDocument:
    """Map any loader's output to the canonical model. Never raises for missing optional fields."""
    meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}

    brand = canonical_brand(_pick(meta, raw, keys=("brand", "manufacturer", "make")), category)
    model = clean_text(_pick(meta, raw, keys=("model", "part_name", "product_name", "title", "name")))
    if not model and category == "cars":
        model = prettify_slug(raw.get("id") or raw.get("document_id"))
    product_url = clean_text(_pick(meta, raw, keys=("product_url", "source_url", "link")))
    text = str(raw.get("text") or "").strip()

    features = clean_features(raw.get("features") or meta.get("features"))
    description = clean_text(_pick(meta, raw, keys=("description", "summary", "overview")))
    if not description and features:
        description = "; ".join(features)[:500]

    colors = extract_color_variants(
        _pick(meta, raw, keys=("color_variants", "colour_variants", "colors", "colours")) or [],
        features,
        description,
    )

    images = _normalize_images(raw, store)

    reserved = {"brand", "model", "part_name", "price", "description", "features", "color_variants",
                "product_url", "source_url", "source_type", "category", "images"}
    extra = {k: v for k, v in meta.items() if k not in reserved and v not in (None, "", [], {})}

    return NormalizedDocument(
        # Spare parts (sha256) and cars (file stem) already emit stable ids; motorcycles/scooters emit none.
        document_id=str(raw.get("document_id") or raw.get("id") or _stable_document_id(category, product_url, brand, model, text)),
        category=category,
        text=text,
        brand=brand,
        model=model,
        price=clean_text(_pick(meta, raw, keys=("price", "ex_showroom_price", "mrp"))),
        description=description,
        features=features,
        color_variants=colors,
        product_url=product_url,
        source_type=clean_text(meta.get("source_type")) or _SOURCE_TYPE.get(category),
        text_embedding=[float(x) for x in (raw.get("text_embedding") or [])],
        images=images,
        extra=extra,
    )


def build_metadata(doc: NormalizedDocument) -> dict[str, Any]:
    """The chunk-level JSONB contract. Retrieval, the UI and the audit all rely on these keys."""
    metadata: dict[str, Any] = {
        **doc.extra,
        "brand": doc.brand,
        "model": doc.model,
        "price": doc.price,
        "description": doc.description,
        "features": doc.features,
        "color_variants": doc.color_variants,
        "product_url": doc.product_url,
        "source_url": doc.product_url,
        "source_type": doc.source_type,
        "images": [image.public_dict() for image in doc.images],
    }
    return {k: v for k, v in metadata.items() if v not in (None, "", [], {})}

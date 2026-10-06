"""Pure helpers that clean scraped attributes. No I/O, fully unit-testable."""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

# Brand written by the source site / loader -> canonical display form.
_BRAND_CANON = {"hero": "Hero", "tata": "Tata", "tvs": "TVS", "ktm": "KTM"}
# A loader that emits no brand (the PDF brochure loader) is attributed to its source site.
CATEGORY_DEFAULT_BRAND = {"cars": "Tata"}

_SLUG_NOISE = re.compile(r"\b(brochure|brochures|pdf|catalogue|catalog|leaflet|download|web|final|new)\b", re.I)

_COLOR_WORDS = {
    "red", "black", "blue", "white", "grey", "gray", "silver", "green", "yellow", "orange", "brown",
    "maroon", "purple", "pink", "gold", "golden", "bronze", "copper", "beige", "teal", "violet",
    "titanium", "bronze", "cyan", "burgundy", "navy",
}
_COLOR_LIST_RE = re.compile(r"\bcolou?rs?\s*(?:options?|variants?)?\s*[:\-–]\s*(?P<body>[^.\n|]{3,200})", re.I)
_AVAILABLE_IN_RE = re.compile(r"\bavailable\s+in\s+(?P<body>[^.\n|]{3,200}?)\s+colou?rs?\b", re.I)
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z&'\-]*(?: [A-Za-z][A-Za-z&'\-]*){0,3}$")
_SPLIT_RE = re.compile(r"\s*(?:,|;|/|\band\b|&)\s*", re.I)


def canonical_brand(value: Any, category: str | None = None) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return CATEGORY_DEFAULT_BRAND.get(category or "")
    mapped = _BRAND_CANON.get(text.lower())
    if mapped:
        return mapped
    return text.title() if text.islower() else text


def prettify_slug(value: Any) -> str | None:
    """'Tiago-EV-brochure' -> 'Tiago EV'. Used only when a brochure carries no model field."""
    text = _SLUG_NOISE.sub(" ", str(value or "").replace("_", " ").replace("-", " "))
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def clean_text(value: Any) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text or None


def clean_features(items: Iterable[Any] | None, limit: int = 30) -> list[str]:
    """Dedupe, drop empty / absurd entries. (Scrapers using ``ul li`` also capture menu items;
    the length/dedupe filter removes most noise, selector scoping in the loader is the real fix.)"""
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items or []:
        text = clean_text(item)
        if not text or not (2 <= len(text) <= 200) or text.lower() in seen:
            continue
        seen.add(text.lower())
        cleaned.append(text)
        if len(cleaned) == limit:
            break
    return cleaned


def _is_color_label(label: str) -> bool:
    words = label.lower().split()
    return bool(words) and len(words) <= 4 and words[-1] in _COLOR_WORDS and bool(_LABEL_RE.match(label))


def extract_color_variants(*sources: Any, limit: int = 12) -> list[str]:
    """Best-effort, *explicit-mention-only* colour extraction (never inferred from image content).

    Recognises (a) swatch-style entries such as ``"Sports Red"`` in a features list,
    (b) ``"Colours: Red, Black and Blue"`` and (c) ``"available in Red and Black colours"``.
    """
    found: list[str] = []
    seen: set[str] = set()

    def add(label: str) -> None:
        label = re.sub(r"\s+", " ", label).strip(" .:-")
        if _is_color_label(label) and label.lower() not in seen:
            seen.add(label.lower())
            found.append(label.title() if label.islower() else label)

    for source in sources:
        entries = source if isinstance(source, (list, tuple)) else [source]
        for entry in entries:
            text = clean_text(entry)
            if not text:
                continue
            add(text)  # swatch label on its own
            for pattern in (_COLOR_LIST_RE, _AVAILABLE_IN_RE):
                for match in pattern.finditer(text):
                    for part in _SPLIT_RE.split(match.group("body")):
                        add(part)
    return found[:limit]

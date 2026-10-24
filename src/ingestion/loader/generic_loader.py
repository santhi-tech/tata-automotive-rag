from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from config.logger import setup_logger
from ingestion.embedding.embedder import Embedder
from ingestion.images import PersistentImageStore
from ingestion.loader.base_loader import BaseLoader


logger = setup_logger(__name__)


class GenericVehicleLoader(BaseLoader):
    """Dynamic vehicle loader: discovers product pages per SOURCE and returns RAG documents in memory."""

    # ------------------------------------------------------------
    # URL exclusions
    # ------------------------------------------------------------

    EXCLUDED_PATH_TERMS = (
        "/about",
        "/contact",
        "/privacy",
        "/terms",
        "/policy",
        "/login",
        "/register",
        "/account",
        "/cart",
        "/checkout",
        "/wishlist",
        "/search",
        "/faq",
        "/blog",
        "/news",
        "/sitemap",
        "/dealer",
        "/dealers",
        "/service",
        "/offers",
        "/finance",
        "/insurance",
        "/careers",
        "/investor",
        "/wp-admin",
        "/wp-login",
    )

    EXCLUDED_IMAGE_TERMS = (
        "logo",
        "favicon",
        "icon",
        "placeholder",
        "payment",
        "facebook",
        "instagram",
        "youtube",
        "twitter",
        "linkedin",
        "whatsapp",
        "captcha",
        "sprite",
        "arrow",
        "close",
        "menu",
    )

    VEHICLE_SIGNALS = (
        "engine",
        "transmission",
        "mileage",
        "fuel",
        "battery",
        "range",
        "power",
        "torque",
        "ground clearance",
        "wheelbase",
        "safety",
        "airbag",
        "abs",
        "brake",
        "dimensions",
        "variant",
        "colour",
        "color",
        "price",
        "specification",
        "features",
        "vehicle",
        "motorcycle",
        "scooter",
        "car",
        "suv",
        "bike",
    )

    PAGE_SIGNALS = VEHICLE_SIGNALS
    MIN_IMAGE_SIZE = (200, 120)
    NAV_TEXT_LIMIT = 5000

    VISION_PROMPT = (
        "Analyze this vehicle image for a multimodal "
        "automotive knowledge base. Describe ONLY visible "
        "information. Identify visible vehicle characteristics, "
        "body style, design elements, dashboard/display elements, "
        "wheels, lights, colors, badges and visible text. "
        "If text or labels are visible, reproduce them accurately. "
        "Do not invent specifications, price, compatibility, "
        "safety ratings or performance information."
    )

    def __init__(self, embedder: Optional[Embedder] = None):
        if not self.SOURCES:
            raise ValueError(f"{self.__class__.__name__}.SOURCES cannot be empty")

        super().__init__(self.SOURCES[0]["url"])

        self.embedder = embedder or Embedder()

        category_env = self.CATEGORY.upper()

        self.vision_timeout = int(os.getenv(f"{category_env}_VISION_TIMEOUT_SECONDS", "90"))

        self.embed_timeout = int(os.getenv(f"{category_env}_EMBED_TIMEOUT_SECONDS", "30"))

        self.max_images_per_product = max(0, int(os.getenv(f"{category_env}_MAX_IMAGES_PER_PRODUCT", "4")))

        self.max_products = max(1, int(os.getenv(f"{category_env}_MAX_PRODUCTS", "5")))

        self.enable_vision = os.getenv(f"{category_env}_ENABLE_VISION", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

        self.image_store = PersistentImageStore(self.CATEGORY)

        logger.info(
            "%s initialized | category=%s | products=%d | images=%d",
            self.__class__.__name__,
            self.CATEGORY,
            self.max_products,
            self.max_images_per_product,
        )

    # ============================================================
    # URL DISCOVERY
    # ============================================================

    def _is_candidate_product_url(self, source_url: str, candidate_url: str) -> bool:

        source = self._normalize_url(source_url)

        candidate = self._normalize_url(candidate_url)

        if not candidate or candidate == source:
            return False

        source_host = urlparse(source).netloc.lower()

        candidate_parsed = urlparse(candidate)

        if candidate_parsed.scheme not in ("http", "https"):
            return False

        if candidate_parsed.netloc.lower() != source_host:
            return False

        path = candidate_parsed.path.lower().rstrip("/")

        if not path:
            return False

        if any(term in path for term in self.EXCLUDED_PATH_TERMS):
            return False

        if path.endswith((".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".zip", ".mp4")):
            return False

        # Hero files each model under its own category (/en-in/scooters/xoom.html). /buy-now/ pages are
        # category-neutral booking forms linked from the site-wide menu: accepting them (or the other category's
        # path, or any "BUY NOW" link text) is how scooters were stored as motorcycles.
        return f"/{self.CATEGORY}/" in path and "/buy-now/" not in path and "test-ride" not in path

    async def discover_product_links(self, page: Page, source: Dict[str, Any]) -> List[str]:
        source_url = source["url"]

        logger.info("Vehicle discovery | category=%s | url=%s", self.CATEGORY, source_url)

        links: set[str] = set()

        try:
            await page.goto(source_url, wait_until="domcontentloaded", timeout=120000)

            await page.wait_for_timeout(3000)

            await self._scroll_page(page)

            link_data = await page.locator("a[href]").evaluate_all(
                """
                elements => elements.map(a => ({
                    href: a.href || "",
                    text: (
                        a.innerText ||
                        a.textContent ||
                        ""
                    ).trim()
                }))
                """
            )

            for item in link_data:
                href = self._normalize_url(item.get("href", ""))

                if self._is_candidate_product_url(
                    source_url=source_url, candidate_url=href
                ):
                    links.add(href.split("?", 1)[0])  # ?tab=… variants are the same page; the scraper never switches tabs

        except PlaywrightTimeoutError:
            logger.exception("Discovery timeout | %s", source_url)

        except Exception:
            logger.exception("Discovery failed | %s", source_url)

        result = sorted(links)

        logger.info("Discovered %d vehicle URLs | category=%s", len(result), self.CATEGORY)

        return result

    # ============================================================
    # TEXT EXTRACTION HELPERS
    # ============================================================

    async def _extract_first_text(self, page: Page, selectors: List[str], timeout: int = 3000) -> str:

        for selector in selectors:
            try:
                locator = page.locator(selector).first

                if await locator.count():
                    value = (await locator.inner_text(timeout=timeout)).strip()

                    if value:
                        return value

            except Exception:
                continue

        return ""

    async def _extract_all_text(self, page: Page, selectors: List[str]) -> List[str]:

        values: List[str] = []

        for selector in selectors:
            try:
                locator = page.locator(selector)

                count = await locator.count()

                for index in range(count):
                    try:
                        value = (await locator.nth(index).inner_text(timeout=2000)).strip()

                        if value:
                            values.append(value)

                    except Exception:
                        continue

            except Exception:
                continue

        return values

    # ============================================================
    # VEHICLE PAGE EXTRACTION
    # ============================================================

    async def _get_product_container(self, page: Page) -> Optional[str]:
        """Find the product-specific content container, skipping site-wide banners.
        Hero pages have a top banner with shared content; actual product details are below."""
        containers = [
            "main",
            "[role='main']",
            "[class*='product-detail']",
            "[class*='product-section']",
            "[class*='vehicle-detail']",
            "[id*='product']",
        ]
        for selector in containers:
            count = await page.locator(selector).count()
            if count > 0:
                return selector
        return None

    async def _extract_from_container(self, page: Page, container_selector: Optional[str], selectors: List[str], timeout: int = 3000) -> str:
        """Extract text from a specific container, or fall back to page-wide search."""
        if container_selector:
            for selector in selectors:
                try:
                    locator = page.locator(container_selector).locator(selector).first
                    if await locator.count():
                        value = (await locator.inner_text(timeout=timeout)).strip()
                        if value and not self._is_navigation_text(value):
                            return value
                except Exception:
                    continue
        return await self._extract_first_text(page, selectors, timeout)

    def _is_navigation_text(self, text: str) -> bool:
        """Filter out banner/navigation text that appears on all pages."""
        nav_indicators = ("select model", "buy now", "test ride", "add to cart", "explore", "shop now")
        return any(phrase in text.lower() for phrase in nav_indicators)

    async def extract_product_page(self, page: Page, product_url: str, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:

        try:
            await page.goto(product_url, wait_until="domcontentloaded", timeout=120000)

            await page.wait_for_timeout(1800)

            product_container = await self._get_product_container(page)

            title = await self._extract_from_container(
                page,
                product_container,
                [
                    "h1",
                    "[class*='vehicle-name']",
                    "[class*='product-title']",
                    "[class*='model-name']",
                    "[class*='vehicle-title']",
                    "h2",
                ],
            )

            description = await self._extract_from_container(
                page,
                product_container,
                [
                    "[class*='description']",
                    "[class*='vehicle-description']",
                    "[class*='product-description']",
                    "[id*='description']",
                    "article",
                    "p[class*='product']",
                ],
            )

            price = await self._extract_from_container(
                page, product_container, ["[class*='price']", "[id*='price']", ".price", "[data-price]"]
            )

            # ----------------------------------------------------
            # COMPLETE PAGE TEXT
            # ----------------------------------------------------

            page_text = ""

            try:
                page_text = (await page.locator("body").inner_text(timeout=10000)).strip()

            except Exception:
                logger.warning("Could not extract body text | %s", product_url)

            if not title and not page_text:
                return None

            if self._looks_like_navigation_page(title, page_text):
                logger.info("Skipping non-vehicle page | %s", product_url)
                return None

            # ----------------------------------------------------
            # JSON-LD
            # ----------------------------------------------------

            json_ld = await self._extract_json_ld(page)

            # ----------------------------------------------------
            # VARIANTS
            # ----------------------------------------------------

            variants = await self._extract_all_text(
                page, ["[class*='variant']", "[class*='trim']", "[class*='model-variant']", "[data-variant]"]
            )

            # ----------------------------------------------------
            # COLORS
            # ----------------------------------------------------

            colors = await self._extract_all_text(
                page,
                [
                    "[class*='color-name']",
                    "[class*='colour-name']",
                    "[class*='color']",
                    "[class*='colour']",
                    "[data-color]",
                    "[data-colour]",
                ],
            )

            # ----------------------------------------------------
            # FEATURES
            # ----------------------------------------------------

            features = await self._extract_all_text(
                page, ["[class*='feature']", "[class*='highlight']", "[class*='key-feature']", "[data-feature]"]
            )

            # ----------------------------------------------------
            # SAFETY
            # ----------------------------------------------------

            safety = await self._extract_all_text(page, ["[class*='safety']", "[class*='security']", "[data-safety]"])

            # ----------------------------------------------------
            # SPECIFICATIONS
            # ----------------------------------------------------

            specifications = await self._extract_specifications(page)

            # ----------------------------------------------------
            # IMAGES
            # ----------------------------------------------------

            images = await self._extract_images(page, product_url, product_container)

            return {
                "brand": source["brand"],
                "vehicle_type": self.CATEGORY,
                "product_url": product_url,
                "vehicle_name": (title or self._fallback_name(product_url)),
                "description": description,
                "price": price,
                "variants": self._dedupe(variants),
                "colors": self._dedupe(colors),
                "features": self._dedupe(features),
                "safety": self._dedupe(safety),
                "specifications": specifications,
                "page_text": page_text,
                "json_ld": json_ld,
                "images": images,
                "extraction_timestamp": self._timestamp(),
            }

        except PlaywrightTimeoutError:
            logger.exception("Vehicle page timeout | %s", product_url)
            return None

        except Exception:
            logger.exception("Vehicle extraction failed | %s", product_url)
            return None

    async def _extract_specifications(self, page: Page) -> Dict[str, str]:

        specifications: Dict[str, str] = {}

        # --------------------------------------------------------
        # TABLES
        # --------------------------------------------------------

        try:
            rows = await page.locator("table tr").evaluate_all(
                """
                rows => rows.map(row => {
                    const cells = Array.from(
                        row.querySelectorAll("th, td")
                    ).map(c =>
                        (c.innerText || "").trim()
                    );

                    return cells;
                })
                """
            )

            for row in rows:
                if len(row) >= 2:
                    key = row[0]
                    value = " | ".join(x for x in row[1:] if x)

                    if key and value:
                        specifications[key[:200]] = value[:1000]

        except Exception:
            logger.debug("Table specification extraction skipped")

        # --------------------------------------------------------
        # COMMON SPEC ATTRIBUTE BLOCKS
        # --------------------------------------------------------

        selectors = [
            "[class*='specification']",
            "[class*='specs']",
            "[class*='technical']",
            "[class*='detail']",
            "[data-spec]",
        ]

        for selector in selectors:
            try:
                elements = await page.locator(selector).evaluate_all(
                    """
                    elements => elements.map(e => ({
                        text: (
                            e.innerText ||
                            e.textContent ||
                            ""
                        ).trim()
                    }))
                    """
                )

                for element in elements:
                    text = (element.get("text") or "").strip()

                    if not text:
                        continue

                    lines = [line.strip() for line in text.splitlines() if line.strip()]

                    for index in range(0, len(lines) - 1, 2):
                        key = lines[index]
                        value = lines[index + 1]

                        if len(key) < 100 and value:
                            specifications.setdefault(key, value[:1000])

            except Exception:
                continue

        return specifications

    # ============================================================
    # IMAGES
    # ============================================================

    async def _extract_images(self, page: Page, product_url: str, product_container: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract images, preferring product-specific container over whole page."""
        selector = f"{product_container} img" if product_container else "img"

        image_elements = await page.locator(selector).evaluate_all(
            """
            images => images.map(img => ({
                src:
                    img.currentSrc ||
                    img.src ||
                    img.getAttribute("data-src") ||
                    img.getAttribute("data-lazy-src") ||
                    img.getAttribute("data-original") ||
                    "",
                alt: img.alt || "",
                width: img.naturalWidth || 0,
                height: img.naturalHeight || 0
            }))
            """
        )

        results: List[Dict[str, Any]] = []

        seen: set[str] = set()

        for image in image_elements:
            raw_url = image.get("src", "")

            if not raw_url:
                continue

            image_url = urljoin(product_url, raw_url)

            if image_url.startswith("data:"):
                continue

            if image_url in seen:
                continue

            if not self._is_useful_image(image):
                continue

            seen.add(image_url)

            results.append(
                {
                    "url": image_url,
                    "alt": (image.get("alt") or ""),
                    "width": (image.get("width") or 0),
                    "height": (image.get("height") or 0),
                }
            )

            if len(results) >= self.max_images_per_product:
                break

        return results

    # ============================================================
    # RAG DOCUMENT
    # ============================================================

    async def build_rag_document(self, product: Dict[str, Any], image_results: List[Dict[str, Any]]) -> Dict[str, Any]:

        sections: List[str] = []

        sections.append(f"Brand: {product['brand']}")

        sections.append(f"Vehicle type: {product['vehicle_type']}")

        sections.append(f"Vehicle name: {product['vehicle_name']}")

        if product.get("price"):
            sections.append(f"Price: {product['price']}")

        if product.get("description"):
            sections.append("Description:\n" + product["description"])

        if product.get("variants"):
            sections.append("Variants:\n" + "\n".join(product["variants"]))

        if product.get("colors"):
            sections.append("Colors:\n" + "\n".join(product["colors"]))

        if product.get("features"):
            sections.append("Features:\n" + "\n".join(product["features"]))

        if product.get("safety"):
            sections.append("Safety:\n" + "\n".join(product["safety"]))

        specifications = product.get("specifications") or {}

        if specifications:
            spec_text = "\n".join(f"{key}: {value}" for key, value in specifications.items())

            sections.append("Specifications:\n" + spec_text)

        sections.append("Product page:\n" + product["product_url"])

        if product.get("page_text"):
            sections.append("Source page content:\n" + product["page_text"])

        # --------------------------------------------------------
        # IMAGE KNOWLEDGE
        # --------------------------------------------------------

        image_text: List[str] = []

        for image in image_results:
            if image.get("caption"):
                image_text.append("Image description:\n" + image["caption"])

            if image.get("ocr_text"):
                image_text.append("Image OCR:\n" + image["ocr_text"])

        if image_text:
            sections.append("\n\n".join(image_text))

        rag_text = "\n\n".join(section for section in sections if section).strip()

        text_embedding = await self._ollama_embed(rag_text)

        return {
            "id": self._make_document_id(product),
            "text": rag_text,
            "text_embedding": text_embedding,
            "metadata": {
                "brand": product["brand"],
                "vehicle_type": product["vehicle_type"],
                "vehicle_name": product["vehicle_name"],
                "price": product.get("price"),
                "product_url": product["product_url"],
                "source_url": product["product_url"],
                "source_type": "vehicle",
                "category": self.CATEGORY,
                "extraction_timestamp": product["extraction_timestamp"],
                "embedding_model": self.embedder.embed_model,
                "embedding_dimension": len(text_embedding),
            },
            "images": image_results,
        }

    @staticmethod
    def _make_document_id(product: Dict[str, Any]) -> str:

        value = f"{product['brand']}:{product['vehicle_type']}:{product['product_url']}"

        return hashlib.sha256(value.encode("utf-8")).hexdigest()

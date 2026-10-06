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


class SparePartsLoader(BaseLoader):
    """Dynamic spare-parts loader for Tata and Hero; product URLs are discovered at runtime."""

    CATEGORY = "spare_parts"
    SOURCES = [
        {"brand": "tata", "url": "https://tgpindia.com/engine-parts/", "vehicle_types": ["car_engines"]},
        {
            "brand": "hero",
            "url": "https://shop.heromotocorp.com/en/collection/spare",
            "vehicle_types": ["motorcycle", "scooter"],
        },
    ]

    # Navigation/account/legal/UI URLs should never become product documents.
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
        "banner",
        "sprite",
    )

    PAGE_SIGNALS = ("part number", "sku", "add to cart", "price", "spare part", "product")

    VISION_PROMPT = (
        "Analyze this automotive spare-part image. Describe only "
        "visible information. Identify the part if possible, visible "
        "labels, part numbers, text, shape and useful characteristics. "
        "Do not invent compatibility information."
    )

    def __init__(self, embedder: Optional[Embedder] = None):
        # BaseLoader needs a base URL. Actual sources are processed dynamically.
        super().__init__(self.SOURCES[0]["url"])
        self.embedder = embedder or Embedder()
        self.vision_timeout = int(os.getenv("SPARE_PARTS_VISION_TIMEOUT_SECONDS", "90"))
        self.embed_timeout = int(os.getenv("SPARE_PARTS_EMBED_TIMEOUT_SECONDS", "30"))
        self.max_images_per_product = max(0, int(os.getenv("SPARE_PARTS_MAX_IMAGES_PER_PRODUCT", "2")))
        self.max_products = max(1, int(os.getenv("SPARE_PARTS_MAX_PRODUCTS", "5")))
        self.enable_vision = os.getenv("SPARE_PARTS_ENABLE_VISION", "true").strip().lower() in {"1", "true", "yes", "y"}
        self.image_store = PersistentImageStore(self.CATEGORY)
        logger.info(
            "Spare-parts loader limits | products=%d | images_per_product=%d",
            self.max_products,
            self.max_images_per_product,
        )

    # ============================================================
    # PRODUCT URL DISCOVERY
    # ============================================================

    def _is_candidate_product_url(self, source_url: str, candidate_url: str, brand: str, link_text: str = "") -> bool:
        """
        URL classifier.

        We intentionally do NOT hard-code model names.

        The classifier removes obvious navigation/UI URLs and accepts links
        that look product-oriented. The link text is also used as a signal,
        which is useful for sites whose URL structure is not standardized.
        """

        source = self._normalize_url(source_url)
        candidate = self._normalize_url(candidate_url)

        if not candidate or candidate == source:
            return False

        parsed_source = urlparse(source)
        parsed = urlparse(candidate)

        if parsed.scheme not in ("http", "https"):
            return False

        if parsed.netloc.lower() != parsed_source.netloc.lower():
            return False

        path = parsed.path.lower().rstrip("/")
        text = " ".join((link_text or "").lower().split())

        if not path:
            return False

        if any(term in path for term in self.EXCLUDED_PATH_TERMS):
            return False

        # TGP category pages must never be classified as products. In the
        # previous version /body-parts-rubber-parts was accepted because
        # the generic word "part" appeared in its URL.
        tgp_category_markers = (
            "/body-parts-rubber-parts",
            "/engine-parts",
            "/frictional-parts",
            "/filters-bearings",
            "/transmission-parts",
            "/accessories",
            "/lubricants",
            "/def",
            "/product",
            "/collection/",
            "/collections/",
        )
        if brand == "tata" and any(marker in path for marker in tgp_category_markers):
            return False

        # Never crawl obvious file/document links as product pages.
        if path.endswith((".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".zip")):
            return False

        # Site-specific URL patterns are structural, not model hard-coding.
        if brand == "hero":
            if "/collection/" in path:
                return False
            if "/product/" in path or "/products/" in path:
                return True

        if brand == "tata":
            if "/product/" in path or "/products/" in path:
                return True

            # TGP may expose product pages through WooCommerce-like slugs.
            # A product-looking slug plus spare/part terminology is accepted.
            product_terms = (
                "spare",
                "part",
                "brake",
                "filter",
                "clutch",
                "lamp",
                "mirror",
                "bumper",
                "radiator",
                "bearing",
                "gasket",
                "sensor",
            )
            if any(term in path for term in product_terms):
                return True

        # Generic fallback: accept a link whose visible text strongly looks
        # like a product/part, but reject generic navigation labels.
        generic_product_terms = ("spare", "part", "product", "buy", "view product", "add to cart")

        if any(term in text for term in generic_product_terms):
            return True

        return False

    async def discover_product_links(self, page: Page, source: Dict[str, Any]) -> List[str]:
        """
        Discover product pages using the already-running Playwright page.

        This method deliberately does not launch or close Playwright.
        Lifecycle ownership belongs to download_and_extract().
        """
        source_url, brand = source["url"], source["brand"]

        logger.info("Discovering spare parts | brand=%s | url=%s", brand, source_url)

        links: set[str] = set()

        try:
            await page.goto(source_url, wait_until="domcontentloaded", timeout=120000)

            await page.wait_for_timeout(3000)
            await self._scroll_page(page, wait_ms=1000)

            link_data = await page.locator("a[href]").evaluate_all(
                """
                elements => elements.map(a => ({
                    href: a.href || "",
                    text: (a.innerText || a.textContent || "").trim()
                }))
                """
            )

            for item in link_data:
                href = self._normalize_url(item.get("href", ""))
                text = item.get("text", "")

                if self._is_candidate_product_url(
                    source_url=source_url, candidate_url=href, brand=brand, link_text=text
                ):
                    links.add(href)

        except PlaywrightTimeoutError:
            logger.exception("Timed out discovering product links | brand=%s | url=%s", brand, source_url)
        except Exception:
            logger.exception("Failed discovering product links | brand=%s | url=%s", brand, source_url)

        result = sorted(links)

        logger.info("Discovered %d candidate product links | brand=%s", len(result), brand)

        return result

    # ============================================================
    # PRODUCT PAGE EXTRACTION
    # ============================================================

    async def extract_product_page(self, page: Page, product_url: str, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract one product page. Failure is isolated to this URL."""
        brand, vehicle_types = source["brand"], source["vehicle_types"]

        try:
            await page.goto(product_url, wait_until="domcontentloaded", timeout=120000)

            await page.wait_for_timeout(1500)

            # ----------------------------------------------------
            # PAGE TITLE
            # ----------------------------------------------------

            title = ""

            try:
                locator = page.locator("h1").first
                if await locator.count():
                    title = (await locator.inner_text(timeout=5000)).strip()
            except Exception:
                pass

            # ----------------------------------------------------
            # DESCRIPTION
            # ----------------------------------------------------

            description = ""

            selectors = [
                "[class*='description']",
                "[class*='product-description']",
                "[id*='description']",
                ".description",
                "article",
                "main",
            ]

            for selector in selectors:
                try:
                    locator = page.locator(selector).first

                    if await locator.count():
                        text = (await locator.inner_text(timeout=3000)).strip()

                        if text:
                            description = text
                            break

                except Exception:
                    continue

            # ----------------------------------------------------
            # PRICE
            # ----------------------------------------------------

            price = None

            price_selectors = ["[class*='price']", "[id*='price']", ".price", ".product-price"]

            for selector in price_selectors:
                try:
                    locator = page.locator(selector).first

                    if await locator.count():
                        value = (await locator.inner_text(timeout=2000)).strip()

                        if value:
                            price = value
                            break

                except Exception:
                    continue

            # ----------------------------------------------------
            # IMAGES
            # ----------------------------------------------------

            image_elements = await page.locator("img").evaluate_all(
                """
                images => images.map(img => ({
                    src: img.currentSrc ||
                         img.src ||
                         img.getAttribute("data-src") ||
                         img.getAttribute("data-lazy-src") ||
                         "",
                    alt: img.alt || "",
                    width: img.naturalWidth || 0,
                    height: img.naturalHeight || 0
                }))
                """
            )

            image_urls: List[Dict[str, Any]] = []

            for image in image_elements:
                raw_url = image.get("src", "")
                if not raw_url:
                    continue

                image_url = urljoin(product_url, raw_url)

                if image_url.startswith("data:"):
                    continue

                if not self._is_useful_image(image):
                    continue

                image_urls.append(
                    {
                        "url": image_url,
                        "alt": image.get("alt") or "",
                        "width": image.get("width") or 0,
                        "height": image.get("height") or 0,
                    }
                )

            # Deduplicate while preserving order.
            unique_images: Dict[str, Dict[str, Any]] = {}

            for image in image_urls:
                unique_images.setdefault(image["url"], image)

            image_urls = list(unique_images.values())

            # Apply max_images_per_product limit here
            image_urls = image_urls[: self.max_images_per_product]

            # ----------------------------------------------------
            # COMPLETE PAGE TEXT
            # ----------------------------------------------------

            page_text = ""

            try:
                body = page.locator("body")
                page_text = (await body.inner_text(timeout=10000)).strip()

            except Exception:
                logger.warning("Could not extract body text | %s", product_url)

            if not title and not page_text:
                logger.warning("Empty product page | %s", product_url)
                return None

            # If a weak URL classifier let a page through but the page
            # obviously looks like navigation, do not create a RAG document.
            if self._looks_like_navigation_page(title, page_text):
                logger.info("Skipping non-product page | %s", product_url)
                return None

            return {
                "brand": brand,
                "vehicle_types": vehicle_types,
                "product_url": product_url,
                "part_name": title or self._fallback_name(product_url, "unknown-part"),
                "description": description,
                "price": price,
                "page_text": page_text,
                "images": image_urls,
                "extraction_timestamp": self._timestamp(),
            }

        except PlaywrightTimeoutError:
            logger.exception("Product page timed out | %s", product_url)
            return None

        except Exception:
            logger.exception("Product extraction failed | %s", product_url)
            return None

    # ============================================================
    # RAG DOCUMENT
    # ============================================================

    async def build_rag_document(self, product: Dict[str, Any], image_results: List[Dict[str, Any]]) -> Dict[str, Any]:

        image_text: List[str] = []

        for image in image_results:
            if image.get("caption"):
                image_text.append(f"Image description: {image['caption']}")

            if image.get("ocr_text"):
                image_text.append(f"Image OCR: {image['ocr_text']}")

        rag_parts = [
            f"Brand: {product['brand']}",
            ("Vehicle types: " + ", ".join(product["vehicle_types"])),
            f"Part name: {product['part_name']}",
        ]

        if product.get("price"):
            rag_parts.append(f"Price: {product['price']}")

        if product.get("description"):
            rag_parts.append(f"Description: {product['description']}")

        rag_parts.append(f"Product page: {product['product_url']}")

        if product.get("page_text"):
            rag_parts.append(f"Page content:\n{product['page_text']}")

        if image_text:
            rag_parts.append("\n".join(image_text))

        rag_text = "\n\n".join(part for part in rag_parts if part).strip()

        # --------------------------------------------------------
        # Text embedding
        # --------------------------------------------------------

        text_embedding = await self._ollama_embed(rag_text)

        return {
            "id": self._make_document_id(product),
            "text": rag_text,
            "text_embedding": text_embedding,
            "metadata": {
                "brand": product["brand"],
                "vehicle_type": product["vehicle_types"],
                "part_name": product["part_name"],
                "price": product.get("price"),
                "product_url": product["product_url"],
                "source_type": "spare_part",
                "source_url": product["product_url"],
                "extraction_timestamp": product["extraction_timestamp"],
                "embedding_type": ("image_description_text_embedding"),
            },
            "images": image_results,
        }

    @staticmethod
    def _make_document_id(product: Dict[str, Any]) -> str:
        value = f"{product['brand']}:{product['product_url']}"

        return hashlib.sha256(value.encode("utf-8")).hexdigest()

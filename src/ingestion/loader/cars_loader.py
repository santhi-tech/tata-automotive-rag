from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse, urldefrag

import aiohttp
from playwright.async_api import Page, async_playwright

from config.logger import setup_logger
from ingestion.embedding.embedder import Embedder
from ingestion.loader.base_loader import BaseLoader
from ingestion.images import PersistentImageStore


LOGGER = setup_logger("ingestion.loader.cars")


class CarsLoader(BaseLoader):
    """
    Production-oriented Tata Cars web loader.

    Important:
    - Does NOT write JSON.
    - Does NOT write PDFs.
    - Returns List[Dict[str, Any]] in memory.
    - Uses Playwright for dynamic discovery.
    - Uses BaseLoader's Embedder-backed helpers for Ollama/image operations.
    - Uses PersistentImageStore for durable images.
    - Is resilient to Tata URL structure changes.
    """

    CATEGORY = "cars"
    BRAND = "tata"

    SOURCE_URLS = ["https://tata.cars/","https://tata.cars/news-and-events.html"]

    # ------------------------------------------------------------------
    # TATA DOMAIN / URL CONFIGURATION
    # ------------------------------------------------------------------

    TATA_ALLOWED_HOSTS = {
        "tata.cars",
        "www.tata.cars",
        "cars.tatamotors.com",
        "www.cars.tatamotors.com",
    }

    # Tata currently exposes passenger vehicle pages using structures such as:
    #
    #   /sierra/ice/request-a-call-back.html
    #   /harrier/ice/request-a-call-back.html
    #
    # The first path segment is treated as the model slug.

    TATA_ENERGY_TYPES = {
        "ice",
        "ev",
        "cng",
        "hybrid",
    }

    TATA_NON_PRODUCT_SEGMENTS = {
        "organisation",
        "investors",
        "csr",
        "newsroom",
        "articles",
        "accessories",
        "dealer-locator",
        "digital-showroom",
        "trade-in",
        "pre-owned",
        "rewards",
        "service",
        "support",
        "book-test-drive",
        "careers",
        "account",
        "finance",
        "contact",
        "privacy",
        "terms",
        "search",
        "sitemap",
    }

    EXCLUDED_PATH_PARTS = (
        "/content/dam/",
        "/news",
        "/news-and-events",
        "/privacy",
        "/terms",
        "/contact",
        "/dealer",
        "/dealers",
        "/finance",
        "/service",
        "/careers",
        "/about",
        "/sitemap",
        "/search",
        "/login",
        "/register",
        "/account",
        "/organisation",
        "/investors",
        "/csr",
        "/support",
        "/accessories",
    )

    NON_PRODUCT_EXTENSIONS = (
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".gif",
        ".svg",
        ".css",
        ".js",
        ".xml",
        ".json",
        ".ico",
    )
    EXCLUDED_IMAGE_TERMS = ("logo", "icon", "loader", "placeholder", "gif_")

    VISION_PROMPT = (
        "Analyze this vehicle image. "
        "Describe visible vehicle design, "
        "body style, exterior features, "
        "color, wheels, lights, badges, "
        "interior elements if visible, "
        "and any readable text. "
        "Return concise factual information "
        "useful for a vehicle knowledge RAG system."
    )

    def __init__(self) -> None:
        # Keep this compatible with your existing BaseLoader.
        super().__init__(self.SOURCE_URLS[0])

        self.logger = LOGGER

        self.run_id = str(uuid.uuid4())
        self.extraction_timestamp = datetime.now(timezone.utc).isoformat()

        self.image_store = PersistentImageStore(self.CATEGORY)

        self.embedder = Embedder()

        self.max_products = int(os.getenv("CARS_MAX_PRODUCTS", os.getenv("VEHICLE_MAX_PRODUCTS", "50")))

        self.max_images_per_product = int(
            os.getenv("CARS_MAX_IMAGES_PER_PRODUCT", os.getenv("VEHICLE_MAX_IMAGES_PER_PRODUCT", "8"))
        )

        self.enable_vision = (
            os.getenv("CARS_ENABLE_VISION", os.getenv("VEHICLE_ENABLE_VISION", "true")).lower() == "true"
        )

        self.vision_timeout = int(os.getenv("CARS_VISION_TIMEOUT_SECONDS", "120"))

        self.embed_timeout = int(os.getenv("CARS_EMBED_TIMEOUT_SECONDS", "120"))

        self.max_image_bytes = int(os.getenv("CARS_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))

        self.user_agent = os.getenv(
            "VEHICLE_USER_AGENT",
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/154.0.0.0 Safari/537.36"
            ),
        )

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------

    async def download_and_extract(self) -> List[Dict[str, Any]]:
        """
        Execute the complete cars ingestion pipeline.

        Returns:
            List of RAG-ready vehicle documents.

        No JSON is written.
        """

        self.logger.info("=" * 70)
        self.logger.info(
            "CarsLoader started | run_id=%s | brand=%s | category=%s", self.run_id, self.BRAND, self.CATEGORY
        )
        self.logger.info("=" * 70)

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)

            context = await browser.new_context(
                user_agent=self.user_agent,
                viewport={"width": 1440, "height": 900},
                locale="en-IN",
                timezone_id="Asia/Kolkata",
            )

            page = await context.new_page()

            try:
                await self._ollama_preflight()

                vehicle_urls = await self.discover_vehicle_urls(page)

                self.logger.info("Vehicle discovery complete | count=%d", len(vehicle_urls))

                if not vehicle_urls:
                    self.logger.error(
                        "ZERO VEHICLE URLS DISCOVERED. This is a discovery failure, not an empty catalogue."
                    )

                    await self._diagnose_source_page(page)

                    return []

                selected_urls = vehicle_urls[: self.max_products]

                self.logger.info("Product pages selected | count=%d | max=%d", len(selected_urls), self.max_products)

                documents: List[Dict[str, Any]] = []

                for index, vehicle_url in enumerate(selected_urls, start=1):
                    self.logger.info("[%d/%d] Processing vehicle | %s", index, len(selected_urls), vehicle_url)

                    try:
                        document = await self._process_vehicle_page(page, vehicle_url)

                        if document:
                            documents.append(document)

                            self.logger.info(
                                "[%d/%d] Vehicle processed | title=%s", index, len(selected_urls), document.get("title")
                            )

                    except Exception as exc:
                        self.logger.exception(
                            "[%d/%d] Vehicle failed | url=%s | error=%s", index, len(selected_urls), vehicle_url, exc
                        )

                self.logger.info("CarsLoader ingestion complete | documents=%d", len(documents))

                return documents

            finally:
                await context.close()
                await browser.close()

    # ------------------------------------------------------------------
    # DISCOVERY
    # ------------------------------------------------------------------

    async def discover_vehicle_urls(self, page: Page) -> List[str]:
        """
        Discover Tata vehicle detail pages.

        Important design decision:

        We DO NOT depend on a single CSS selector.

        Tata's website has changed its frontend structure over time.
        Therefore discovery uses anchors + URL heuristics + link text.
        """

        discovered: Dict[str, int] = {}

        for source_index, source_url in enumerate(self.SOURCE_URLS, start=1):
            self.logger.info(
                "[%d/%d] SOURCE | category=%s | brand=%s | url=%s",
                source_index,
                len(self.SOURCE_URLS),
                self.CATEGORY,
                self.BRAND,
                source_url,
            )

            try:
                response = await page.goto(source_url, wait_until="domcontentloaded", timeout=60_000)

                status = response.status if response is not None else None

                self.logger.info("Source response | url=%s | status=%s", source_url, status)

                if status and status >= 400:
                    self.logger.warning("Source returned HTTP %s | url=%s", status, source_url)

                await self._wait_for_dynamic_content(page)

                # Scroll to trigger lazy-loaded model cards.
                await self._scroll_page(page)

                links = await page.locator("a[href]").evaluate_all("anchors => anchors.map(a => ({href: a.href}))")

                self.logger.info("Raw anchor count | url=%s | count=%d", source_url, len(links))

                for link in links:
                    candidate = self._normalize_url(link.get("href") or "")

                    if not candidate:
                        continue

                    score = self._score_vehicle_url(candidate)

                    if score <= 0:
                        continue

                    previous = discovered.get(candidate, 0)

                    discovered[candidate] = max(previous, score)

                self.logger.info("Discovery candidates accumulated | total=%d", len(discovered))

            except Exception as exc:
                self.logger.exception("Source discovery failed | url=%s | error=%s", source_url, exc)

        # Highest confidence first.
        sorted_candidates = sorted(discovered.items(), key=lambda item: (-item[1], item[0]))

        self.logger.info("Scored candidates before validation | count=%d", len(sorted_candidates))

        # Validate candidates by visiting the actual page.
        validated: List[str] = []

        for url, score in sorted_candidates:
            if len(validated) >= self.max_products:
                break

            try:
                if await self._is_vehicle_detail_page(page, url):
                    validated.append(url)

                    self.logger.info("VALID VEHICLE | score=%d | url=%s", score, url)

            except Exception as exc:
                self.logger.warning("Candidate validation failed | url=%s | error=%s", url, exc)

        return validated

    # ------------------------------------------------------------------
    # URL NORMALIZATION
    # ------------------------------------------------------------------

    def _normalize_url(self, href: str,) -> Optional[str]:
        """
        Normalize and validate Tata URLs.

        Supports both the current Tata Cars domain and the
        older Tata Motors Cars domain.

        Examples accepted: https://tata.cars/sierra/ice/request-a-call-back.html

        Examples rejected:

            javascript:void(0)

            https://tata.cars/service/...

            https://tata.cars/news-and-events.html

            https://external-domain.com/...
        """

        if not href:
            return None

        href = href.strip()

        if href.startswith(("javascript:","mailto:","tel:","#",)):
            return None

        try:
            url = urljoin(
                self.SOURCE_URLS[0],
                href,
            )

            url, _fragment = urldefrag(url)

            parsed = urlparse(url)

        except Exception:
            return None

        if parsed.scheme not in {"http","https"}:
            return None

        host = parsed.netloc.lower().split(":")[0]

        if host not in self.TATA_ALLOWED_HOSTS:
            return None

        path = parsed.path.lower()

        if not path or path == "/":
            return None

        # Static assets are never vehicle pages.
        if path.endswith(self.NON_PRODUCT_EXTENSIONS):
            return None

        # Reject known non-product areas.
        for excluded in self.EXCLUDED_PATH_PARTS:
            if excluded in path:
                return None

        # Remove query parameters and fragments.example: https://tata.cars/nexon/ice/request-a-call-back.html
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        return clean_url.rstrip("/")
    # ------------------------------------------------------------------
    # URL SCORING
    # ------------------------------------------------------------------

    def _score_vehicle_url(self, url: str) -> int:
        """Score a URL by tata.cars structure: /{model}/{energy}.html (e.g. /nexon/ev.html) is a vehicle page.

        No model names are hard-coded: any first segment that is not a known non-product area
        counts as a model slug when the second segment is an energy type.
        """
        segments = urlparse(url).path.lower().removesuffix(".html").strip("/").split("/")

        if len(segments) < 2 or segments[0] in self.TATA_NON_PRODUCT_SEGMENTS or segments[1] not in self.TATA_ENERGY_TYPES:
            return 0

        rest = segments[2:]

        if not rest or rest == ["overview"]:  # /nexon/ev.html, /harrier/ev/overview.html
            return 10

        if "request-a-call-back" in rest:  # lead-capture form, not vehicle content
            return 0

        return 1  # edition / sub-pages such as /sierra/ice/edition/dark.html

    # ------------------------------------------------------------------
    # VEHICLE PAGE VALIDATION
    # ------------------------------------------------------------------

    async def _is_vehicle_detail_page(self, page: Page, url: str) -> bool:

        response = await page.goto(url, wait_until="domcontentloaded", timeout=45_000)

        status = response.status if response is not None else None

        if status and status >= 400:
            self.logger.warning("Candidate rejected due to HTTP status | status=%s | url=%s", status, url)
            return False

        await self._wait_for_dynamic_content(page)

        title = (await page.title()).strip()

        h1 = ""

        h1_locator = page.locator("h1").first

        if await h1_locator.count():
            try:
                h1 = (await h1_locator.inner_text()).strip()
            except Exception:
                pass

        body_text = ""

        try:
            body_text = (await page.locator("body").inner_text())[:20_000]
        except Exception:
            pass

        combined = (f"{title}\n{h1}\n{body_text}").lower()

        vehicle_signal = 0

        for keyword in (
            "engine",
            "transmission",
            "horsepower",
            "torque",
            "ground clearance",
            "wheelbase",
            "airbags",
            "boot space",
            "fuel",
            "mileage",
            "range",
            "suv",
            "hatchback",
            "sedan",
            "tata",
        ):
            if keyword in combined:
                vehicle_signal += 1

        # A product page should have a meaningful title/h1
        # and multiple vehicle signals.
        if not h1 and not title:
            return False

        return vehicle_signal >= 2

    # ------------------------------------------------------------------
    # PRODUCT PROCESSING
    # ------------------------------------------------------------------

    async def _process_vehicle_page(self, page: Page, vehicle_url: str) -> Optional[Dict[str, Any]]:

        response = await page.goto(vehicle_url, wait_until="domcontentloaded", timeout=60_000)

        status = response.status if response is not None else None

        if status and status >= 400:
            self.logger.warning("Vehicle page HTTP error | status=%s | url=%s", status, vehicle_url)
            return None

        await self._wait_for_dynamic_content(page)

        await self._scroll_page(page)

        title = await self._extract_title(page)

        if not title:
            self.logger.warning("Vehicle page has no title | url=%s", vehicle_url)
            return None

        description = await self._extract_description(page)

        json_ld = await self._extract_json_ld(page)

        specs = await self._extract_specs(page)

        features = await self._extract_features(page)

        colors = await self._extract_colors(page, json_ld)

        image_urls = await self._extract_images(page)

        brochure_urls = await self._extract_brochures(page)

        page_text = await self._extract_page_text(page)

        price = await self._extract_price(page)

        rag_text = self._build_rag_text(
            title=title,
            description=description,
            price=price,
            specs=specs,
            features=features,
            colors=colors,
            page_text=page_text,
            json_ld=json_ld,
        )

        image_results: List[Dict[str, Any]] = []

        if image_urls and self.enable_vision:
            image_results = await self._process_images(image_urls=image_urls, vehicle_url=vehicle_url, title=title)

        text_embedding = await self._ollama_embed(rag_text)

        document = {
            "run_id": self.run_id,
            "brand": self.BRAND,
            "vehicle_type": self.CATEGORY,
            "title": title,
            "source_url": vehicle_url,
            "product_url": vehicle_url,
            "brochure_urls": brochure_urls,
            "description": description,
            "price": price,
            "specs": specs,
            "features": features,
            "colors": colors,
            "image_urls": image_urls,
            "images": image_results,
            "text": rag_text,
            "text_embedding": text_embedding,
            "json_ld": json_ld,
            "extraction_timestamp": self.extraction_timestamp,
        }

        return document

    # ------------------------------------------------------------------
    # EXTRACTION
    # ------------------------------------------------------------------

    async def _extract_title(self, page: Page) -> str:
        """Model name. On tata.cars the page <title> is reliable ("Tata Altroz - Price, Features ... | TATA.CARS",
        "Nexon EV"); the first <h1> is often a site-wide banner ("Explore the Complete Tata Cars Range")."""
        title = re.split(r"\s+[|\-–]\s+", re.sub(r"\s+", " ", await page.title()).strip())[0]
        title = re.sub(r"^tata\s+", "", title, flags=re.I).strip()  # brand is stored separately
        if title:
            return title
        try:
            return re.sub(r"\s+", " ", await page.locator("h1").first.inner_text(timeout=3000)).strip()
        except Exception:
            return ""

    async def _extract_description(self, page: Page) -> str:

        selectors = ["meta[name='description']", "meta[property='og:description']"]

        for selector in selectors:
            locator = page.locator(selector).first

            if not await locator.count():
                continue

            value = await locator.get_attribute("content")

            if value:
                return re.sub(r"\s+", " ", value).strip()

        return ""

    async def _extract_specs(self, page: Page) -> Dict[str, str]:

        specs: Dict[str, str] = {}

        rows = await page.locator("table tr").all()

        for row in rows:
            try:
                cells = await row.locator("th, td").all_inner_texts()

                cells = [re.sub(r"\s+", " ", cell).strip() for cell in cells]

                cells = [cell for cell in cells if cell]

                if len(cells) >= 2:
                    key = cells[0]
                    value = " | ".join(cells[1:])

                    if key and value:
                        specs[key] = value

            except Exception:
                continue

        return specs

    async def _extract_features(self, page: Page) -> List[str]:

        selectors = ["[class*='feature']", "[class*='Feature']", "[data-testid*='feature']"]

        results: List[str] = []

        for selector in selectors:
            try:
                texts = await page.locator(selector).all_inner_texts()

                for text in texts:
                    text = re.sub(r"\s+", " ", text).strip()

                    if text and len(text) > 2 and len(text) < 500:
                        results.append(text)

            except Exception:
                continue

        return self._dedupe(results)[:200]

    async def _extract_colors(self, page: Page, json_ld: List[Dict[str, Any]]) -> List[str]:

        colors: List[str] = []

        for item in json_ld:
            if not isinstance(item, dict):
                continue

            value = item.get("color")

            if isinstance(value, str):
                colors.append(value)

            elif isinstance(value, list):
                colors.extend(str(v) for v in value)

        selectors = ["[class*='color']", "[class*='Color']", "[class*='colour']", "[class*='Colour']"]

        for selector in selectors:
            try:
                texts = await page.locator(selector).all_inner_texts()

                for text in texts:
                    text = re.sub(r"\s+", " ", text).strip()

                    if text and len(text) < 200:
                        colors.append(text)

            except Exception:
                continue

        return self._dedupe(colors)[:100]

    async def _extract_images(self, page: Page) -> List[str]:
        # Lazy images keep a placeholder (e.g. gif_100x100) in src until scrolled into view; the real
        # URL is in data-src. naturalWidth describes whatever is loaded, so it is only trusted without data-src.
        images = await page.locator("img").evaluate_all(
            """
            imgs => imgs.map(img => {
                const lazy = img.getAttribute("data-src") || img.getAttribute("data-lazy-src");
                return {
                    src: lazy || img.currentSrc || img.src,
                    alt: img.alt || "",
                    width: lazy ? 0 : img.naturalWidth,
                    height: lazy ? 0 : img.naturalHeight
                };
            })
            """
        )

        results: List[str] = []

        for image in images:
            if not image.get("src") or not self._is_useful_image(image):
                continue

            src, _fragment = urldefrag(urljoin(page.url, image["src"]))
            parsed = urlparse(src)

            # The vision model cannot read SVG; GIFs on tata.cars are placeholders/spinners.
            if parsed.scheme not in ("http", "https") or parsed.path.lower().endswith((".svg", ".gif")):
                continue

            results.append(src)

        return self._dedupe(self._model_images(results, page.url))[: self.max_images_per_product]

    @staticmethod
    def _model_images(image_urls: List[str], page_url: str) -> List[str]:
        """Images named after the page's model (/altroz/ice.html -> *altroz*), else all of them.

        Every tata.cars page starts with the site-wide model menu (aeris, curvv, ...), which would
        otherwise fill the per-product image quota with other cars.
        """
        # ponytail: filename heuristic; scope to the page's main content if model images stop carrying the slug.
        slug = urlparse(page_url).path.strip("/").split("/")[0].lower()
        own = [url for url in image_urls if slug and slug in urlparse(url).path.lower()]
        return own or image_urls

    async def _extract_brochures(self, page: Page) -> List[str]:

        hrefs = await page.locator("a[href]").evaluate_all(
            """
            anchors => anchors.map(a => a.href)
            """
        )

        results = []

        for href in hrefs:
            if not href:
                continue

            if ".pdf" not in href.lower():
                continue

            href = urljoin(page.url, href)

            results.append(urldefrag(href)[0])

        return self._dedupe(results)

    async def _extract_price(self, page: Page) -> Optional[str]:

        selectors = ["[class*='price']", "[class*='Price']", "[data-testid*='price']"]

        for selector in selectors:
            try:
                texts = await page.locator(selector).all_inner_texts()

                for text in texts:
                    text = re.sub(r"\s+", " ", text).strip()

                    if "₹" in text or "rs." in text.lower() or "lakh" in text.lower():
                        return text[:300]

            except Exception:
                continue

        return None

    async def _extract_page_text(self, page: Page) -> str:

        text = await page.locator("body").inner_text()

        text = re.sub(r"\n{3,}", "\n\n", text)

        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    # ------------------------------------------------------------------
    # RAG TEXT
    # ------------------------------------------------------------------

    def _build_rag_text(
        self,
        title: str,
        description: str,
        price: Optional[str],
        specs: Dict[str, str],
        features: List[str],
        colors: List[str],
        page_text: str,
        json_ld: List[Dict[str, Any]],
    ) -> str:

        sections: List[str] = []

        sections.append(f"Brand: {self.BRAND}")

        sections.append(f"Vehicle Type: {self.CATEGORY}")

        sections.append(f"Vehicle: {title}")

        if description:
            sections.append(f"Description:\n{description}")

        if price:
            sections.append(f"Price:\n{price}")

        if colors:
            sections.append("Colors:\n" + "\n".join(f"- {color}" for color in colors))

        if specs:
            sections.append(
                "Technical Specifications:\n" + "\n".join(f"- {key}: {value}" for key, value in specs.items())
            )

        if features:
            sections.append("Features:\n" + "\n".join(f"- {feature}" for feature in features))

        if json_ld:
            sections.append("Structured Data:\n" + json.dumps(json_ld, ensure_ascii=False, indent=2)[:20_000])

        sections.append("Official Page Content:\n" + page_text)

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # IMAGE PIPELINE
    # ------------------------------------------------------------------

    async def _process_images(self, image_urls: List[str], vehicle_url: str, title: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        connector = aiohttp.TCPConnector(limit=4, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            with tempfile.TemporaryDirectory(prefix="vehicle_image_") as tmpdir:
                for image_url in image_urls:
                    record = await self._image_record(
                        session, image_url, tmpdir, self.max_image_bytes, vehicle_url=vehicle_url, vehicle_title=title
                    )
                    if record:
                        results.append(record)
        return results

    # ------------------------------------------------------------------
    # PAGE UTILITIES
    # ------------------------------------------------------------------

    async def _wait_for_dynamic_content(self, page: Page) -> None:

        try:
            await page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass

        await page.wait_for_timeout(2_000)

    async def _scroll_page(self, page: Page) -> None:

        try:
            await page.evaluate(
                """
                async () => {
                    await new Promise(resolve => {
                        let total = 0;
                        const distance = 700;
                        const timer = setInterval(() => {
                            window.scrollBy(0, distance);
                            total += distance;

                            if (
                                total >=
                                document.body.scrollHeight
                            ) {
                                clearInterval(timer);
                                resolve();
                            }
                        }, 200);
                    });
                }
                """
            )

            await page.wait_for_timeout(1500)

        except Exception as exc:
            self.logger.debug("Scroll failed | error=%s", exc)

    async def _diagnose_source_page(self, page: Page) -> None:
        """
        Diagnostic information specifically for
        discovering why zero URLs occurred.
        """

        try:
            title = await page.title()

            body_text = await page.locator("body").inner_text()

            links = await page.locator("a[href]").count()

            self.logger.error(
                "DISCOVERY DIAGNOSTICS | page=%s | title=%s | anchors=%d | body_chars=%d",
                page.url,
                title,
                links,
                len(body_text),
            )

            self.logger.error("First 1000 body chars:\n%s", body_text[:1000])

            hrefs = await page.locator("a[href]").evaluate_all(
                """
                anchors => anchors
                    .slice(0, 50)
                    .map(a => ({
                        href: a.href,
                        text: (
                            a.innerText ||
                            a.textContent ||
                            ""
                        ).trim()
                    }))
                """
            )

            for item in hrefs:
                self.logger.error("DISCOVERY LINK | text=%s | href=%s", item.get("text"), item.get("href"))

        except Exception as exc:
            self.logger.exception("Discovery diagnostics failed | error=%s", exc)

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urldefrag, urlparse

import aiohttp
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from config.logger import setup_logger
from ingestion.embedding.embedder import DEFAULT_VISION_PROMPT

logger = setup_logger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36"
)


class BaseLoader:
    """Shared crawl + Ollama helpers.

    Subclasses using the Ollama/image helpers set: embedder, image_store,
    vision_timeout, embed_timeout, enable_vision (and optionally VISION_PROMPT).
    Subclasses using ``download_and_extract`` also set SOURCES and max_products and implement
    ``discover_product_links``, ``extract_product_page`` and ``build_rag_document``.
    """

    CATEGORY = "vehicle"
    SOURCES: List[Dict[str, Any]] = []
    VISION_PROMPT = DEFAULT_VISION_PROMPT
    EXCLUDED_IMAGE_TERMS: tuple[str, ...] = ()
    MIN_IMAGE_SIZE = (120, 120)  # (width, height); images with unknown size pass
    NAVIGATION_TITLES = frozenset(
        {
            "home",
            "about us",
            "contact",
            "contact us",
            "privacy policy",
            "terms and conditions",
            "login",
            "register",
            "shopping cart",
        }
    )
    NAV_TEXT_LIMIT = 3000  # a longer page with none of PAGE_SIGNALS is treated as a landing page
    PAGE_SIGNALS: tuple[str, ...] = ()

    def __init__(self, base_url: str):
        self.base_url = base_url

    # -- common helpers -------------------------------------------------------
    @staticmethod
    def _headers() -> Dict[str, str]:
        return {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Remove fragments and normalize a URL enough for deduplication."""
        clean, _fragment = urldefrag((url or "").strip())
        return clean.rstrip("/") or clean

    @staticmethod
    def _same_host(url_a: str, url_b: str) -> bool:
        return urlparse(url_a).netloc.lower() == urlparse(url_b).netloc.lower()

    @staticmethod
    def _safe_filename(value: str, default: str = "image") -> str:
        value = os.path.basename(value).split("?")[0].strip()
        if not value:
            return default
        # Keep extension but remove characters unsafe on Windows.
        name, ext = os.path.splitext(value)
        safe = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in name)
        ext = "".join(char if char.isalnum() or char == "." else "_" for char in ext)
        return f"{safe[:100] or default}{ext}"

    @staticmethod
    def _dedupe(values: List[str]) -> List[str]:
        """Whitespace-normalised, case-insensitive dedupe that keeps the first spelling."""
        seen: set[str] = set()
        result: List[str] = []
        for value in values:
            normalized = " ".join(str(value).split())
            if normalized and normalized.lower() not in seen:
                seen.add(normalized.lower())
                result.append(normalized)
        return result

    @staticmethod
    def _fallback_name(product_url: str, default: str = "unknown-vehicle") -> str:
        slug = urlparse(product_url).path.rstrip("/").split("/")[-1] or default
        return slug.replace("-", " ").replace("_", " ").strip().title()

    def _is_useful_image(self, image: Dict[str, Any]) -> bool:
        """Reject logos, UI icons and tiny assets."""
        src, alt = str(image.get("src") or "").lower(), str(image.get("alt") or "").lower()
        if any(term in src or term in alt for term in self.EXCLUDED_IMAGE_TERMS):
            return False
        width, height = int(image.get("width") or 0), int(image.get("height") or 0)
        min_width, min_height = self.MIN_IMAGE_SIZE
        return not (width and height and (width < min_width or height < min_height))

    def _looks_like_navigation_page(self, title: str, page_text: str) -> bool:
        """Conservative guard against indexing home/category/navigation pages."""
        if (title or "").strip().lower() in self.NAVIGATION_TITLES:
            return True
        text = (page_text or "").strip().lower()
        return len(text) > self.NAV_TEXT_LIMIT and not any(signal in text for signal in self.PAGE_SIGNALS)

    async def _extract_json_ld(self, page: Page) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        for raw in await page.locator('script[type="application/ld+json"]').all_text_contents():
            try:
                parsed = json.loads(raw)
            except ValueError:
                continue
            items = parsed if isinstance(parsed, list) else [parsed]
            documents.extend(item for item in items if isinstance(item, dict))
        return documents

    # -- playwright -----------------------------------------------------------
    async def _create_context(self, browser: Browser) -> BrowserContext:
        context = await browser.new_context(
            user_agent=USER_AGENT, locale="en-IN", viewport={"width": 1440, "height": 1000}
        )
        await context.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})
        return context

    async def _create_page(self, context: BrowserContext) -> Page:
        return await context.new_page()

    async def _safe_close_page(self, page: Optional[Page]) -> None:
        if page is None:
            return
        try:
            await page.close()
        except Exception as exc:
            logger.debug("Page close skipped: %s", exc)

    async def _safe_close_context(self, context: Optional[BrowserContext]) -> None:
        if context is None:
            return
        try:
            await context.close()
        except Exception as exc:
            # Closing an already-disconnected Playwright target must not turn
            # a successful/partial crawl into a fatal ingestion error.
            logger.warning("Browser context close skipped: %s", exc)

    async def _safe_close_browser(self, browser: Optional[Browser]) -> None:
        if browser is None:
            return
        try:
            await browser.close()
        except Exception as exc:
            logger.warning("Browser close skipped: %s", exc)

    async def _scroll_page(self, page: Page, max_scrolls: int = 20, wait_ms: int = 800) -> None:
        """Load lazy content without assuming a fixed product count."""
        previous_height = 0
        for _ in range(max_scrolls):
            try:
                current_height = await page.evaluate("document.body ? document.body.scrollHeight : 0")
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(wait_ms)
                new_height = await page.evaluate("document.body ? document.body.scrollHeight : 0")
                if new_height == previous_height:
                    break
                previous_height = max(previous_height, current_height)
            except Exception:
                logger.exception("Page scrolling failed")
                break

    # -- images ---------------------------------------------------------------
    async def download_image(
        self, session: aiohttp.ClientSession, image_url: str, directory: str, max_bytes: Optional[int] = None
    ) -> Optional[str]:
        """Stream the image to ``directory``, then retain it in durable category storage."""
        try:
            filename = self._safe_filename(os.path.basename(urlparse(image_url).path))
            # Same basename can occur on multiple product pages; a URL hash prevents overwrites.
            digest = hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:12]
            name, ext = os.path.splitext(filename)
            filepath = os.path.join(directory, f"{name[:70]}_{digest}{ext or '.jpg'}")
            async with session.get(
                image_url, headers=self._headers(), timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response.raise_for_status()
                if max_bytes and int(response.headers.get("Content-Length") or 0) > max_bytes:
                    logger.warning("Image skipped, too large | url=%s", image_url)
                    return None
                total = 0
                with open(filepath, "wb") as file:
                    while chunk := await response.content.read(64 * 1024):
                        total += len(chunk)
                        if max_bytes and total > max_bytes:
                            logger.warning("Image exceeded max size while streaming | url=%s", image_url)
                            return None
                        file.write(chunk)
            return str(self.image_store.persist(filepath, image_url))
        except Exception:
            logger.exception("Image download failed | %s", image_url)
            return None

    # -- ollama (via Embedder, off the event loop) -----------------------------
    async def _ollama_preflight(self) -> bool:
        """False if Ollama is unreachable or the embedding model is missing; disables vision if its model is."""
        try:
            installed = await asyncio.to_thread(self.embedder.list_models)
        except Exception:
            logger.exception("Ollama preflight failed | host=%s", self.embedder.ollama_host)
            return False

        def has(model: str) -> bool:
            return any(name.split(":")[0] == model.split(":")[0] for name in installed)

        embed_ok, vision_ok = has(self.embedder.embed_model), has(self.embedder.vision_model)
        logger.info(
            "Ollama preflight | embed=%s [%s] | vision=%s [%s]",
            self.embedder.embed_model,
            embed_ok,
            self.embedder.vision_model,
            vision_ok,
        )
        if not embed_ok:
            logger.error("Embedding model not installed: %s", self.embedder.embed_model)
            return False
        if self.enable_vision and not vision_ok:
            logger.warning("Vision model not installed: %s. Continuing without vision.", self.embedder.vision_model)
            self.enable_vision = False
        return True

    async def _ollama_vision(self, image_path: str) -> str:
        try:
            result = await asyncio.to_thread(
                self.embedder.describe_image, image_path, self.vision_timeout, self.VISION_PROMPT
            )
            logger.info("VISION DONE | image=%s | chars=%d", os.path.basename(image_path), len(result))
            return result
        except Exception:
            logger.exception("Image description failed | %s", image_path)
            return ""

    async def _ollama_embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            return []
        try:
            return list(await asyncio.to_thread(self.embedder.embed_text, text, self.embed_timeout))
        except Exception:
            logger.exception("Text embedding failed")
            return []

    async def process_image(self, image_path: str) -> Dict[str, Any]:
        result = {"caption": "", "ocr_text": "", "embedding": []}
        if not self.enable_vision:
            return result
        result["caption"] = await self._ollama_vision(image_path)
        if result["caption"]:
            result["embedding"] = await self._ollama_embed(result["caption"])
        return result

    async def _image_record(
        self,
        session: aiohttp.ClientSession,
        image_url: str,
        directory: str,
        max_bytes: Optional[int] = None,
        **extra: Any,
    ) -> Optional[Dict[str, Any]]:
        """Download + describe one image; ``None`` if the download failed."""
        path = await self.download_image(session, image_url, directory, max_bytes)
        if not path:
            return None
        return {"image_url": image_url, "local_path": path, **extra, **await self.process_image(path)}

    # -- crawl pipeline -------------------------------------------------------
    async def download_and_extract(self) -> List[Dict[str, Any]]:
        """One Playwright lifecycle per run. A failing source, product or image is logged and skipped;
        shutdown errors never mask the documents already built."""
        results: List[Dict[str, Any]] = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            with tempfile.TemporaryDirectory(prefix=f"{self.CATEGORY}_") as tmpdir:
                playwright = browser = context = page = None
                try:
                    if not await self._ollama_preflight():
                        return []
                    playwright = await async_playwright().start()
                    browser = await playwright.chromium.launch(headless=True)
                    context = await self._create_context(browser)
                    page = await self._create_page(context)
                    for source_index, source in enumerate(self.SOURCES, start=1):
                        logger.info(
                            "[%d/%d] SOURCE | category=%s | brand=%s | url=%s",
                            source_index,
                            len(self.SOURCES),
                            self.CATEGORY,
                            source["brand"],
                            source["url"],
                        )
                        product_links = (await self.discover_product_links(page, source))[: self.max_products]
                        logger.info("Product pages selected | count=%d", len(product_links))
                        for index, product_url in enumerate(product_links, start=1):
                            logger.info("[%d/%d] PRODUCT | %s", index, len(product_links), product_url)
                            try:
                                product = await self.extract_product_page(page, product_url, source)
                                if not product:
                                    continue
                                image_results = []
                                for image_index, image in enumerate(product["images"], start=1):
                                    record = await self._image_record(
                                        session, image["url"], tmpdir, alt=image.get("alt", ""), image_index=image_index
                                    )
                                    if record:
                                        image_results.append(record)
                                document = await self.build_rag_document(product, image_results)
                                results.append(document)
                                logger.info(
                                    "DOCUMENT CREATED | category=%s | url=%s | images=%d | dims=%d",
                                    self.CATEGORY,
                                    product_url,
                                    len(image_results),
                                    len(document["text_embedding"]),
                                )
                            except Exception:
                                logger.exception("Product processing failed | %s", product_url)
                except Exception:
                    logger.exception("%s pipeline failed", self.__class__.__name__)
                finally:
                    await self._safe_close_page(page)
                    await self._safe_close_context(context)
                    await self._safe_close_browser(browser)
                    if playwright is not None:
                        try:
                            await playwright.stop()
                        except Exception as exc:
                            logger.warning("Playwright stop skipped: %s", exc)
        logger.info("%s ingestion complete | documents=%d", self.__class__.__name__, len(results))
        return results

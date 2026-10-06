# import sys
# import asyncio
# import aiohttp
# from contextlib import nullcontext
# from typing import List, Dict, Any
# from playwright.async_api import async_playwright
# from PIL import Image
# import torch
# import os
# from transformers import BlipProcessor, BlipForConditionalGeneration
# import pytesseract

# from ingestion.loader.base_loader import BaseLoader
# from config.logger import setup_logger
# from ingestion.embedding.embedder import Embedder
# from ingestion.images import PersistentImageStore

# # Point pytesseract to the local installation
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# # Ensure console prints UTF-8 (fixes rupee symbol crash on Windows)
# sys.stdout.reconfigure(encoding='utf-8')

# logger = setup_logger(__name__)

# # --- BLIP captioning ---
# device = "cuda" if torch.cuda.is_available() else "cpu"
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# # --- Embedder instance for text + image embeddings ---
# embedder = Embedder()

# async def async_download(session, url, filepath):
#     """Download a file asynchronously."""
#     try:
#         async with session.get(url, timeout=60) as resp:
#             resp.raise_for_status()
#             with open(filepath, "wb") as f:
#                 f.write(await resp.read())
#         return filepath
#     except Exception as e:
#         logger.error("Download failed: %s", e)
#         return None

# def generate_caption(image_path: str) -> str:
#     """Generate a caption for an image using BLIP."""
#     try:
#         raw_image = Image.open(image_path).convert("RGB")
#         inputs = blip_processor(raw_image, return_tensors="pt").to(device)
#         out = blip_model.generate(**inputs)
#         return blip_processor.decode(out[0], skip_special_tokens=True)
#     except Exception as e:
#         logger.error("Captioning failed for %s: %s", image_path, e)
#         return "Uncaptioned image"

# def run_ocr(image_path: str) -> str:
#     """Run OCR on an image using Tesseract."""
#     try:
#         return pytesseract.image_to_string(Image.open(image_path)).strip()
#     except Exception as e:
#         logger.error("OCR failed for %s: %s", image_path, e)
#         return ""

# def get_image_embedding(image_path: str) -> List[float]:
#     """Generate an embedding for an image via Embedder (LLaVA + text embed)."""
#     try:
#         return embedder.embed_image(image_path)
#     except Exception as e:
#         logger.error("Image embedding failed for %s: %s", image_path, e)
#         return []

# class ScootersLoader(BaseLoader):
#     BASE_URL = "https://www.heromotocorp.com/en-in/scooters.html"

#     def __init__(self):
#         super().__init__(self.BASE_URL)
#         self.max_products = max(1, int(os.getenv("SCOOTERS_MAX_PRODUCTS", "5")))
#         self.max_images_per_product = max(0, int(os.getenv("SCOOTERS_MAX_IMAGES_PER_PRODUCT", "2")))
#         self.image_store = PersistentImageStore("scooters")
#         logger.info("Scooters loader limits | products=%d | images_per_product=%d",
#                     self.max_products, self.max_images_per_product)

#     async def discover_products(self) -> List[Dict[str, Any]]:
#         products = []
#         async with async_playwright() as p:
#             browser = await p.chromium.launch(headless=True)
#             page = await browser.new_page()
#             await page.goto(self.base_url, wait_until="domcontentloaded", timeout=60000)
#             await page.wait_for_timeout(3000)

#             # Scroll to load dynamic content
#             for _ in range(10):
#                 prev_height = await page.evaluate("document.body.scrollHeight")
#                 await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
#                 await page.wait_for_timeout(1000)
#                 new_height = await page.evaluate("document.body.scrollHeight")
#                 if new_height == prev_height:
#                     break

#             # Extract product cards
#             cards = await page.locator("div.product-card").evaluate_all(
#                 "els => els.map(e => ({"
#                 "title: e.querySelector('.product-card-title')?.innerText,"
#                 "features: Array.from(e.querySelectorAll('.product-card-feature-item')).map(f => f.innerText),"
#                 "price: e.querySelector('.product-card-price-value')?.innerText,"
#                 "img: e.querySelector('img')?.src,"
#                 "link: e.querySelector('a.product-card-action-btn')?.href"
#                 "}))"
#             )
#             products.extend(cards)
#             await browser.close()
#         logger.info("Discovered %d scooter products", len(products))
#         return products

#     async def download_and_extract(self) -> List[Dict[str, Any]]:
#         results = []
#         products = (await self.discover_products())[:self.max_products]
#         if not products:
#             logger.warning("No scooter products found at %s", self.base_url)
#             return []

#         async with aiohttp.ClientSession() as session:
#             for prod in products:
#                 img_url = prod.get("img")
#                 img_path = None
#                 image_embedding = []
#                 if self.max_images_per_product and img_url:
#                     filename = str(self.image_store.destination_for(img_url))
#                     # Deduplicate: skip download if already exists
#                     if not os.path.exists(filename):
#                         img_path = await async_download(session, img_url, filename)
#                     else:
#                         img_path = filename
#                     if img_path:
#                         image_embedding = get_image_embedding(img_path)

#                 caption = generate_caption(img_path) if img_path else None
#                 ocr_text = run_ocr(img_path) if img_path else None

#                 text_to_embed = f"{prod.get('title')} | {prod.get('price')} | Features: {', '.join(prod.get('features') or [])}"
#                 try:
#                     text_embedding = embedder.embed_text(text_to_embed)
#                 except Exception as e:
#                     logger.error("Text embedding failed for %s: %s", prod.get("title"), e)
#                     text_embedding = []

#                 results.append({
#                     "brand": "Hero",
#                     "category": "scooter",
#                     "model": prod.get("title"),
#                     "features": prod.get("features"),
#                     "price": prod.get("price"),
#                     "image_url": img_url,
#                     "local_image": img_path,
#                     "caption": caption,
#                     "ocr_text": ocr_text,
#                     "source_url": prod.get("link"),
#                     "image_embeddings": image_embedding or [],
#                     "images": ([{
#                         "image_url": img_url,
#                         "local_path": img_path,
#                         "caption": caption or "",
#                         "ocr_text": ocr_text or "",
#                         "embedding": image_embedding,
#                     }] if img_path else []),
#                     "text": text_to_embed,
#                     "text_embedding": text_embedding or []
#                 })
#         logger.info("Scooters loader finished | products=%d", len(results))
#         return results

# if __name__ == "__main__":
#     loader = ScootersLoader()
#     data = asyncio.run(loader.download_and_extract())
#     print(f"Scraped {len(data)} scooter products")
#     for d in data:
#         print(d["model"], d["price"], d["features"], d["image_url"], len(d.get("image_embeddings", [])))
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from ingestion.embedding.embedder import Embedder
from ingestion.loader.generic_loader import (
    GenericVehicleLoader,
)


class ScootersLoader(GenericVehicleLoader):
    """
    Hero MotoCorp scooter loader.

    Scooter model names are discovered dynamically.
    """

    CATEGORY = "scooters"

    SOURCES: List[Dict[str, Any]] = [
        {
            "brand": "hero",
            "url": (
                "https://www.heromotocorp.com/"
                "en-in/scooters.html"
            ),
            "vehicle_type": "scooter",
        }
    ]

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
    ):
        super().__init__(
            embedder=embedder
        )


async def main() -> None:

    loader = ScootersLoader()

    documents = (
        await loader.download_and_extract()
    )

    print(
        f"\nScooter RAG documents: "
        f"{len(documents)}"
    )

    for document in documents:

        metadata = document[
            "metadata"
        ]

        print(
            "\n----------------------------------------"
        )

        print(
            "Vehicle:",
            metadata.get(
                "vehicle_name"
            ),
        )

        print(
            "Brand:",
            metadata.get(
                "brand"
            ),
        )

        print(
            "Type:",
            metadata.get(
                "vehicle_type"
            ),
        )

        print(
            "Price:",
            metadata.get(
                "price"
            ),
        )

        print(
            "Images:",
            len(
                document.get(
                    "images",
                    [],
                )
            ),
        )

        print(
            "Embedding dimensions:",
            len(
                document.get(
                    "text_embedding",
                    [],
                )
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
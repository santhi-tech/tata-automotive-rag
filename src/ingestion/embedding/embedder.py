from __future__ import annotations

import base64
import io
from typing import Any, List

import requests
from PIL import Image

from config.logger import setup_logger
from config.settings import get

logger = setup_logger(__name__)

DEFAULT_VISION_PROMPT = (
    "Describe this vehicle image in detail. Identify the vehicle type, visible model characteristics, "
    "exterior design, colors, visible features, wheels, lights, interior elements, text, badges, and other "
    "useful visual information. Do not invent details that cannot be seen."
)


def _jpeg_base64(image_path: str) -> str:
    """Re-encode as JPEG. CDNs serve AVIF/WebP under .jpg URLs, and Ollama's decoder rejects them (HTTP 400)."""
    with Image.open(image_path) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class Embedder:
    """Thin synchronous Ollama client: text embeddings, generation and image description."""

    def __init__(self, gen_model: str | None = None, embed_model: str | None = None, vision_model: str | None = None):
        self.gen_model = gen_model or get("GENERATION_MODEL")
        self.embed_model = embed_model or get("EMBEDDING_MODEL", "nomic-embed-text:latest")
        self.vision_model = vision_model or get("VISION_MODEL", "llava:latest")
        # Caps the caption length. On CPU, llava writes ~7 tokens/s; an uncapped ~370-token
        # caption takes ~60s and blows the loaders' vision timeouts. 120 tokens ~ 15s.
        self.vision_max_tokens = int(get("VISION_MAX_TOKENS", "120"))
        host = get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/").removesuffix("/api")
        self.ollama_host = host.replace("localhost", "127.0.0.1")
        logger.info("Ollama host=%s | gen=%s | embed=%s | vision=%s",
                    self.ollama_host, self.gen_model, self.embed_model, self.vision_model)

    def _post(self, endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        response = requests.post(f"{self.ollama_host}/api/{endpoint}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def list_models(self, timeout: float = 5) -> set[str]:
        response = requests.get(f"{self.ollama_host}/api/tags", timeout=timeout)
        response.raise_for_status()
        return {str(item.get("name", "")) for item in response.json().get("models", [])}

    def embed_text(self, text: str, timeout: float = 60) -> List[float]:
        if not text or not text.strip():
            logger.warning("Empty text received for embedding")
            return []
        data = self._post("embed", {"model": self.embed_model, "input": text}, timeout)
        if not data.get("embeddings"):
            raise RuntimeError(f"Unexpected embedding response: {data}")
        return data["embeddings"][0]

    def generate_text(self, prompt: str, timeout: float = 120) -> str:
        if not prompt or not prompt.strip():
            return ""
        data = self._post("generate", {"model": self.gen_model, "prompt": prompt, "stream": False}, timeout)
        if data.get("response") is None:
            raise RuntimeError(f"Unexpected Ollama response: {data}")
        return data["response"]

    def describe_image(self, image_path: str, timeout: float = 120, prompt: str = DEFAULT_VISION_PROMPT) -> str:
        payload = {
            "model": self.vision_model, "prompt": prompt, "images": [_jpeg_base64(image_path)], "stream": False,
            "options": {"num_predict": self.vision_max_tokens},
        }
        data = self._post("generate", payload, timeout)
        if data.get("response") is None:
            raise RuntimeError(f"Unexpected vision response: {data}")
        return data["response"].strip()

    def embed_image(self, image_path: str, timeout: float = 120) -> List[float]:
        """Image -> vision description -> text embedding."""
        description = self.describe_image(image_path, timeout=timeout)
        if not description:
            logger.warning("Empty image description | file=%s", image_path)
            return []
        logger.info("Image description generated | file=%s | chars=%d", image_path, len(description))
        return self.embed_text(description, timeout=timeout)

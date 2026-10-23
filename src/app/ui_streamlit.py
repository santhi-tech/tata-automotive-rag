from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config.logger import setup_logger
from retrieval.retriever import SUPPORTED_LANGUAGES, own_image, parse_metadata, product_name, retrieve_with_sources

logger = setup_logger(__name__)


def row_value(row: Any, field: str, default: Any = None) -> Any:
    return row[field] if field in row.keys() else default


def render_sources(rows: list[Any]) -> None:
    if not rows:
        return
    st.markdown("**Sources**")
    for row in rows:
        metadata = parse_metadata(row_value(row, "metadata", {}))
        name = product_name(metadata) or row_value(row, "chunk_id", "")
        url = metadata.get("product_url") or metadata.get("source_url")
        st.markdown(f"- [{name}]({url})" if url else f"- {name}")


def render_product_cards(rows: list[Any]) -> None:
    if not rows:
        return
    cols = st.columns(min(len(rows), 3))
    for i, row in enumerate(rows):
        metadata = parse_metadata(row_value(row, "metadata", {}))
        name = product_name(metadata) or row_value(row, "chunk_id", "")
        with cols[i % len(cols)]:
            image = own_image(metadata)
            if image:
                st.image(image, caption=str(name))
            else:
                st.markdown(f"**{name}**")
            if metadata.get("product_url"):
                st.link_button("Open product page", metadata["product_url"])


st.set_page_config(page_title="Tata Automotive RAG", page_icon="T", layout="wide")
st.title("Tata Automotive RAG")
st.caption("Categories: cars | motorcycles | scooters | spare_parts")

with st.sidebar:
    st.header("Search settings")
    language = st.selectbox("Response language", list(SUPPORTED_LANGUAGES))
    top_k = st.slider("Results", min_value=1, max_value=10, value=5)

query = st.text_input("Ask about vehicles, features, parts, or specifications")
search_clicked = st.button("Search", type="primary")

if search_clicked:
    if not query.strip():
        st.warning("Enter a question before searching.")
    else:
        logger.info("UI search started | query=%r", query)
        try:
            with st.spinner("Searching the indexed knowledge base..."):
                answer, rows = asyncio.run(retrieve_with_sources(query, top_k=top_k, language=language))
                st.subheader("Answer")
                st.markdown(answer)
                render_sources(rows)
                render_product_cards(rows)
            logger.info("UI search completed | results=%d", len(rows))
        except TimeoutError:
            logger.exception("UI search timed out")
            st.error("The local Ollama model did not respond in time. Check that it is running, then try again.")
        except Exception as exc:
            logger.exception("UI search failed")
            st.error(f"Retrieval failed: {exc}")

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
from retrieval.retriever import parse_metadata, retrieve_with_sources, search_images_for_query, wants_table

logger = setup_logger(__name__)


def row_value(row: Any, field: str, default: Any = None) -> Any:
    return row[field] if field in row.keys() else default


def render_sources(rows: list[Any], show_table: bool) -> None:
    st.subheader("Sources")
    if not rows:
        st.info("No indexed results matched this query.")
        return

    prepared_rows = [(row, parse_metadata(row_value(row, "metadata", {}))) for row in rows]
    if show_table:
        table_rows = []
        for row, metadata in prepared_rows:
            vehicle_type = metadata.get("vehicle_type", "Not indexed")
            table_rows.append({
                "Part / model": metadata.get("part_name") or metadata.get("model") or row_value(row, "chunk_id", ""),
                "Price": metadata.get("price") or "Not indexed",
                "Vehicle type": ", ".join(vehicle_type) if isinstance(vehicle_type, list) else vehicle_type,
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

    for row, metadata in prepared_rows:
        title = metadata.get("part_name") or metadata.get("model") or row_value(row, "image_id") or row_value(row, "chunk_id", "Source")
        with st.expander(str(title)):
            if row_value(row, "text"):
                st.write(str(row_value(row, "text"))[:1000])
            image_reference = metadata.get("local_path") or metadata.get("image_url")
            if image_reference:
                st.image(image_reference, caption=metadata.get("caption") or None)
            elif metadata.get("product_url"):
                st.link_button("Open product page", metadata["product_url"])
            logger.debug("Source metadata | source=%s | metadata=%s", title, metadata)


st.set_page_config(page_title="Tata Automotive RAG", page_icon="T", layout="wide")
st.title("Tata Automotive RAG")

with st.sidebar:
    st.header("Search settings")
    modality = st.radio("Search type", ["Text", "Image"], horizontal=True)
    category = st.selectbox("Category", ["All", "cars", "motorcycles", "scooters", "spare_parts"])
    top_k = st.slider("Results", min_value=1, max_value=10, value=5)

query = st.text_input("Ask about vehicles, features, parts, or specifications")
search_clicked = st.button("Search", type="primary")

if search_clicked:
    if not query.strip():
        st.warning("Enter a question before searching.")
    else:
        selected_category = None if category == "All" else category
        logger.info("UI search started | modality=%s | category=%s | query=%r", modality, selected_category, query)
        try:
            with st.spinner("Searching the indexed knowledge base..."):
                if modality == "Text":
                    answer, rows = asyncio.run(
                        retrieve_with_sources(query, category=selected_category, top_k=top_k)
                    )
                    st.subheader("Answer")
                    st.markdown(answer)
                else:
                    rows = asyncio.run(
                        search_images_for_query(query, category=selected_category, top_k=top_k)
                    )
            render_sources(rows, show_table=wants_table(query))
            logger.info("UI search completed | results=%d", len(rows))
        except TimeoutError:
            logger.exception("UI search timed out")
            st.error("The local Ollama model did not respond in time. Check that it is running, then try again.")
        except Exception as exc:
            logger.exception("UI search failed")
            st.error(f"Retrieval failed: {exc}")

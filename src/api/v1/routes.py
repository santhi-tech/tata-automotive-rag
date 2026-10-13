"""Product routes — all under /api/v1/ via the router prefix."""
from __future__ import annotations

import json
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.deps import ProductItem, ProductListResponse, get_conn, get_current_user

router = APIRouter(dependencies=[Depends(get_current_user)])

VALID_CATEGORIES = {"cars", "motorcycles", "scooters", "spare_parts"}


def _parse_product(row: asyncpg.Record) -> ProductItem:
    meta: dict[str, Any] = {}
    raw = row.get("metadata")
    if raw:
        try:
            meta = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (json.JSONDecodeError, TypeError):
            pass

    name = meta.get("part_name") or meta.get("vehicle_name") or meta.get("model") or None
    vtype = meta.get("vehicle_type") or meta.get("vehicle_types")
    if isinstance(vtype, list):
        vtype = ", ".join(str(v) for v in vtype)

    text_preview = str(row.get("text") or "")[:500] or None

    return ProductItem(
        category=row["category"],
        name=name,
        brand=meta.get("brand"),
        price=meta.get("price"),
        vehicle_type=str(vtype) if vtype else None,
        product_url=meta.get("product_url") or meta.get("source_url"),
        text=text_preview if not name else None,
    )


async def _fetch_products(category: str, limit: int, offset: int) -> ProductListResponse:
    if category not in VALID_CATEGORIES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"category must be one of {sorted(VALID_CATEGORIES)}")
    conn = await get_conn()
    try:
        rows = await conn.fetch(
            "SELECT category, text, metadata::text as metadata FROM brochure_chunks WHERE category = $1 ORDER BY chunk_id LIMIT $2 OFFSET $3",
            category, limit, offset,
        )
    finally:
        await conn.close()
    products = [_parse_product(r) for r in rows]
    return ProductListResponse(category=category, count=len(products), products=products)


@router.get("/cars", response_model=ProductListResponse)
async def get_cars(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    return await _fetch_products("cars", limit, offset)


@router.get("/motorcycles", response_model=ProductListResponse)
async def get_motorcycles(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    return await _fetch_products("motorcycles", limit, offset)


@router.get("/scooters", response_model=ProductListResponse)
async def get_scooters(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    return await _fetch_products("scooters", limit, offset)


@router.get("/spare-parts", response_model=ProductListResponse)
async def get_spare_parts(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    return await _fetch_products("spare_parts", limit, offset)


@router.get("/products", response_model=ProductListResponse)
async def get_products(
    category: str = Query(..., description="cars, motorcycles, scooters, or spare_parts"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    return await _fetch_products(category, limit, offset)


@router.get("/search", response_model=ProductListResponse)
async def search_products(
    q: str = Query(..., min_length=1, description="Search query"),
    category: str | None = Query(None),
    limit: int = Query(10, ge=1, le=50),
):
    if category and category not in VALID_CATEGORIES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"category must be one of {sorted(VALID_CATEGORIES)}")
    terms = " & ".join(q.lower().split()[:5])
    conn = await get_conn()
    try:
        if category:
            rows = await conn.fetch(
                """SELECT category, text, metadata::text as metadata FROM brochure_chunks
                   WHERE to_tsvector('english', text) @@ to_tsquery('english', $1) AND category = $2
                   ORDER BY ts_rank(to_tsvector('english', text), to_tsquery('english', $1)) DESC LIMIT $3""",
                terms, category, limit,
            )
        else:
            rows = await conn.fetch(
                """SELECT category, text, metadata::text as metadata FROM brochure_chunks
                   WHERE to_tsvector('english', text) @@ to_tsquery('english', $1)
                   ORDER BY ts_rank(to_tsvector('english', text), to_tsquery('english', $1)) DESC LIMIT $2""",
                terms, limit,
            )
    finally:
        await conn.close()
    products = [_parse_product(r) for r in rows]
    return ProductListResponse(category=category or "all", count=len(products), products=products)

"""Tata Automotive RAG API — app factory and startup."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config.logger import setup_logger
from api.deps import get_conn
from api.v1.auth import router as auth_router
from api.v1.routes import router as v1_router

logger = setup_logger(__name__)

app = FastAPI(
    title="Tata Automotive RAG API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(auth_router, prefix="/api/v1", tags=["auth"])
app.include_router(v1_router, prefix="/api/v1", tags=["products"])


@app.on_event("startup")
async def startup():
    conn = await get_conn()
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS api_users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """)
    finally:
        await conn.close()
    logger.info("API started | users table ready | docs at /docs")


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Tata Automotive RAG API", "docs": "/docs", "version": "v1"}

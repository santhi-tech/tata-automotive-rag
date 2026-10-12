"""Shared dependencies: DB connection, auth, JWT, models."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import asyncpg
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from config.settings import DB_PARAMS

JWT_SECRET = os.environ.get("JWT_SECRET", "tata-rag-dev-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24

security = HTTPBearer()


async def get_conn() -> asyncpg.Connection:
    return await asyncpg.connect(timeout=10, command_timeout=20, **DB_PARAMS)


def create_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")


# ---------------------------------------------------------------------------
# Shared Pydantic models
# ---------------------------------------------------------------------------

class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=128)

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRE_HOURS * 3600

class ProductItem(BaseModel):
    category: str
    name: str | None = None
    brand: str | None = None
    price: str | None = None
    vehicle_type: str | None = None
    product_url: str | None = None
    text: str | None = None

class ProductListResponse(BaseModel):
    category: str
    count: int
    products: list[ProductItem]

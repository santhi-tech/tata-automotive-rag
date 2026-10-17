"""Auth routes: signup and login."""
from __future__ import annotations

import bcrypt
from fastapi import APIRouter, HTTPException, status

from api.deps import SignupRequest, TokenResponse, create_token, get_conn

router = APIRouter()


@router.post("/signup", response_model=TokenResponse, status_code=201)
async def signup(req: SignupRequest):
    conn = await get_conn()
    try:
        existing = await conn.fetchval("SELECT 1 FROM api_users WHERE username = $1", req.username)
        if existing:
            raise HTTPException(status.HTTP_409_CONFLICT, "Username already taken")
        hashed = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
        await conn.execute("INSERT INTO api_users (username, password_hash) VALUES ($1, $2)", req.username, hashed)
    finally:
        await conn.close()
    return TokenResponse(access_token=create_token(req.username))


@router.post("/login", response_model=TokenResponse)
async def login(req: SignupRequest):
    conn = await get_conn()
    try:
        row = await conn.fetchrow("SELECT password_hash FROM api_users WHERE username = $1", req.username)
    finally:
        await conn.close()
    if not row or not bcrypt.checkpw(req.password.encode(), row["password_hash"].encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Bad credentials")
    return TokenResponse(access_token=create_token(req.username))

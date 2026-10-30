"""Conversation replay for operators.

  GET /admin/replay                       the replay page (pick a conversation, see it)
  GET /admin/conversations?limit=20       recent conversations
  GET /admin/conversations/{id}           transcript, tool calls, policy decisions, timings

These show customer details, so they're guarded: with ADMIN_TOKEN set, requests need
"Authorization: Bearer <token>"; without it, only requests from this machine are served.
"""

import hmac
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.observability import conversation

router = APIRouter(prefix="/admin")
PAGE = Path(__file__).resolve().parents[1] / "observability" / "static" / "replay.html"
LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


def _authorized(request: Request) -> None:
    token = request.app.state.settings.admin_token
    if token is not None and token.get_secret_value():
        given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given.encode(), token.get_secret_value().encode()):
            raise HTTPException(401, "Admin token required.")
        return
    host = request.client.host if request.client else ""
    if host not in LOOPBACK:
        raise HTTPException(403, "Set ADMIN_TOKEN to use the admin API from another machine.")


@router.get("/replay", include_in_schema=False)
async def page() -> FileResponse:
    # The page holds no data; it calls the guarded API below.
    return FileResponse(PAGE, media_type="text/html")


@router.get("/conversations", dependencies=[Depends(_authorized)])
async def recent(request: Request, limit: int = Query(20, ge=1, le=200)) -> list[dict[str, Any]]:
    async with AsyncSession(request.app.state.engine) as db:
        return await conversation.list_recent(db, limit)


@router.get("/conversations/{conversation_id}", dependencies=[Depends(_authorized)])
async def one(request: Request, conversation_id: int) -> dict[str, Any]:
    salt = request.app.state.settings.postgres_db
    async with AsyncSession(request.app.state.engine) as db:
        data = await conversation.load(db, conversation_id, salt=salt)
    if data is None:
        raise HTTPException(404, "No such conversation.")
    return data

"""Conversation replay for operators.

  GET /admin/replay                       the replay page (pick a conversation, see it)
  GET /admin/conversations?limit=20       recent conversations
  GET /admin/conversations/{id}           transcript, tool calls, policy decisions, timings
  GET /admin/evals                        eval harness runs, newest first
  GET /admin/evals/latest                 the newest run's results.json
  GET /admin/evals/{run}                  one run's results.json

These show customer details, so they're guarded: with ADMIN_TOKEN set, requests need
"Authorization: Bearer <token>"; without it, only requests from this machine are served.
"""

import hmac
import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.observability import conversation

router = APIRouter(prefix="/admin")
PAGE = Path(__file__).resolve().parents[1] / "observability" / "static" / "replay.html"
LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")  # a results/ folder, never a path


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


# --- eval results -------------------------------------------------------------------------


def _runs(request: Request) -> list[Path]:
    """results.json files under the results directory, newest first (by file time)."""
    base: Path = request.app.state.settings.eval_results_dir
    if not base.is_dir():
        return []
    files = [p for p in base.glob("*/results.json") if RUN_NAME.fullmatch(p.parent.name)]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"Unreadable results file: {path.parent.name}") from exc
    return {"name": path.parent.name, **data}


@router.get("/evals", dependencies=[Depends(_authorized)])
async def eval_runs(request: Request) -> list[dict[str, Any]]:
    out = []
    for path in _runs(request):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        summary = data.get("summary") or {}
        out.append(
            {
                "name": path.parent.name,
                "run_id": data.get("run_id"),
                "started_at": data.get("started_at"),
                "agent_model": data.get("agent_model"),
                "runs": summary.get("runs"),
                "task_success_rate": summary.get("task_success_rate"),
            }
        )
    return out


@router.get("/evals/latest", dependencies=[Depends(_authorized)])
async def latest_eval(request: Request) -> dict[str, Any]:
    runs = _runs(request)
    if not runs:
        raise HTTPException(404, "No eval results yet: run `make eval-smoke`.")
    return _read(runs[0])


@router.get("/evals/{name}", dependencies=[Depends(_authorized)])
async def one_eval(request: Request, name: str) -> dict[str, Any]:
    if not RUN_NAME.fullmatch(name):
        raise HTTPException(404, "No such eval run.")
    path = request.app.state.settings.eval_results_dir / name / "results.json"
    if not path.is_file():
        raise HTTPException(404, "No such eval run.")
    return _read(path)

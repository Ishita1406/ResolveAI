import asyncio
from typing import Literal

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import get_settings

router = APIRouter()


class ComponentStatus(BaseModel):
    status: Literal["ok", "error"]
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, ComponentStatus]


async def ping_postgres(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


async def ping_redis(client: Redis) -> None:
    await client.ping()


async def _check(probe, timeout_s: float) -> ComponentStatus:
    try:
        await asyncio.wait_for(probe, timeout=timeout_s)
    except Exception as exc:
        # Exception type only: messages can leak hostnames or usernames.
        return ComponentStatus(status="error", error=type(exc).__name__)
    return ComponentStatus(status="ok")


@router.get("/health", response_model=HealthResponse)
async def health(request: Request, response: Response) -> HealthResponse:
    timeout_s = get_settings().health_check_timeout_s
    postgres, redis = await asyncio.gather(
        _check(ping_postgres(request.app.state.engine), timeout_s),
        _check(ping_redis(request.app.state.redis), timeout_s),
    )
    checks = {"postgres": postgres, "redis": redis}
    healthy = all(c.status == "ok" for c in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="ok" if healthy else "degraded", checks=checks)

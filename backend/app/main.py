from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import create_async_engine

from app.api.health import router as health_router
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.engine = create_async_engine(settings.database_url(), pool_pre_ping=True)
    app.state.redis = Redis.from_url(settings.redis_url)
    try:
        yield
    finally:
        await app.state.redis.aclose()
        await app.state.engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(title="ResolveAI", lifespan=lifespan)
    app.include_router(health_router)
    return app


app = create_app()

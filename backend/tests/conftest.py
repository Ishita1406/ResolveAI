import re
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from app.config import Settings, get_settings
from app.db.seed import build_dataset, write_dataset
from app.tools.context import ToolContext
from tests.tool_helpers import make_ctx

BACKEND_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    # Tests monkeypatch env vars; don't let a cached Settings leak between tests.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _ensure_database(settings: Settings, name: str) -> None:
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError(f"unsafe test database name: {name!r}")
    admin = create_engine(
        settings.database_url().set(database="postgres"), isolation_level="AUTOCOMMIT"
    )
    try:
        with admin.connect() as conn:
            exists = conn.scalar(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name})
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        admin.dispose()


@pytest.fixture(scope="session")
def test_db_settings() -> Iterator[Settings]:
    """Migrate (down to base, then up) and seed the test database once per session."""
    get_settings.cache_clear()
    base = get_settings()
    name = base.test_postgres_db
    if name == base.postgres_db or "test" not in name:
        pytest.exit(f"Refusing to use {name!r} as the (destructive) test database.", returncode=2)
    _ensure_database(base, name)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("POSTGRES_DB", name)
        get_settings.cache_clear()
        settings = get_settings()

        alembic_cfg = Config(BACKEND_DIR / "alembic.ini")
        command.downgrade(alembic_cfg, "base")
        command.upgrade(alembic_cfg, "head")

        engine = create_engine(settings.database_url())
        with Session(engine) as session, session.begin():
            write_dataset(session, build_dataset())
        engine.dispose()

        yield settings


@pytest.fixture
async def db_session(test_db_settings: Settings) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(test_db_settings.database_url(), poolclass=NullPool)
    async with AsyncSession(engine) as session:
        yield session
        await session.rollback()
    await engine.dispose()


@pytest.fixture
async def tx_db(test_db_settings: Settings) -> AsyncIterator[AsyncSession]:
    """Session inside an outer transaction that is always rolled back.

    Tools call commit()/rollback(); in create_savepoint mode those only end a savepoint,
    so write tools behave normally yet leave the seeded test database unchanged.
    """
    engine = create_async_engine(test_db_settings.database_url(), poolclass=NullPool)
    async with engine.connect() as conn:
        outer = await conn.begin()
        session = AsyncSession(bind=conn, join_transaction_mode="create_savepoint")
        try:
            yield session
        finally:
            await session.close()
            await outer.rollback()
    await engine.dispose()


@pytest.fixture
def ctx(tx_db: AsyncSession) -> ToolContext:
    return make_ctx(tx_db)

"""A freshly seeded database for every scenario run.

One template is built per harness run (migrate, seed, ingest the policy documents); each run
then gets `CREATE DATABASE ... TEMPLATE`, a file-level copy that takes well under a second,
and the copy is dropped afterwards. Runs never see each other's writes.

Every name starts with "<POSTGRES_DB>_eval_" and is checked against the app and test
databases, so the harness can't touch either.
"""

import asyncio
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import Conversation, Order, Refund, Return, Ticket
from app.db.seed import build_dataset, write_dataset
from app.retrieval.chunking import load_documents
from app.retrieval.embedding import Embedder
from app.retrieval.ingest import ingest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
# Tables whose new rows the checks look at: the template's max id is the baseline.
TRACKED = {
    "refunds": Refund,
    "returns": Return,
    "orders": Order,
    "tickets": Ticket,
    "conversations": Conversation,
}


class Databases:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.prefix = f"{settings.postgres_db}_eval_"
        self.template = f"{self.prefix}template"
        self.baseline: dict[str, int] = {}
        self._admin = create_engine(
            settings.database_url().set(database="postgres"), isolation_level="AUTOCOMMIT"
        )
        self._lock = asyncio.Lock()  # CREATE DATABASE from one template, one at a time

    def _check(self, name: str) -> str:
        if not re.fullmatch(r"[a-z0-9_]+", name) or not name.startswith(self.prefix):
            raise ValueError(f"refusing to manage database {name!r}")
        if name in (self.settings.postgres_db, self.settings.test_postgres_db):
            raise ValueError(f"refusing to manage database {name!r}")
        return name

    def settings_for(self, name: str, **overrides) -> Settings:
        return self.settings.model_copy(update={"postgres_db": self._check(name), **overrides})

    def _execute(self, sql: str) -> None:
        with self._admin.connect() as conn:
            conn.execute(text(sql))

    def _drop(self, name: str) -> None:
        self._execute(f'DROP DATABASE IF EXISTS "{self._check(name)}" WITH (FORCE)')

    async def build_template(self, embedder: Embedder) -> None:
        await asyncio.to_thread(self._drop, self.template)
        await asyncio.to_thread(self._execute, f'CREATE DATABASE "{self.template}"')
        env = {**os.environ, "POSTGRES_DB": self.template}
        # A subprocess: alembic's env.py reads settings from the environment.
        await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=BACKEND_DIR,
            env=env,
            check=True,
            capture_output=True,
        )
        url = self.settings_for(self.template).database_url()
        engine = create_engine(url)
        try:
            with Session(engine) as session, session.begin():
                write_dataset(session, build_dataset())
        finally:
            engine.dispose()
        aengine = create_async_engine(url)
        try:
            async with AsyncSession(aengine) as db:
                await ingest(db, embedder, load_documents(self.settings.policy_docs_dir))
                for key, model in TRACKED.items():
                    self.baseline[key] = await db.scalar(
                        select(func.coalesce(func.max(model.id), 0))
                    )
        finally:
            await aengine.dispose()

    async def create_run_db(self) -> str:
        name = self._check(f"{self.prefix}{uuid.uuid4().hex[:12]}")
        async with self._lock:
            await asyncio.to_thread(
                self._execute, f'CREATE DATABASE "{name}" TEMPLATE "{self.template}"'
            )
        return name

    async def drop(self, name: str) -> None:
        await asyncio.to_thread(self._drop, name)

    async def drop_template(self) -> None:
        await asyncio.to_thread(self._drop, self.template)

    async def drop_leftovers(self) -> list[str]:
        """Run databases left by an interrupted harness run."""

        def names() -> list[str]:
            with self._admin.connect() as conn:
                rows = conn.execute(
                    text("SELECT datname FROM pg_database WHERE datname LIKE :p ESCAPE '\\'"),
                    {"p": self.prefix.replace("_", "\\_") + "%"},  # "_" is a LIKE wildcard
                )
                return [
                    r[0] for r in rows if r[0].startswith(self.prefix) and r[0] != self.template
                ]

        leftovers = await asyncio.to_thread(names)
        for name in leftovers:
            await self.drop(name)
        return leftovers

    def dispose(self) -> None:
        self._admin.dispose()

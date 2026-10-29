"""The real app, served by uvicorn on a free local port, bound to one run's database."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from datetime import datetime

import uvicorn

from app.agent.llm import LLMClient
from app.config import Settings
from app.main import create_app
from app.retrieval.embedding import Embedder


class _Server(uvicorn.Server):
    # Many servers share this process; none of them may take over Ctrl-C.
    @contextlib.contextmanager
    def capture_signals(self):
        yield


@contextlib.asynccontextmanager
async def serve(
    settings: Settings, *, llm: LLMClient, embedder: Embedder, clock: Callable[[], datetime]
) -> AsyncIterator[str]:
    """Yields the WebSocket URL of /ws/chat."""
    app = create_app(settings, llm=llm, embedder=embedder, clock=clock)
    server = _Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on")
    )
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            if task.done():
                task.result()  # startup failed: raise its error
                raise RuntimeError("server stopped during startup")
            await asyncio.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}/ws/chat"
    finally:
        server.should_exit = True
        await task

"""Text embeddings. Production uses a local ONNX model through fastembed (no API key, no
per-call cost, no network once cached); tests use a deterministic stand-in."""

import asyncio
import threading
from pathlib import Path
from typing import Protocol

from app.config import Settings, get_settings
from app.db.models import EMBEDDING_DIM


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    async def warm(self) -> None: ...


class FastEmbedder:
    def __init__(self, model_name: str, *, cache_dir: Path, query_prefix: str = "") -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.query_prefix = query_prefix
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding  # heavy import; only when used

                self._model = TextEmbedding(self.model_name, cache_dir=str(self.cache_dir))
            return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_checked(v.tolist()) for v in self._load().passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        (vector,) = self._load().query_embed([self.query_prefix + text])
        return _checked(vector.tolist())

    async def warm(self) -> None:
        """Load the model off the event loop (first use downloads it into the cache)."""
        await asyncio.to_thread(self._load)


def _checked(vector: list[float]) -> list[float]:
    if len(vector) != EMBEDDING_DIM:
        raise ValueError(
            f"embedding has {len(vector)} dimensions; the policy_chunks column holds "
            f"{EMBEDDING_DIM}. Use a {EMBEDDING_DIM}-dimension model or add a migration."
        )
    return vector


def embedder_from_settings(settings: Settings | None = None) -> FastEmbedder:
    settings = settings or get_settings()
    return FastEmbedder(
        settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
        query_prefix=settings.embedding_query_prefix,
    )

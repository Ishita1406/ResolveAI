"""A deterministic stand-in for the embedding model, so retrieval tests need no download.

Words (lower-cased, crudely stemmed) are hashed into the vector with a sign, then the vector
is L2-normalised: texts sharing words get a higher cosine similarity. That is enough to test
the plumbing (storage, ordering, fusion, versioning); retrieval *quality* is measured with
the real model by app.retrieval.evaluate and tests/test_retrieval_quality.py.
"""

import hashlib
import math
import re

from app.db.models import EMBEDDING_DIM
from app.retrieval.search import PolicyRetriever

_STOP_WORDS = "a an the is it i my do you to of for and or can be what how if in on me we our"
STOP = frozenset((_STOP_WORDS + " your does").split())


def _terms(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [re.sub(r"(ing|ed|es|s)$", "", w) or w for w in words if w not in STOP]


class HashingEmbedder:
    def __init__(self, model_name: str = "test/hashing-v1") -> None:
        self.model_name = model_name
        self.queries: list[str] = []

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIM
        for term in _terms(text):
            digest = hashlib.sha256(term.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
            vector[index] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._embed(text)

    async def warm(self) -> None:
        pass


def make_retriever(embedder: HashingEmbedder | None = None, **kwargs) -> PolicyRetriever:
    return PolicyRetriever(embedder or HashingEmbedder(), **kwargs)

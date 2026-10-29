"""Hybrid policy search: PostgreSQL full-text search and pgvector similarity, each ranked on
its own and merged with reciprocal rank fusion (RRF).

RRF uses only ranks, so the two scores (ts_rank_cd and cosine distance), which are on
unrelated scales, never need normalising: score(d) = sum over rankings of 1 / (k + rank).
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import Text, cast, func, literal_column, select
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PolicyChunk
from app.retrieval.embedding import Embedder

Mode = Literal["hybrid", "vector", "text"]
MODES: tuple[Mode, ...] = ("vector", "text", "hybrid")

_ENGLISH = literal_column("'english'::regconfig")


def any_term_query(query: str):
    """Questions are natural language, so match ANY term (plainto_tsquery alone requires all
    of them); ts_rank_cd then favours sections matching more terms, closer together, and in
    the heading. Stop words are dropped, so "is it?" matches nothing."""
    all_terms = cast(func.plainto_tsquery(_ENGLISH, query), Text)
    return cast(func.replace(all_terms, "&", "|"), TSQUERY)


@dataclass(frozen=True)
class Passage:
    document: str
    title: str
    section: str
    section_id: str
    content: str
    version: str
    score: float

    @property
    def key(self) -> str:
        return f"{self.document}#{self.section_id}"

    @property
    def citation(self) -> str:
        return f"{self.title}: {self.section}"


class IndexEmptyError(Exception):
    """No chunks for this embedding model: run the ingestion script."""


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    # Ties (e.g. an item 1st in one list and absent from the other, vs. the reverse) keep a
    # stable order: by score, then by the first list's order, then by id.
    first = {item: i for i, item in enumerate(rankings[0])} if rankings else {}
    return sorted(scores.items(), key=lambda kv: (-kv[1], first.get(kv[0], len(first)), kv[0]))


class PolicyRetriever:
    def __init__(
        self,
        embedder: Embedder,
        *,
        mode: Mode = "hybrid",
        top_k: int = 3,
        candidates: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.mode = mode
        self.top_k = top_k
        self.candidates = candidates
        self.rrf_k = rrf_k

    async def vector_ranking(self, db: AsyncSession, query: str) -> list[int]:
        """Chunk ids by cosine distance to the query embedding, best first."""
        model = self.embedder.model_name
        vector = await asyncio.to_thread(self.embedder.embed_query, query)
        ids = list(
            await db.scalars(
                select(PolicyChunk.id)
                .where(PolicyChunk.embedding_model == model)
                .order_by(PolicyChunk.embedding.cosine_distance(vector), PolicyChunk.id)
                .limit(self.candidates)
            )
        )
        if not ids:
            raise IndexEmptyError(f"no policy chunks embedded with {model}")
        return ids

    async def text_ranking(self, db: AsyncSession, query: str) -> list[int]:
        """Chunk ids matching any query term, by ts_rank_cd, best first. No embedding."""
        tsquery = any_term_query(query)
        return list(
            await db.scalars(
                select(PolicyChunk.id)
                .where(
                    PolicyChunk.embedding_model == self.embedder.model_name,
                    PolicyChunk.tsv.op("@@")(tsquery),
                )
                .order_by(func.ts_rank_cd(PolicyChunk.tsv, tsquery).desc(), PolicyChunk.id)
                .limit(self.candidates)
            )
        )

    async def ranked(self, db: AsyncSession, query: str, mode: Mode) -> list[tuple[int, float]]:
        """(chunk id, fused score), best first. Runs only the rankings the mode needs."""
        rankings = []
        if mode in ("vector", "hybrid"):
            rankings.append(await self.vector_ranking(db, query))
        if mode in ("text", "hybrid"):
            rankings.append(await self.text_ranking(db, query))
        return reciprocal_rank_fusion(rankings, self.rrf_k)

    async def search(
        self, db: AsyncSession, query: str, *, mode: Mode | None = None, k: int | None = None
    ) -> list[Passage]:
        ranked = (await self.ranked(db, query, mode or self.mode))[: k or self.top_k]
        if not ranked:
            return []
        rows = {
            row.id: row
            for row in await db.scalars(
                select(PolicyChunk).where(PolicyChunk.id.in_([i for i, _ in ranked]))
            )
        }
        return [
            Passage(
                document=rows[i].document,
                title=rows[i].title,
                section=rows[i].section,
                section_id=rows[i].section_id,
                content=rows[i].content,
                version=rows[i].version,
                score=round(score, 6),
            )
            for i, score in ranked
        ]

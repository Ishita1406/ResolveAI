"""Ingestion, hybrid search and the search_policies tool, against the test database.

Uses the deterministic HashingEmbedder: these tests check behaviour, not retrieval quality
(see test_retrieval_quality.py for the real model)."""

import numpy as np
import pytest
from sqlalchemy import func, select, update

from app.config import POLICY_DOCS_DIR
from app.db.models import EMBEDDING_DIM, PolicyChunk
from app.retrieval.chunking import chunk_document, load_documents
from app.retrieval.embedding import FastEmbedder
from app.retrieval.ingest import ingest
from app.retrieval.search import IndexEmptyError, reciprocal_rank_fusion
from app.tools.errors import ErrorCode
from app.tools.registry import call_tool
from tests.retrieval_helpers import HashingEmbedder, make_retriever
from tests.tool_helpers import make_ctx

pytestmark = pytest.mark.anyio

CHUNKS = load_documents(POLICY_DOCS_DIR)


async def _count(db, *where):
    return await db.scalar(select(func.count()).select_from(PolicyChunk).where(*where))


@pytest.fixture
async def indexed(tx_db):
    await ingest(tx_db, HashingEmbedder(), CHUNKS)
    return tx_db


# --- reciprocal rank fusion (pure) ----------------------------------------------------------


def test_rrf_rewards_agreement_between_rankings():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 4, 1]], k=60)
    assert [item for item, _ in fused] == [1, 3, 2, 4]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_ties_follow_the_first_ranking():
    # 7 is 1st in one list, 8 is 1st in the other: equal scores, first list wins.
    assert [i for i, _ in reciprocal_rank_fusion([[7], [8]], k=60)] == [7, 8]
    assert reciprocal_rank_fusion([[], []], k=60) == []


# --- ingestion ------------------------------------------------------------------------------


@pytest.mark.db
async def test_ingest_is_idempotent(tx_db):
    embedder = HashingEmbedder()
    first = await ingest(tx_db, embedder, CHUNKS)
    assert (sorted(first.embedded), first.chunks) == (
        ["damaged-items", "refunds", "returns", "shipping"], 21)  # fmt: skip
    again = await ingest(tx_db, embedder, CHUNKS)
    assert (again.embedded, len(again.unchanged), again.chunks) == ([], 4, 0)
    forced = await ingest(tx_db, embedder, CHUNKS, force=True)
    assert forced.chunks == 21
    assert await _count(tx_db) == 21


@pytest.mark.db
async def test_changed_document_is_replaced_and_removed_one_deleted(indexed):
    path = POLICY_DOCS_DIR / "returns.md"
    edited = chunk_document("returns", path.read_text() + "\n## New section\n\nNew text.\n")
    others = [c for c in CHUNKS if c.document not in ("returns", "shipping")]
    report = await ingest(indexed, HashingEmbedder(), others + edited)
    assert report.embedded == ["returns"] and report.removed == ["shipping"]
    versions = await indexed.scalars(
        select(PolicyChunk.version).where(PolicyChunk.document == "returns").distinct()
    )
    assert list(versions) == [edited[0].version]
    assert await _count(indexed, PolicyChunk.document == "returns") == 6
    assert await _count(indexed, PolicyChunk.document == "shipping") == 0


@pytest.mark.db
async def test_new_embedding_model_re_embeds_everything(indexed):
    report = await ingest(indexed, HashingEmbedder("test/hashing-v2"), CHUNKS)
    assert report.chunks == 21
    models = await indexed.scalars(select(PolicyChunk.embedding_model).distinct())
    assert list(models) == ["test/hashing-v2"]


@pytest.mark.db
async def test_tsv_is_generated_from_heading_and_text(indexed):
    goodwill = await indexed.scalar(
        select(PolicyChunk).where(PolicyChunk.section_id == "goodwill-refunds")
    )
    assert "goodwil" in goodwill.tsv and "specialist" in goodwill.tsv


# --- search ---------------------------------------------------------------------------------


@pytest.mark.db
async def test_hybrid_search_returns_cited_passages(indexed):
    passages = await make_retriever().search(indexed, "Can I get a goodwill refund?")
    assert len(passages) == 3
    top = passages[0]
    assert top.key == "refunds#goodwill-refunds"
    assert top.citation == "Refunds policy: Goodwill refunds"
    assert "specialist" in top.content
    assert [p.score for p in passages] == sorted((p.score for p in passages), reverse=True)


@pytest.mark.db
async def test_each_mode_uses_its_own_ranking(indexed):
    retriever = make_retriever(candidates=21)
    embedder = retriever.embedder
    assert len(await retriever.vector_ranking(indexed, "shipping fee")) == 21  # all have one
    text_ids = await retriever.text_ranking(indexed, "shipping fee")
    assert text_ids and len(text_ids) < 21  # only chunks containing a term
    before = len(embedder.queries)
    await retriever.search(indexed, "shipping fee", mode="text")
    assert len(embedder.queries) == before  # text-only search never embeds the query
    text_first = (await retriever.search(indexed, "shipping fee", mode="text", k=1))[0]
    assert text_first.key == "shipping#how-much-does-shipping-cost"


@pytest.mark.db
async def test_stop_words_only_fall_back_to_vector_ranking(indexed):
    retriever = make_retriever()
    assert await retriever.search(indexed, "is it?", mode="text") == []
    assert len(await retriever.search(indexed, "is it?", mode="hybrid")) == 3


@pytest.mark.db
async def test_chunks_from_another_model_are_not_searched(indexed):
    with pytest.raises(IndexEmptyError):
        await make_retriever(HashingEmbedder("test/other-model")).search(indexed, "refund")


@pytest.mark.db
async def test_mixed_model_rows_are_ignored(indexed):
    await indexed.execute(
        update(PolicyChunk)
        .where(PolicyChunk.document == "refunds")
        .values(embedding_model="test/stale-model")
    )
    passages = await make_retriever().search(indexed, "goodwill refund", k=21)
    assert passages and all(p.document != "refunds" for p in passages)


# --- the tool -------------------------------------------------------------------------------


@pytest.mark.db
async def test_search_policies_needs_no_verification(indexed):
    ctx = make_ctx(indexed)
    ctx.retriever = make_retriever()
    result = await call_tool(ctx, "search_policies", {"query": "goodwill refund"})
    assert result.ok, result.error
    first = result.data["passages"][0]
    assert first == {
        "source": "Refunds policy: Goodwill refunds",
        "section_id": "refunds#goodwill-refunds",
        "text": first["text"],
    }
    assert not ctx.state.is_verified


@pytest.mark.db
async def test_search_policies_unavailable(tx_db):
    off = await call_tool(make_ctx(tx_db), "search_policies", {"query": "refunds"})
    assert off.error.code is ErrorCode.SEARCH_UNAVAILABLE
    ctx = make_ctx(tx_db)
    ctx.retriever = make_retriever()
    empty = await call_tool(ctx, "search_policies", {"query": "refunds"})
    assert empty.error.code is ErrorCode.SEARCH_UNAVAILABLE and "ingestion" in empty.error.hint
    short = await call_tool(ctx, "search_policies", {"query": "a"})
    assert short.error.code is ErrorCode.INVALID_ARGUMENTS


# --- the real embedder's wrapper (model stubbed) --------------------------------------------


class _StubModel:
    def __init__(self, dim=EMBEDDING_DIM):
        self.dim, self.seen = dim, []

    def query_embed(self, texts):
        self.seen += texts
        return [np.ones(self.dim, dtype=np.float32) for _ in texts]

    def passage_embed(self, texts):
        self.seen += texts
        return [np.ones(self.dim, dtype=np.float32) for _ in texts]


def test_query_prefix_is_added_to_queries_only(tmp_path):
    embedder = FastEmbedder("m", cache_dir=tmp_path, query_prefix="Represent: ")
    embedder._model = stub = _StubModel()
    embedder.embed_query("refund?")
    embedder.embed_documents(["passage"])
    assert stub.seen == ["Represent: refund?", "passage"]


def test_wrong_dimension_fails_clearly(tmp_path):
    embedder = FastEmbedder("m", cache_dir=tmp_path)
    embedder._model = _StubModel(dim=768)
    with pytest.raises(ValueError, match="384"):
        embedder.embed_query("refund?")


@pytest.mark.db
async def test_configured_mode_is_used_by_default(indexed):
    embedder = HashingEmbedder()
    retriever = make_retriever(embedder, mode="text")
    passages = await retriever.search(indexed, "shipping fee")
    assert passages[0].key == "shipping#how-much-does-shipping-cost"
    assert embedder.queries == []  # text mode: no embedding

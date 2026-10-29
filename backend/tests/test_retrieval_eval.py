"""The retrieval test set and the metrics script.

The quality floor at the bottom uses the real embedding model and is skipped when the model
isn't in the local cache (python -m app.retrieval.ingest downloads it)."""

import pytest

from app.config import POLICY_DOCS_DIR, get_settings
from app.retrieval.chunking import load_documents
from app.retrieval.embedding import embedder_from_settings
from app.retrieval.evaluate import Metrics, evaluate, load_questions, percentile, report, score
from app.retrieval.ingest import ingest
from app.retrieval.search import MODES, PolicyRetriever
from tests.retrieval_helpers import HashingEmbedder, make_retriever

pytestmark = pytest.mark.anyio

QUESTIONS = load_questions()
CHUNKS = load_documents(POLICY_DOCS_DIR)


def test_test_set_has_30_questions_covering_every_section():
    assert len(QUESTIONS) == 30
    sections = {c.key for c in CHUNKS}
    expected = {q.section for q in QUESTIONS}
    assert expected <= sections, expected - sections  # a renamed heading fails here
    assert expected == sections  # and every section is asked about


def test_score():
    assert score([1, 2, None, 6]) == Metrics(
        recall_at_1=0.25, recall_at_3=0.5, recall_at_5=0.5, mrr=(1 + 1 / 2 + 1 / 6) / 4
    )


def test_percentile_is_nearest_rank():
    values = [float(v) for v in range(1, 31)]
    assert percentile(values, 95) == 29.0
    assert percentile(values, 50) == 15.0
    assert percentile([7.0], 95) == 7.0


@pytest.mark.db
async def test_evaluate_ranks_every_question_in_every_mode(tx_db):
    await ingest(tx_db, HashingEmbedder(), CHUNKS)
    results = await evaluate(tx_db, make_retriever(candidates=1000), QUESTIONS)
    assert set(results) == set(MODES)
    assert all(len(r.ranks) == len(r.latencies_ms) == 30 for r in results.values())
    # Vector and hybrid rank every chunk, so every question gets a rank.
    assert None not in results["vector"].ranks and None not in results["hybrid"].ranks
    text = report(results, QUESTIONS, header={"Model": "test"})
    assert "| Hybrid (RRF) |" in text and "| t6 |" in text


@pytest.mark.db
async def test_evaluate_refuses_a_test_set_the_index_cannot_answer(tx_db):
    await ingest(tx_db, HashingEmbedder(), [c for c in CHUNKS if c.document != "shipping"])
    with pytest.raises(ValueError, match="shipping#"):
        await evaluate(tx_db, make_retriever(), QUESTIONS)


@pytest.fixture(scope="module")
def real_embedder():
    settings = get_settings()
    try:
        from fastembed import TextEmbedding

        TextEmbedding(
            settings.embedding_model,
            cache_dir=str(settings.embedding_cache_dir),
            local_files_only=True,
        )
    except Exception:
        pytest.skip(f"{settings.embedding_model} isn't cached; run python -m app.retrieval.ingest")
    return embedder_from_settings(settings)


@pytest.mark.db
async def test_quality_floor_with_the_real_model(tx_db, real_embedder):
    """A regression guard, set below the measured baseline in evals/retrieval/results.md."""
    await ingest(tx_db, real_embedder, CHUNKS)
    results = await evaluate(tx_db, PolicyRetriever(real_embedder, candidates=1000), QUESTIONS)
    vector, text, hybrid = (results[m].metrics for m in ("vector", "text", "hybrid"))
    assert vector.recall_at_5 >= 0.9
    assert hybrid.recall_at_5 >= 0.85 and hybrid.recall_at_3 >= 0.75
    assert hybrid.mrr >= 0.65
    assert text.recall_at_5 >= 0.6

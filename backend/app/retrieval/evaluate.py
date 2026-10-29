"""Retrieval metrics on evals/retrieval/questions.yaml: recall@1, recall@5 and MRR for
vector-only, full-text-only and hybrid (RRF) search, plus per-query latency.

    python -m app.retrieval.evaluate                          # print the report
    python -m app.retrieval.evaluate --output ../evals/retrieval/results.md
    python -m app.retrieval.evaluate --query-prefix ""        # without BGE's query instruction

Runs against the configured database, which must have been ingested with the configured
embedding model (python -m app.retrieval.ingest). Each question has exactly one correct
section, so recall@k is the share of questions whose section ranks in the top k, and MRR is
the mean of 1/rank (0 when a method doesn't return the section at all).
"""

import argparse
import asyncio
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.config import get_settings
from app.db.models import PolicyChunk
from app.retrieval.embedding import embedder_from_settings
from app.retrieval.search import MODES, IndexEmptyError, Mode, PolicyRetriever

QUESTIONS = Path(__file__).resolve().parents[3] / "evals" / "retrieval" / "questions.yaml"


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    section: str


@dataclass(frozen=True)
class Metrics:
    recall_at_1: float
    recall_at_3: float  # the agent's search_policies returns the top 3
    recall_at_5: float
    mrr: float


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    data = yaml.safe_load(path.read_text())
    questions = [Question(**q) for q in data["questions"]]
    if len({q.id for q in questions}) != len(questions):
        raise ValueError(f"{path}: duplicate question ids")
    return questions


def score(ranks: list[int | None]) -> Metrics:
    """`ranks`: the correct section's 1-based rank per question, None if not returned."""
    n = len(ranks)
    return Metrics(
        recall_at_1=sum(r is not None and r <= 1 for r in ranks) / n,
        recall_at_3=sum(r is not None and r <= 3 for r in ranks) / n,
        recall_at_5=sum(r is not None and r <= 5 for r in ranks) / n,
        mrr=sum(1 / r for r in ranks if r is not None) / n,
    )


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (no interpolation: with 30 samples, p95 is the 29th)."""
    ordered = sorted(values)
    return ordered[max(math.ceil(p / 100 * len(ordered)) - 1, 0)]


@dataclass
class ModeResult:
    ranks: list[int | None]
    latencies_ms: list[float]

    @property
    def metrics(self) -> Metrics:
        return score(self.ranks)


async def evaluate(
    db: AsyncSession, retriever: PolicyRetriever, questions: list[Question]
) -> dict[Mode, ModeResult]:
    keys = {
        row.id: f"{row.document}#{row.section_id}"
        for row in await db.execute(
            select(PolicyChunk.id, PolicyChunk.document, PolicyChunk.section_id).where(
                PolicyChunk.embedding_model == retriever.embedder.model_name
            )
        )
    }
    missing = sorted({q.section for q in questions} - set(keys.values()))
    if missing:
        raise ValueError(f"sections not in the index: {', '.join(missing)}")
    results: dict[Mode, ModeResult] = {}
    for mode in MODES:
        result = ModeResult(ranks=[], latencies_ms=[])
        for q in questions:
            started = time.perf_counter()
            ranked = [keys[i] for i, _ in await retriever.ranked(db, q.question, mode)]
            result.latencies_ms.append((time.perf_counter() - started) * 1000)
            result.ranks.append(ranked.index(q.section) + 1 if q.section in ranked else None)
        results[mode] = result
    return results


def report(
    results: dict[Mode, ModeResult], questions: list[Question], *, header: dict[str, str]
) -> str:
    lines = ["# Retrieval evaluation", ""]
    lines += [f"- **{k}:** {v}" for k, v in header.items()]
    lines += [
        "",
        "| Method | Recall@1 | Recall@3 | Recall@5 | MRR | Mean latency | p95 latency |",
        "|---|---|---|---|---|---|---|",
    ]
    names = {"vector": "Vector only", "text": "Full-text only", "hybrid": "Hybrid (RRF)"}
    for mode, result in results.items():
        m = result.metrics
        lines.append(
            f"| {names[mode]} | {m.recall_at_1:.3f} | {m.recall_at_3:.3f} "
            f"| {m.recall_at_5:.3f} | {m.mrr:.3f} "
            f"| {statistics.mean(result.latencies_ms):.1f} ms "
            f"| {percentile(result.latencies_ms, 95):.1f} ms |"
        )
    lines += [
        "",
        "Rank of the correct section per question (- = not returned):",
        "",
        "| Id | Question | Expected section | Vector | Full-text | Hybrid |",
        "|---|---|---|---|---|---|",
    ]
    for i, q in enumerate(questions):
        ranks = [results[m].ranks[i] for m in MODES]
        cells = " | ".join("-" if r is None else str(r) for r in ranks)
        lines.append(f"| {q.id} | {q.question} | `{q.section}` | {cells} |")
    return "\n".join(lines) + "\n"


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recall@1, recall@5 and MRR per search mode.")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--output", type=Path, help="also write the report to this file")
    parser.add_argument("--query-prefix", help="override EMBEDDING_QUERY_PREFIX ('' for none)")
    args = parser.parse_args(argv)

    settings = get_settings()
    embedder = embedder_from_settings(settings)
    if args.query_prefix is not None:
        embedder.query_prefix = args.query_prefix
    questions = load_questions(args.questions)
    # Every chunk is ranked, so MRR and the per-question ranks are exact, not cut off.
    retriever = PolicyRetriever(embedder, candidates=1000, rrf_k=settings.retrieval_rrf_k)
    await embedder.warm()  # keep model loading out of the latency figures
    engine = create_async_engine(settings.database_url())
    try:
        async with AsyncSession(engine) as db:
            chunks = await db.scalar(
                select(func.count())
                .select_from(PolicyChunk)
                .where(PolicyChunk.embedding_model == embedder.model_name)
            )
            results = await evaluate(db, retriever, questions)
    except IndexEmptyError:
        print(
            f"No chunks embedded with {embedder.model_name} in {settings.postgres_db}. "
            "Run: python -m app.retrieval.ingest",
            file=sys.stderr,
        )
        return 1
    finally:
        await engine.dispose()
    header = {
        "Run": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "Embedding model": f"`{embedder.model_name}`",
        "Query prefix": f"`{embedder.query_prefix}`" if embedder.query_prefix else "none",
        "Index": f"{chunks} sections from docs/policies/",
        "Questions": f"{len(questions)} ({args.questions.name})",
        "Fusion": f"RRF, k = {retriever.rrf_k}",
        "Latency": "per query, in process: query embedding (vector and hybrid) and SQL",
    }
    text = report(results, questions, header=header)
    print(text)
    if args.output:
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

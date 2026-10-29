"""Load docs/policies/*.md into policy_chunks.

    python -m app.retrieval.ingest           # embed new or changed documents only
    python -m app.retrieval.ingest --force   # re-embed everything

Per document, all in one transaction: unchanged (same text hash and embedding model) is
skipped; changed is replaced; a document no longer on disk is deleted. So the table always
matches the files, and searches never mix two versions of a document or two models.
"""

import argparse
import asyncio
import sys
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.config import get_settings
from app.db.models import PolicyChunk
from app.retrieval.chunking import Chunk, load_documents
from app.retrieval.embedding import Embedder, embedder_from_settings


@dataclass
class IngestReport:
    embedded: list[str] = field(default_factory=list)  # documents (re-)embedded
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    chunks: int = 0  # chunks written


async def ingest(
    db: AsyncSession, embedder: Embedder, chunks: list[Chunk], *, force: bool = False
) -> IngestReport:
    report = IngestReport()
    stored: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for document, version, model in await db.execute(
        select(PolicyChunk.document, PolicyChunk.version, PolicyChunk.embedding_model).distinct()
    ):
        stored[document].add((version, model))

    by_document: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_document[chunk.document].append(chunk)

    for document, doc_chunks in by_document.items():
        if not force and stored.get(document) == {(doc_chunks[0].version, embedder.model_name)}:
            report.unchanged.append(document)
            continue
        vectors = await asyncio.to_thread(
            embedder.embed_documents, [c.embedding_text() for c in doc_chunks]
        )
        await db.execute(delete(PolicyChunk).where(PolicyChunk.document == document))
        db.add_all(
            PolicyChunk(
                document=c.document,
                title=c.title,
                section=c.section,
                section_id=c.section_id,
                position=c.position,
                content=c.content,
                version=c.version,
                embedding_model=embedder.model_name,
                embedding=vector,
            )
            for c, vector in zip(doc_chunks, vectors, strict=True)
        )
        report.embedded.append(document)
        report.chunks += len(doc_chunks)

    report.removed = sorted(set(stored) - set(by_document))
    if report.removed:
        await db.execute(delete(PolicyChunk).where(PolicyChunk.document.in_(report.removed)))
    await db.commit()
    return report


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed docs/policies/*.md into policy_chunks.")
    parser.add_argument("--force", action="store_true", help="re-embed unchanged documents too")
    args = parser.parse_args(argv)
    settings = get_settings()
    chunks = load_documents(settings.policy_docs_dir)
    engine = create_async_engine(settings.database_url())
    try:
        async with AsyncSession(engine) as db:
            report = await ingest(db, embedder_from_settings(), chunks, force=args.force)
    finally:
        await engine.dispose()
    print(
        f"{settings.postgres_db}: {len(chunks)} sections in {settings.policy_docs_dir} "
        f"(model {settings.embedding_model})\n"
        f"  embedded:  {', '.join(report.embedded) or '-'} ({report.chunks} chunks)\n"
        f"  unchanged: {', '.join(report.unchanged) or '-'}\n"
        f"  removed:   {', '.join(report.removed) or '-'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

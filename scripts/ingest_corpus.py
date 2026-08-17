"""Ingest and index benchmark corpus documents (Stage 4 support script).

The golden dataset points at real ``element_id`` values, so the corpus behind a
committed experiment has to be reproducible. This script rebuilds it from
``benchmarks/corpus/`` without going through the HTTP API, which keeps it usable
from CI and from a plain shell:

    python -m scripts.ingest_corpus                       # every corpus file
    python -m scripts.ingest_corpus staff_handbook.pdf    # named files only
    python -m scripts.ingest_corpus --list                # show what is ingested

Both ingestion and indexing are idempotent (ADR-036 deterministic chunk IDs plus
upsert-by-point-ID), so re-running is safe and cheap: an already-indexed version
re-embeds nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import select

from app.core.logging import get_logger, setup_logging
from app.db.models.document import Document
from app.db.models.version import DocumentVersion
from app.db.session import get_session_factory
from app.ingestion.chunking.service import get_chunking_service
from app.ingestion.service import get_ingestion_service
from app.retrieval.indexer import get_chunk_indexer

logger = get_logger("scripts.ingest_corpus")

CORPUS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "corpus"


async def list_ingested() -> int:
    """Print the documents already in the canonical store, with element counts."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(Document.id, Document.title, DocumentVersion.id)
                .join(DocumentVersion, DocumentVersion.document_id == Document.id)
                .order_by(Document.title)
            )
        ).all()

    if not rows:
        print("No documents ingested.")
        return 0

    for document_id, title, version_id in rows:
        print(f"{title}\n  document_id={document_id}\n  version_id={version_id}")
    return 0


async def ingest_one(path: Path) -> bool:
    """Ingest, chunk, and index one file. Returns False on failure."""
    ingestion = get_ingestion_service()
    chunking = get_chunking_service()
    indexer = get_chunk_indexer()
    session_factory = get_session_factory()

    print(f"\n=== {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    content = path.read_bytes()

    # Each step commits explicitly: outside FastAPI there is no `get_db_session`
    # dependency to commit on the way out, and an uncommitted version is invisible
    # to the chunker that runs in the next session.
    try:
        async with session_factory() as session:
            result = await ingestion.ingest_document(
                session=session,
                file_content=content,
                filename=path.name,
                metadata_dict={},
            )
            await session.commit()
        state = "already ingested" if result.is_duplicate else "ingested"
        print(
            f"  {state}: {result.total_pages} pages, {result.total_elements} elements\n"
            f"  document_id={result.document_id}\n  version_id={result.version_id}"
        )

        async with session_factory() as session:
            chunk_result = await chunking.chunk_version(
                session=session, version_id=result.version_id
            )
            await session.commit()
        print(
            f"  chunked: {chunk_result.total_chunks} chunks "
            f"({chunk_result.chunks_created} new, strategy={chunk_result.strategy})"
        )

        async with session_factory() as session:
            index_result = await indexer.index_version(
                session=session, version_id=result.version_id
            )
            await session.commit()
        print(
            f"  indexed: {index_result.chunks_embedded} embedded, "
            f"{index_result.chunks_skipped} already current, "
            f"{index_result.points_upserted} points upserted"
        )
    except Exception as exc:  # noqa: BLE001 - one bad file must not abort the corpus
        print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.exception("corpus_ingest_failed", filename=path.name)
        return False

    return True


async def main_async(args: argparse.Namespace) -> int:
    setup_logging()

    if args.list:
        return await list_ingested()

    if args.files:
        paths = [CORPUS_DIR / name for name in args.files]
        missing = [p.name for p in paths if not p.is_file()]
        if missing:
            print(f"Not found in {CORPUS_DIR}: {', '.join(missing)}", file=sys.stderr)
            return 2
    else:
        paths = sorted(CORPUS_DIR.glob("*.pdf"))

    failures = [path.name for path in paths if not await ingest_one(path)]

    print(f"\n{len(paths) - len(failures)}/{len(paths)} documents ingested and indexed.")
    if failures:
        print(f"Failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest_corpus", description=__doc__)
    parser.add_argument("files", nargs="*", help="corpus filenames (default: every PDF)")
    parser.add_argument("--list", action="store_true", help="list ingested documents and exit")
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

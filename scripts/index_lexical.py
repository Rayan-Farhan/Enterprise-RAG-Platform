"""Backfill the OpenSearch BM25 index from PostgreSQL (Task 6.1).

The corpus was chunked and vector-indexed before the lexical channel existed.
This writes every chunk of the current ``CHUNKING_VERSION`` into OpenSearch,
straight from the authoritative rows (ADR-002). It calls no model provider, so
it is free to re-run; documents are keyed on the deterministic chunk ID, so a
re-run overwrites in place and never duplicates.

    python -m scripts.index_lexical            # every version with current chunks
    python -m scripts.index_lexical --recreate # drop and rebuild the index first

``--recreate`` is for a mapping or analyzer change: OpenSearch cannot re-analyze
text already indexed, so a changed analyzer only takes effect on a fresh index.
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.db.models.chunk import Chunk
from app.db.session import get_session_factory
from app.retrieval.indexer import get_lexical_indexer
from app.retrieval.lexical_store import get_lexical_store

logger = get_logger("scripts.index_lexical")


async def main_async(args: argparse.Namespace) -> int:
    setup_logging()
    settings = get_settings()
    store = get_lexical_store()

    if args.recreate and store.client.indices.exists(index=store.index_name):
        store.client.indices.delete(index=store.index_name)
        print(f"Dropped index '{store.index_name}'.")

    session_factory = get_session_factory()
    async with session_factory() as session:
        version_ids = (
            (
                await session.execute(
                    select(Chunk.version_id)
                    .where(Chunk.chunking_version == settings.CHUNKING_VERSION)
                    .distinct()
                )
            )
            .scalars()
            .all()
        )

    if not version_ids:
        print(f"No chunks under chunking version '{settings.CHUNKING_VERSION}'.")
        return 1

    indexer = get_lexical_indexer()
    total = 0
    for version_id in version_ids:
        async with session_factory() as session:
            result = await indexer.index_version(session=session, version_id=version_id)
        total += result.documents_indexed
        print(f"  {version_id}: {result.documents_indexed} chunks")

    indexed = store.count(chunking_version=settings.CHUNKING_VERSION)
    print(
        f"\n{total} chunks written across {len(version_ids)} versions; "
        f"'{store.index_name}' now holds {indexed} for '{settings.CHUNKING_VERSION}'."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="index_lexical", description=__doc__)
    parser.add_argument(
        "--recreate", action="store_true", help="drop and rebuild the index (mapping change)"
    )
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

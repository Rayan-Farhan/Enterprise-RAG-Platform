"""Backfill the OpenSearch retrieval indexes from PostgreSQL (Tasks 6.1, 6.2).

The corpus was chunked and vector-indexed before these channels existed. This
writes every chunk of the current ``CHUNKING_VERSION`` into OpenSearch straight
from the authoritative rows (ADR-002), keyed on the deterministic chunk ID, so
a re-run overwrites in place and never duplicates.

    python -m scripts.index_lexical                 # BM25 index
    python -m scripts.index_lexical --sparse        # neural sparse index
    python -m scripts.index_lexical --recreate      # drop and rebuild first

BM25 indexing calls no model and rewrites every chunk; it is free to re-run.
Sparse indexing runs the encoder inside OpenSearch (about a second per chunk on
CPU), so it skips chunks already encoded unless ``--force``. It needs
``scripts/setup_neural_sparse.py`` to have deployed the models first.

``--recreate`` is for a mapping, analyzer or model change: OpenSearch cannot
re-analyze or re-encode documents already indexed.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.db.models.chunk import Chunk
from app.db.session import get_session_factory
from app.retrieval.indexer import get_lexical_indexer, get_sparse_indexer
from app.retrieval.lexical_store import get_lexical_store
from app.retrieval.sparse_store import get_sparse_store

logger = get_logger("scripts.index_lexical")


async def main_async(args: argparse.Namespace) -> int:
    setup_logging()
    settings = get_settings()
    store = get_sparse_store() if args.sparse else get_lexical_store()

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

    started = time.monotonic()
    total = 0
    for version_id in version_ids:
        async with session_factory() as session:
            if args.sparse:
                sparse = await get_sparse_indexer().index_version(
                    session=session, version_id=version_id, force=args.force
                )
                total += sparse.documents_encoded
                print(
                    f"  {version_id}: {sparse.documents_encoded} encoded, "
                    f"{sparse.documents_skipped} already present"
                )
            else:
                lexical = await get_lexical_indexer().index_version(
                    session=session, version_id=version_id
                )
                total += lexical.documents_indexed
                print(f"  {version_id}: {lexical.documents_indexed} chunks")

    indexed = store.count(chunking_version=settings.CHUNKING_VERSION)
    print(
        f"\n{total} chunks written across {len(version_ids)} versions in "
        f"{time.monotonic() - started:.0f}s; '{store.index_name}' now holds {indexed} "
        f"for '{settings.CHUNKING_VERSION}'."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="index_lexical", description=__doc__)
    parser.add_argument(
        "--sparse", action="store_true", help="backfill the neural sparse index instead of BM25"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-encode chunks already in the sparse index"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="drop and rebuild the index (mapping change)"
    )
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

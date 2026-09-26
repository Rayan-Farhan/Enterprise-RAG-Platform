"""Apply the curated corpus metadata and sync it into every index (Task 6.3).

The corpus was ingested without HR metadata, so metadata-driven narrowing had
nothing to narrow by. ``benchmarks/corpus/metadata.json`` supplies it per file;
this writes it to PostgreSQL (the authority) and then rewrites the copies in
Qdrant and OpenSearch in place — no re-embedding, no re-encoding.

    python -m scripts.apply_corpus_metadata

Idempotent: re-running writes the same values.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

from app.core.logging import setup_logging
from app.db.models.document import Document
from app.db.models.metadata import DocumentMetadata
from app.db.models.version import DocumentVersion
from app.db.session import get_session_factory
from app.retrieval.metadata_sync import MetadataSync

METADATA_FILE = Path(__file__).resolve().parents[1] / "benchmarks" / "corpus" / "metadata.json"
FIELDS = (
    "department",
    "policy_type",
    "policy_status",
    "country",
    "employee_type",
    "grade",
    "confidentiality",
    "audience",
)


async def main_async() -> int:
    setup_logging()
    curated: dict[str, dict[str, str]] = json.loads(METADATA_FILE.read_text("utf-8"))["documents"]
    session_factory = get_session_factory()
    sync = MetadataSync()

    async with session_factory() as session:
        rows = (
            await session.execute(
                select(Document.title, DocumentVersion.id).join(
                    DocumentVersion, DocumentVersion.document_id == Document.id
                )
            )
        ).all()
    versions = dict(rows)

    missing = sorted(set(curated) - set(versions))
    if missing:
        print(f"Not ingested, skipped: {', '.join(missing)}", file=sys.stderr)

    for title, values in sorted(curated.items()):
        version_id = versions.get(title)
        if version_id is None:
            continue
        unknown = set(values) - set(FIELDS)
        if unknown:
            print(f"{title}: unknown metadata fields {sorted(unknown)}", file=sys.stderr)
            return 2

        async with session_factory() as session:
            record = (
                await session.execute(
                    select(DocumentMetadata).where(DocumentMetadata.version_id == version_id)
                )
            ).scalar_one_or_none()
            if record is None:
                record = DocumentMetadata(version_id=version_id)
                session.add(record)
            for field in FIELDS:
                setattr(record, field, values.get(field))
            await session.commit()

        async with session_factory() as session:
            result = await sync.sync_version(session, version_id)
        print(
            f"{title}: {values.get('department')}/{values.get('policy_type')}/"
            f"{values.get('employee_type')}  "
            f"(bm25 {result.lexical_updated}, sparse {result.sparse_updated} chunks updated)"
        )
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())

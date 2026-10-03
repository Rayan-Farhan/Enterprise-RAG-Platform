"""The ingestion chain's steps over durable state (Task 7.3, ADR-018).

    parse_document → extract_pages → ocr_pages → normalize_document → chunk_document
      → generate_embeddings → index_opensearch → index_qdrant → validate_index → publish_version

Each step reads what an earlier step persisted and persists its own output, so
any step can run again, alone, in a different process: the original file, the
parser output, the page manifest, OCR results and computed vectors live in
object storage under the version's id; pages, elements and chunks live in
PostgreSQL; points live in Qdrant and OpenSearch. A step whose output already
exists does no work, which is what makes a replay cheap.

The steps wrap the Stage 2, 3 and 6 services rather than reimplementing them;
`app/workers/tasks/ingestion.py` binds each one to a Celery task.

An upload is accepted before any of this runs: the document and a `draft`
version are created up front, so every job in the chain - including the first
- belongs to a version the API can already show. The version becomes `active`
only at `publish_version`, after `validate_index` has reconciled its counts.
"""

from __future__ import annotations

import json
import mimetypes
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings, get_settings
from app.core.exceptions import NotFoundException
from app.core.logging import get_logger
from app.db.models.document import Document
from app.db.models.version import DocumentVersion, VersionStatus
from app.db.repositories.chunk_repo import ChunkRepository
from app.db.repositories.document_repo import DocumentRepository
from app.ingestion.adapters.canonical_adapter import CanonicalAdapter
from app.ingestion.chunking.service import ChunkingService
from app.ingestion.dedup import BoilerplateDetector, compute_file_sha256
from app.ingestion.parsers.base import ParsedDocument
from app.ingestion.parsers.router import FormatRouter
from app.retrieval.indexer import (
    ChunkIndexer,
    LexicalIndexer,
    MissingVectorsError,
    SparseIndexer,
)
from app.retrieval.schemas import RetrievalFilters
from app.storage.base import ObjectStorageProtocol, StoragePrefix
from app.storage.minio_service import build_storage_key, get_storage_service

logger = get_logger("app.ingestion.pipeline")

Progress = Callable[[float, str], Awaitable[None]]

# A version is created before its parser is known; parse_document replaces this.
PENDING_PARSER = "pending"


async def _no_progress(fraction: float, message: str) -> None:
    return None


def parsed_key(version_id: uuid.UUID) -> str:
    return f"{StoragePrefix.DERIVED}/{version_id}/parsed.json"


def page_manifest_key(version_id: uuid.UUID) -> str:
    return f"{StoragePrefix.PAGES}/{version_id}/manifest.json"


def ocr_result_key(version_id: uuid.UUID) -> str:
    return f"{StoragePrefix.OCR}/{version_id}/result.json"


def embeddings_key(version_id: uuid.UUID, embedding_version: str) -> str:
    return f"{StoragePrefix.DERIVED}/{version_id}/embeddings/{embedding_version}.json"


def figure_key(version_id: uuid.UUID, figure_id: str, fmt: str) -> str:
    # The same key CanonicalAdapter writes into the figure element's
    # asset_storage_key, so the element points at the bytes stored here.
    return f"{StoragePrefix.IMAGES}/{version_id}/{figure_id}.{fmt}"


class IndexValidationError(Exception):
    """A version's stores disagree on how many chunks it has; it must not be published."""


@dataclass
class AcceptedUpload:
    document_id: uuid.UUID
    version_id: uuid.UUID
    filename: str
    file_hash: str
    storage_key: str
    is_duplicate: bool


@dataclass
class StepOutcome:
    """What one step did, as a line for the job's progress message."""

    summary: str
    noop: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineServices:
    settings: AppSettings
    storage: ObjectStorageProtocol
    router: FormatRouter
    boilerplate: BoilerplateDetector
    chunking: ChunkingService
    chunk_indexer: ChunkIndexer
    lexical_indexer: LexicalIndexer
    sparse_indexer: SparseIndexer


def build_pipeline_services(settings: AppSettings | None = None) -> PipelineServices:
    """Fresh services for one job.

    Not the API's singletons: each job runs on its own event loop, and a service
    that holds loop-bound state (the embedding rate limiter's timestamps are
    harmless, but a client pool would not be) must not cross loops.
    """
    settings = settings or get_settings()
    return PipelineServices(
        settings=settings,
        storage=get_storage_service(),
        router=FormatRouter(),
        boilerplate=BoilerplateDetector(),
        chunking=ChunkingService(settings=settings),
        chunk_indexer=ChunkIndexer(settings=settings),
        lexical_indexer=LexicalIndexer(settings=settings),
        sparse_indexer=SparseIndexer(settings=settings),
    )


def _put_json(storage: ObjectStorageProtocol, key: str, payload: Any) -> None:
    storage.upload_file(
        key=key,
        data=json.dumps(payload).encode("utf-8"),
        content_type="application/json",
    )


def _get_json(storage: ObjectStorageProtocol, key: str) -> Any:
    return json.loads(storage.download_file(key))


async def _version(session: AsyncSession, version_id: uuid.UUID) -> DocumentVersion:
    version = await DocumentRepository(session).get_version_by_id(version_id)
    if version is None:
        raise NotFoundException(f"Document version '{version_id}' was not found")
    return version


def _load_parsed(storage: ObjectStorageProtocol, version_id: uuid.UUID) -> ParsedDocument:
    return ParsedDocument.model_validate_json(storage.download_file(parsed_key(version_id)))


# ── Acceptance (API request path) ────────────────────────────────────────────


async def accept_upload(
    session: AsyncSession,
    storage: ObjectStorageProtocol,
    *,
    file_content: bytes,
    filename: str,
    content_type: str | None = None,
) -> AcceptedUpload:
    """Store the original and create the document with a draft version.

    This is all the request path does: hashing and one upload. An identical file
    already in the repository returns that document instead, as the synchronous
    path did (Task 2.4).
    """
    repo = DocumentRepository(session)
    file_hash = compute_file_sha256(file_content)
    existing = await repo.get_by_hash(file_hash)
    if existing is not None:
        return _duplicate(existing, file_hash)

    storage_key = build_storage_key(StoragePrefix.ORIGINAL, file_hash, filename)
    storage.upload_file(
        key=storage_key,
        data=file_content,
        content_type="application/octet-stream",
        metadata={"filename": filename, "file_hash": file_hash},
    )

    document = Document(
        id=uuid.uuid4(),
        title=filename,
        mime_type=content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream",
        file_size_bytes=len(file_content),
        file_hash=file_hash,
        storage_key=storage_key,
    )
    version = DocumentVersion(
        id=uuid.uuid4(),
        document_id=document.id,
        version_number=1,
        status=VersionStatus.DRAFT.value,
        parser_name=PENDING_PARSER,
    )
    session.add(document)
    try:
        await session.flush()
    except IntegrityError:
        # The same file accepted concurrently by another request.
        await session.rollback()
        existing = await repo.get_by_hash(file_hash)
        if existing is None:
            raise
        return _duplicate(existing, file_hash)
    session.add(version)
    await session.flush()

    logger.info(
        "upload_accepted",
        document_id=str(document.id),
        version_id=str(version.id),
        file_hash=file_hash,
    )
    return AcceptedUpload(
        document_id=document.id,
        version_id=version.id,
        filename=filename,
        file_hash=file_hash,
        storage_key=storage_key,
        is_duplicate=False,
    )


def _duplicate(document: Document, file_hash: str) -> AcceptedUpload:
    versions = sorted(document.versions, key=lambda v: v.version_number)
    if not versions:
        raise NotFoundException(f"Document '{document.id}' has no version")
    return AcceptedUpload(
        document_id=document.id,
        version_id=versions[-1].id,
        filename=document.title,
        file_hash=file_hash,
        storage_key=document.storage_key,
        is_duplicate=True,
    )


# ── Chain steps ──────────────────────────────────────────────────────────────


async def parse_document(
    services: PipelineServices,
    session: AsyncSession,
    version_id: uuid.UUID,
    progress: Progress = _no_progress,
) -> StepOutcome:
    """Run the format router over the original and persist its output."""
    version = await _version(session, version_id)
    key = parsed_key(version_id)
    if services.storage.exists(key):
        return StepOutcome("parser output already stored", noop=True)

    document = version.document
    await progress(0.1, f"downloading {document.title}")
    content = services.storage.download_file(document.storage_key)

    await progress(0.2, "parsing")
    with tempfile.NamedTemporaryFile(suffix=Path(document.title).suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        parsed = services.router.route_and_parse(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    parsed.filename = document.title

    services.storage.upload_file(
        key=key, data=parsed.model_dump_json().encode("utf-8"), content_type="application/json"
    )
    version.parser_name = parsed.parser_name
    version.parsing_duration_ms = parsed.parsing_duration_ms
    version.total_pages = parsed.total_pages or len(parsed.pages)
    await session.commit()
    return StepOutcome(
        f"parsed {version.total_pages} pages with {parsed.parser_name}",
        details={"pages": version.total_pages, "parser": parsed.parser_name},
    )


async def extract_pages(
    services: PipelineServices,
    session: AsyncSession,
    version_id: uuid.UUID,
    progress: Progress = _no_progress,
) -> StepOutcome:
    """Store each extracted figure and a manifest of what every page yielded."""
    await _version(session, version_id)
    key = page_manifest_key(version_id)
    if services.storage.exists(key):
        return StepOutcome("page manifest already stored", noop=True)

    parsed = _load_parsed(services.storage, version_id)
    figures = 0
    for fig in parsed.all_figures:
        if fig.image_bytes:
            services.storage.upload_file(
                key=figure_key(version_id, fig.figure_id, fig.format),
                data=fig.image_bytes,
                content_type=f"image/{fig.format}",
            )
            figures += 1

    pages = []
    for page in parsed.pages:
        chars = sum(len(el.text.strip()) for el in page.elements) + sum(
            len(t.markdown or "") for t in page.tables
        )
        pages.append(
            {
                "page_number": page.page_number,
                "elements": len(page.elements),
                "tables": len(page.tables),
                "figures": len(page.figures),
                "text_chars": chars,
                # A page that yielded no text at all is a scan, an image, or blank.
                "needs_ocr": chars == 0,
            }
        )
    _put_json(services.storage, key, {"version_id": str(version_id), "pages": pages})
    needing = sum(p["needs_ocr"] for p in pages)
    return StepOutcome(
        f"{len(pages)} pages, {figures} figures stored, {needing} pages without text",
        details={"pages": len(pages), "figures": figures, "needs_ocr": needing},
    )


async def ocr_pages(
    services: PipelineServices,
    session: AsyncSession,
    version_id: uuid.UUID,
    progress: Progress = _no_progress,
) -> StepOutcome:
    """Record which pages need OCR and what was recovered.

    No OCR engine is wired in yet (Docling runs its own OCR inside parsing), so
    textless pages are recorded as unrecovered - never filled with placeholder
    text, which would be indexed and retrieved as if it were the page.
    """
    await _version(session, version_id)
    key = ocr_result_key(version_id)
    if services.storage.exists(key):
        return StepOutcome("OCR result already stored", noop=True)

    manifest = _get_json(services.storage, page_manifest_key(version_id))
    needing = [p["page_number"] for p in manifest["pages"] if p["needs_ocr"]]
    _put_json(
        services.storage,
        key,
        {"engine": None, "pages_needing_ocr": needing, "recovered": {}},
    )
    if needing:
        logger.warning(
            "pages_without_text_unrecovered",
            version_id=str(version_id),
            pages=needing[:50],
            count=len(needing),
        )
        return StepOutcome(
            f"{len(needing)} pages without text; no OCR engine configured",
            details={"unrecovered": len(needing)},
        )
    return StepOutcome("every page yielded text; nothing to OCR")


async def normalize_document(
    services: PipelineServices,
    session: AsyncSession,
    version_id: uuid.UUID,
    metadata: dict[str, Any] | None = None,
    progress: Progress = _no_progress,
) -> StepOutcome:
    """Adapt the parser output into canonical pages and elements, in one transaction."""
    version = await _version(session, version_id)
    if version.pages:
        # Pages and elements are written in a single commit, so any page means all.
        return StepOutcome("canonical pages already persisted", noop=True)

    parsed = _load_parsed(services.storage, version_id)
    document = version.document
    _, adapted, pages, elements, metadata_record = CanonicalAdapter.to_canonical_models(
        parsed_doc=parsed,
        file_hash=document.file_hash,
        storage_key=document.storage_key,
        file_size_bytes=document.file_size_bytes,
        metadata_dict=metadata,
        status=version.status,
        document_id=document.id,
        version_id=version.id,
    )
    services.boilerplate.detect_and_flag(elements, total_pages=len(pages))

    document.mime_type = parsed.file_type
    if metadata:
        document.external_id = metadata.get("external_id")
        document.source_priority = metadata.get("source_priority", document.source_priority)
    version.total_elements = adapted.total_elements
    version.effective_from = adapted.effective_from
    version.effective_until = adapted.effective_until
    version.authority = adapted.authority

    await DocumentRepository(session).save_version_content(
        version, pages, elements, metadata_record
    )
    await session.commit()
    boilerplate = sum(1 for e in elements if e.is_boilerplate)
    return StepOutcome(
        f"{len(pages)} pages, {len(elements)} elements ({boilerplate} boilerplate)",
        details={"pages": len(pages), "elements": len(elements), "boilerplate": boilerplate},
    )


async def chunk_document(
    services: PipelineServices,
    session: AsyncSession,
    version_id: uuid.UUID,
    progress: Progress = _no_progress,
) -> StepOutcome:
    result = await services.chunking.chunk_version(session=session, version_id=version_id)
    await session.commit()
    return StepOutcome(
        f"{result.total_chunks} chunks ({result.chunks_created} new, "
        f"{result.chunks_updated} updated, {result.chunks_removed} removed)",
        noop=result.is_noop,
        details={"chunks": result.total_chunks, "chunking_version": result.chunking_version},
    )


async def generate_embeddings(
    services: PipelineServices,
    session: AsyncSession,
    version_id: uuid.UUID,
    progress: Progress = _no_progress,
) -> StepOutcome:
    """Embed every chunk lacking a current vector and persist the vectors.

    Vectors already persisted for a chunk's current content are reused, so a
    replay after a crash - or after index_qdrant failed - makes no provider call.
    """
    indexer = services.chunk_indexer
    embedding_version = services.settings.effective_embedding_version
    key = embeddings_key(version_id, embedding_version)

    stored: dict[str, Any] = (
        _get_json(services.storage, key) if services.storage.exists(key) else {"vectors": {}}
    )
    chunks = await ChunkRepository(session).list_by_version(
        version_id, services.settings.CHUNKING_VERSION
    )
    hashes = {c.id: _content_hash(c.content) for c in chunks}
    reusable = {
        uuid.UUID(cid): entry["vector"]
        for cid, entry in stored["vectors"].items()
        if hashes.get(uuid.UUID(cid)) == entry["content_hash"]
    }

    await progress(0.1, f"embedding up to {len(chunks)} chunks")
    pending = await indexer.embed_pending(session, version_id, reusable=reusable)
    if not pending.chunks:
        return StepOutcome("every chunk already has a current vector", noop=True)

    embedded_now = len(pending.chunks) - pending.chunks_reused
    if embedded_now:
        _put_json(
            services.storage,
            key,
            {
                "embedding_version": embedding_version,
                "dimensions": pending.dimensions,
                "provider": pending.provider,
                "vectors": {
                    str(chunk.id): {
                        "content_hash": hashes[chunk.id],
                        "vector": pending.vectors[chunk.id],
                    }
                    for chunk in pending.chunks
                },
            },
        )
    return StepOutcome(
        f"{embedded_now} chunks embedded, {pending.chunks_reused} reused from an earlier attempt",
        noop=embedded_now == 0,
        details={"embedded": embedded_now, "reused": pending.chunks_reused},
    )


async def index_opensearch(
    services: PipelineServices,
    session: AsyncSession,
    version_id: uuid.UUID,
    progress: Progress = _no_progress,
) -> StepOutcome:
    """Write the version's chunks to the BM25 and neural sparse indexes, as enabled."""
    settings = services.settings
    parts: list[str] = []
    if settings.ENABLE_LEXICAL_INDEXING:
        await progress(0.1, "writing BM25 index")
        lexical = await services.lexical_indexer.index_version(session, version_id)
        parts.append(f"{lexical.documents_indexed} BM25 documents")
    if settings.ENABLE_NEURAL_SPARSE:
        await progress(0.5, "encoding neural sparse index")
        sparse = await services.sparse_indexer.index_version(session, version_id)
        parts.append(
            f"{sparse.documents_encoded} sparse-encoded, {sparse.documents_skipped} already present"
        )
    if not parts:
        return StepOutcome("lexical and sparse indexing are both disabled", noop=True)
    return StepOutcome(", ".join(parts))


async def index_qdrant(
    services: PipelineServices,
    session: AsyncSession,
    version_id: uuid.UUID,
    progress: Progress = _no_progress,
) -> StepOutcome:
    """Upsert the persisted vectors as points and record each chunk's point."""
    embedding_version = services.settings.effective_embedding_version
    key = embeddings_key(version_id, embedding_version)
    stored = _get_json(services.storage, key) if services.storage.exists(key) else {"vectors": {}}
    vectors = {uuid.UUID(cid): entry["vector"] for cid, entry in stored["vectors"].items()}

    # Only generate_embeddings may spend provider calls; a replay of this step
    # must never re-bill, so a chunk without a stored vector is an error here.
    try:
        pending = await services.chunk_indexer.embed_pending(
            session, version_id, reusable=vectors, embed_missing=False
        )
    except MissingVectorsError as exc:
        raise MissingVectorsError(f"{exc}; run generate_embeddings first") from exc
    pending.provider = stored.get("provider", pending.provider)
    result = await services.chunk_indexer.upsert_embedded(session, pending)
    await session.commit()
    return StepOutcome(
        f"{result.points_upserted} points upserted, {result.chunks_skipped} already current",
        noop=result.points_upserted == 0,
        details={"upserted": result.points_upserted},
    )


async def validate_index(
    services: PipelineServices,
    session: AsyncSession,
    version_id: uuid.UUID,
    progress: Progress = _no_progress,
) -> StepOutcome:
    """Reconcile the version's chunk count across PostgreSQL and every index."""
    settings = services.settings
    chunking_version = settings.CHUNKING_VERSION
    embedding_version = settings.effective_embedding_version
    chunks = await ChunkRepository(session).list_by_version(version_id, chunking_version)
    expected = len(chunks)
    only_this_version = RetrievalFilters(version_ids=[version_id])

    counts: dict[str, int] = {
        "chunks with a current vector": sum(
            1 for c in chunks if c.embedding_id and c.embedding_version == embedding_version
        ),
        "qdrant points": services.chunk_indexer.vector_store.count_matching(
            only_this_version, chunking_version, embedding_version
        ),
    }
    if settings.ENABLE_LEXICAL_INDEXING:
        counts["bm25 documents"] = services.lexical_indexer.lexical_store.count(
            chunking_version, only_this_version
        )
    if settings.ENABLE_NEURAL_SPARSE:
        counts["sparse documents"] = services.sparse_indexer.sparse_store.count(
            chunking_version, only_this_version
        )

    if expected == 0:
        raise IndexValidationError("version has no chunks; nothing would be retrievable")
    mismatched = {name: n for name, n in counts.items() if n != expected}
    if mismatched:
        detail = ", ".join(f"{name} {n}" for name, n in mismatched.items())
        raise IndexValidationError(f"expected {expected} chunks everywhere; found {detail}")
    return StepOutcome(
        f"{expected} chunks reconcile across {len(counts)} stores",
        details={"chunks": expected, **counts},
    )


async def publish_version(
    services: PipelineServices,
    session: AsyncSession,
    version_id: uuid.UUID,
    progress: Progress = _no_progress,
) -> StepOutcome:
    """Mark the draft version active.

    A conditional update, so a replay finds nothing to flip. Superseding earlier
    versions and lock-guarding activation are Task 7.9's.
    """
    result = await session.execute(
        update(DocumentVersion)
        .where(
            DocumentVersion.id == version_id,
            DocumentVersion.status == VersionStatus.DRAFT.value,
        )
        .values(status=VersionStatus.ACTIVE.value)
    )
    await session.commit()
    if result.rowcount == 0:  # type: ignore[attr-defined]
        version = await _version(session, version_id)
        return StepOutcome(f"version already {version.status}", noop=True)
    return StepOutcome("version published")


def _content_hash(text: str) -> str:
    return compute_file_sha256(text.encode("utf-8"))

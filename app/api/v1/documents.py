"""Documents management and ingestion router (Task 2.6, ADR-002, ADR-003, ADR-005)."""

import json
import uuid
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.idempotency import (
    IDEMPOTENCY_HEADER,
    IdempotencyService,
    fingerprint,
    get_idempotency_service,
    run_idempotently,
)
from app.api.v1.schemas.documents import (
    DocumentDetailResponse,
    DocumentIngestAccepted,
    DocumentListItem,
    DocumentListResponse,
    DocumentMetadataResponse,
    DocumentVersionResponse,
    ElementResponse,
    IndexVersionResponse,
)
from app.core.config import get_settings
from app.core.exceptions import NotFoundException, ValidationException
from app.core.logging import get_logger
from app.db.models.job import JobStatus
from app.db.models.version import VersionStatus
from app.db.repositories.document_repo import DocumentRepository
from app.db.session import get_db_session
from app.ingestion.chunking.service import ChunkingService, get_chunking_service
from app.ingestion.dedup import compute_file_sha256
from app.ingestion.pipeline import AcceptedUpload, accept_upload
from app.jobs.service import JobAlreadyActive, JobService, get_job_service
from app.retrieval.indexer import (
    ChunkIndexer,
    LexicalIndexer,
    SparseIndexer,
    get_chunk_indexer,
    get_lexical_indexer,
    get_sparse_indexer,
)
from app.storage.base import ObjectStorageProtocol
from app.storage.minio_service import get_storage_service
from app.workers.tasks.ingestion import start_ingestion

logger = get_logger("app.api.documents")
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "",
    response_model=DocumentIngestAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Accept a document and start its ingestion chain",
)
async def ingest_document(
    response: Response,
    file: UploadFile = File(..., description="Document file to parse and ingest"),
    metadata: str | None = Form(
        default=None,
        description="Optional JSON-encoded string containing DocumentMetadataInput fields",
    ),
    session: AsyncSession = Depends(get_db_session),
    storage: ObjectStorageProtocol = Depends(get_storage_service),
    jobs: JobService = Depends(get_job_service),
    idempotency: IdempotencyService = Depends(get_idempotency_service),
    idempotency_key: str | None = Header(
        default=None,
        alias=IDEMPOTENCY_HEADER,
        description="Retry-safe key: a repeat with the same key returns the first response",
    ),
) -> DocumentIngestAccepted:
    """Store the upload and enqueue its ingestion (Task 7.3).

    The request does no parsing: it stores the original, creates the document
    with a draft version, and returns the first job's id. Follow the chain at
    `GET /documents/{id}/jobs`; the version becomes active once every step has
    run. An identical file already in the repository returns 200 with that
    document and no new job. With an `Idempotency-Key`, a retry returns the
    first attempt's response, job id included (Task 7.4).
    """
    settings = get_settings()

    if not file.filename:
        raise ValidationException("Upload file must have a valid filename")

    content = await file.read()
    if len(content) == 0:
        raise ValidationException("Uploaded file is empty (0 bytes)")

    max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_size_bytes:
        raise ValidationException(
            f"File exceeds maximum allowed upload size of {settings.MAX_UPLOAD_SIZE_MB}MB"
        )

    # Parse metadata JSON payload if provided
    metadata_dict: dict[str, Any] = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise ValidationException(f"Invalid JSON metadata payload: {exc}") from exc

    async def accept() -> DocumentIngestAccepted:
        accepted = await accept_upload(
            session,
            storage,
            file_content=content,
            filename=file.filename or "",
            content_type=file.content_type,
        )
        # The job row references the document, so the document must be
        # committed before a worker - in another process - can look either up.
        await session.commit()
        if accepted.is_duplicate:
            return await _answer_duplicate(accepted, metadata_dict, response, jobs)
        job = await start_ingestion(accepted.document_id, accepted.version_id, metadata_dict, jobs)
        return _accepted_response(
            accepted, job.id, "Accepted; ingestion is running. Follow it at /documents/{id}/jobs."
        )

    return await run_idempotently(
        idempotency,
        operation="documents.ingest",
        key=idempotency_key,
        request_fingerprint=fingerprint(compute_file_sha256(content), file.filename, metadata_dict),
        response=response,
        default_status=status.HTTP_202_ACCEPTED,
        model=DocumentIngestAccepted,
        run=accept,
    )


def _accepted_response(
    accepted: AcceptedUpload, job_id: uuid.UUID | None, message: str
) -> DocumentIngestAccepted:
    return DocumentIngestAccepted(
        document_id=accepted.document_id,
        version_id=accepted.version_id,
        job_id=job_id,
        filename=accepted.filename,
        file_hash=accepted.file_hash,
        storage_key=accepted.storage_key,
        is_duplicate=accepted.is_duplicate,
        message=message,
    )


async def _answer_duplicate(
    accepted: AcceptedUpload,
    metadata: dict[str, Any],
    response: Response,
    jobs: JobService,
) -> DocumentIngestAccepted:
    """An identical file is already stored: point at it, starting its chain if it never ran.

    A draft version none of whose jobs ever ran is an upload whose request
    failed between storing the document and handing the first step to a
    worker (a broker outage, a timeout, a killed process). Uploading the file
    again is the natural retry, so it starts the chain rather than returning a
    document that will never be processed.
    """
    jobs_so_far = await jobs.list_for_version(accepted.version_id)
    if (
        accepted.version_status == VersionStatus.DRAFT
        and all(j.attempt == 0 for j in jobs_so_far)
        and not any(j.status in (JobStatus.QUEUED, JobStatus.RUNNING) for j in jobs_so_far)
    ):
        try:
            job = await start_ingestion(accepted.document_id, accepted.version_id, metadata, jobs)
        except JobAlreadyActive as exc:
            # A concurrent retry of the same upload started it first.
            job = exc.job
        return _accepted_response(
            accepted, job.id, "Identical file was stored but never processed; ingestion started."
        )

    response.status_code = status.HTTP_200_OK
    return _accepted_response(
        accepted, None, "Identical file already ingested; returned existing document reference."
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List ingested documents with metadata filtering",
)
async def list_documents(
    limit: int = Query(default=50, ge=1, le=100, description="Max documents to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    department: str | None = Query(default=None, description="Filter by HR department"),
    policy_type: str | None = Query(default=None, description="Filter by policy type"),
    policy_status: str | None = Query(default=None, description="Filter by policy status"),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentListResponse:
    """Retrieve paginated document entries with optional HR metadata filtering."""
    repo = DocumentRepository(session)
    docs, total = await repo.list_documents(
        limit=limit,
        offset=offset,
        department=department,
        policy_type=policy_type,
        policy_status=policy_status,
    )

    items: list[DocumentListItem] = []
    for doc in docs:
        latest_ver = doc.versions[0] if doc.versions else None
        meta = latest_ver.metadata_record if latest_ver else None

        items.append(
            DocumentListItem(
                id=doc.id,
                title=doc.title,
                mime_type=doc.mime_type,
                file_size_bytes=doc.file_size_bytes,
                file_hash=doc.file_hash,
                storage_key=doc.storage_key,
                latest_version=latest_ver.version_number if latest_ver else 1,
                total_pages=latest_ver.total_pages if latest_ver else 0,
                total_elements=latest_ver.total_elements if latest_ver else 0,
                department=meta.department if meta else None,
                policy_type=meta.policy_type if meta else None,
                created_at=doc.created_at,
            )
        )

    return DocumentListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details by ID",
)
async def get_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
) -> DocumentDetailResponse:
    """Retrieve complete metadata and version snapshots for a specific document."""
    repo = DocumentRepository(session)
    doc = await repo.get_by_id(document_id)
    if doc is None:
        raise NotFoundException(f"Document with ID '{document_id}' was not found")

    versions_resp: list[DocumentVersionResponse] = []
    for v in doc.versions:
        meta_resp: DocumentMetadataResponse | None = None
        if v.metadata_record:
            meta_resp = DocumentMetadataResponse(
                department=v.metadata_record.department,
                policy_type=v.metadata_record.policy_type,
                policy_status=v.metadata_record.policy_status,
                country=v.metadata_record.country,
                location=v.metadata_record.location,
                employee_type=v.metadata_record.employee_type,
                grade=v.metadata_record.grade,
                confidentiality=v.metadata_record.confidentiality,
                audience=v.metadata_record.audience,
                custom_attributes=v.metadata_record.custom_attributes,
            )

        versions_resp.append(
            DocumentVersionResponse(
                id=v.id,
                version_number=v.version_number,
                status=v.status,
                total_pages=v.total_pages,
                total_elements=v.total_elements,
                parser_name=v.parser_name,
                parsing_duration_ms=v.parsing_duration_ms,
                effective_from=v.effective_from,
                effective_until=v.effective_until,
                authority=v.authority,
                metadata=meta_resp,
                created_at=v.created_at,
            )
        )

    return DocumentDetailResponse(
        id=doc.id,
        external_id=doc.external_id,
        title=doc.title,
        mime_type=doc.mime_type,
        file_size_bytes=doc.file_size_bytes,
        file_hash=doc.file_hash,
        storage_key=doc.storage_key,
        source_priority=doc.source_priority,
        versions=versions_resp,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.get(
    "/{document_id}/versions/{version_id}/elements",
    response_model=list[ElementResponse],
    summary="Get canonical elements for a document version",
)
async def get_version_elements(
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    limit: int = Query(default=100, ge=1, le=500, description="Max elements to fetch"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    include_boilerplate: bool = Query(
        default=True, description="Include flagged boilerplate elements"
    ),
    session: AsyncSession = Depends(get_db_session),
) -> list[ElementResponse]:
    """Retrieve atomic canonical elements for a given document version."""
    repo = DocumentRepository(session)
    elements = await repo.get_elements_by_version(
        version_id=version_id,
        limit=limit,
        offset=offset,
        include_boilerplate=include_boilerplate,
    )

    return [
        ElementResponse(
            id=el.id,
            element_id=el.element_id,
            parent_id=el.parent_id,
            element_type=el.element_type,
            sequence_index=el.sequence_index,
            page_number=el.page_number,
            text_content=el.text_content,
            content_hash=el.content_hash,
            bounding_box=el.bounding_box,
            table_data=el.table_data,
            asset_storage_key=el.asset_storage_key,
            source_uri=el.source_uri,
            is_boilerplate=el.is_boilerplate,
            boilerplate_reason=el.boilerplate_reason,
        )
        for el in elements
    ]


@router.post(
    "/{document_id}/versions/{version_id}/index",
    response_model=IndexVersionResponse,
    summary="Chunk and index a document version for retrieval",
)
async def index_document_version(
    response: Response,
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    force: bool = Query(
        default=False,
        description="Re-embed every chunk even if already indexed under the current version",
    ),
    session: AsyncSession = Depends(get_db_session),
    chunking_service: ChunkingService = Depends(get_chunking_service),
    indexer: ChunkIndexer = Depends(get_chunk_indexer),
    lexical_indexer: LexicalIndexer = Depends(get_lexical_indexer),
    sparse_indexer: SparseIndexer = Depends(get_sparse_indexer),
    idempotency: IdempotencyService = Depends(get_idempotency_service),
    idempotency_key: str | None = Header(
        default=None,
        alias=IDEMPOTENCY_HEADER,
        description="Retry-safe key: a repeat with the same key returns the first response",
    ),
) -> IndexVersionResponse:
    """Chunk a persisted version and index its chunks into the vector store.

    Both steps are idempotent: chunk IDs are deterministic (ADR-036) and vector
    points are upserted by a deterministic point ID, so re-running this endpoint
    creates zero duplicate chunks and zero duplicate points. Stage 7 replaces this
    synchronous endpoint with the Celery chain while reusing these same functions.
    """

    async def index() -> IndexVersionResponse:
        chunking = await chunking_service.chunk_version(session=session, version_id=version_id)
        if chunking.document_id != document_id:
            raise ValidationException(
                f"Version '{version_id}' does not belong to document '{document_id}'"
            )

        indexing = await indexer.index_version(session=session, version_id=version_id, force=force)
        lexical_indexed = 0
        if indexer.settings.ENABLE_LEXICAL_INDEXING:
            lexical = await lexical_indexer.index_version(session=session, version_id=version_id)
            lexical_indexed = lexical.documents_indexed
        sparse_encoded = 0
        if indexer.settings.ENABLE_NEURAL_SPARSE:
            sparse = await sparse_indexer.index_version(
                session=session, version_id=version_id, force=force
            )
            sparse_encoded = sparse.documents_encoded

        # Committed before the response is stored: a replayed answer must
        # describe work that exists.
        await session.commit()
        return IndexVersionResponse(
            document_id=document_id,
            version_id=version_id,
            strategy=chunking.strategy,
            chunking_version=chunking.chunking_version,
            chunks_created=chunking.chunks_created,
            chunks_updated=chunking.chunks_updated,
            chunks_removed=chunking.chunks_removed,
            total_chunks=chunking.total_chunks,
            total_tokens=chunking.total_tokens,
            chunks_embedded=indexing.chunks_embedded,
            chunks_already_indexed=indexing.chunks_skipped,
            points_upserted=indexing.points_upserted,
            embedding_version=indexing.embedding_version,
            embedding_provider=indexing.provider,
            embedding_dimensions=indexing.dimensions,
            rate_limit_waits=indexing.rate_limit_waits,
            lexical_documents_indexed=lexical_indexed,
            sparse_documents_encoded=sparse_encoded,
            was_noop=chunking.is_noop and indexing.is_noop,
        )

    return await run_idempotently(
        idempotency,
        operation="documents.index_version",
        key=idempotency_key,
        request_fingerprint=fingerprint(document_id, version_id, force),
        response=response,
        default_status=status.HTTP_200_OK,
        model=IndexVersionResponse,
        run=index,
    )


@router.get(
    "/{document_id}/presigned-url",
    summary="Generate presigned download URL for raw document in object storage",
)
async def get_document_presigned_url(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
    storage: ObjectStorageProtocol = Depends(get_storage_service),
) -> dict[str, str]:
    """Generate a temporary presigned URL for viewing the raw source file."""
    repo = DocumentRepository(session)
    doc = await repo.get_by_id(document_id)
    if doc is None:
        raise NotFoundException(f"Document with ID '{document_id}' was not found")

    url = storage.get_presigned_url(doc.storage_key, expires_in_seconds=3600)
    return {"storage_key": doc.storage_key, "presigned_url": url}

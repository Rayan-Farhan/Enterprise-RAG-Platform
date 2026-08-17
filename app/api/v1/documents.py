"""Documents management and ingestion router (Task 2.6, ADR-002, ADR-003, ADR-005)."""

import json
import uuid
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.documents import (
    DocumentDetailResponse,
    DocumentIngestResponse,
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
from app.db.repositories.document_repo import DocumentRepository
from app.db.session import get_db_session
from app.ingestion.chunking.service import ChunkingService, get_chunking_service
from app.ingestion.service import IngestionService, get_ingestion_service
from app.retrieval.indexer import ChunkIndexer, get_chunk_indexer
from app.storage.base import ObjectStorageProtocol
from app.storage.minio_service import get_storage_service

logger = get_logger("app.api.documents")
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document into the canonical repository",
)
async def ingest_document(
    response: Response,
    file: UploadFile = File(..., description="Document file to parse and ingest"),
    metadata: str | None = Form(
        default=None,
        description="Optional JSON-encoded string containing DocumentMetadataInput fields",
    ),
    session: AsyncSession = Depends(get_db_session),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> DocumentIngestResponse:
    """Ingest a source document.

    Uploads to object storage, parses via format router, maps into canonical models,
    detects boilerplate/duplicates, and persists in PostgreSQL.
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

    result = await ingestion_service.ingest_document(
        session=session,
        file_content=content,
        filename=file.filename,
        metadata_dict=metadata_dict,
    )

    if result.is_duplicate:
        response.status_code = status.HTTP_200_OK

    return DocumentIngestResponse(
        document_id=result.document_id,
        version_id=result.version_id,
        filename=result.filename,
        file_hash=result.file_hash,
        storage_key=result.storage_key,
        total_pages=result.total_pages,
        total_elements=result.total_elements,
        is_duplicate=result.is_duplicate,
        created_at=result.created_at,
        message=result.message,
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
    include_boilerplate: bool = Query(default=True, description="Include flagged boilerplate elements"),
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
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    force: bool = Query(
        default=False,
        description="Re-embed every chunk even if already indexed under the current version",
    ),
    session: AsyncSession = Depends(get_db_session),
    chunking_service: ChunkingService = Depends(get_chunking_service),
    indexer: ChunkIndexer = Depends(get_chunk_indexer),
) -> IndexVersionResponse:
    """Chunk a persisted version and index its chunks into the vector store.

    Both steps are idempotent: chunk IDs are deterministic (ADR-036) and vector
    points are upserted by a deterministic point ID, so re-running this endpoint
    creates zero duplicate chunks and zero duplicate points. Stage 7 replaces this
    synchronous endpoint with the Celery chain while reusing these same functions.
    """
    chunking = await chunking_service.chunk_version(session=session, version_id=version_id)
    if chunking.document_id != document_id:
        raise ValidationException(
            f"Version '{version_id}' does not belong to document '{document_id}'"
        )

    indexing = await indexer.index_version(session=session, version_id=version_id, force=force)

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
        was_noop=chunking.is_noop and indexing.is_noop,
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


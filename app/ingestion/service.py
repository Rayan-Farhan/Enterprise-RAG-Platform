"""Synchronous Ingestion Service coordinating storage, parsing, canonical adaptation, and persistence (ADR-003, ADR-005, Task 2.6)."""

import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.repositories.document_repo import DocumentRepository
from app.ingestion.adapters.canonical_adapter import CanonicalAdapter
from app.ingestion.dedup import BoilerplateDetector, compute_file_sha256
from app.ingestion.parsers.base import DocumentParser, ParsedDocument
from app.ingestion.parsers.router import FormatRouter
from app.storage.base import ObjectStorageProtocol, StoragePrefix
from app.storage.minio_service import build_storage_key, get_storage_service

logger = get_logger("app.ingestion.service")


@dataclass
class IngestionResult:
    """Outcome of document ingestion."""

    document_id: uuid.UUID
    version_id: uuid.UUID
    filename: str
    file_hash: str
    storage_key: str
    total_pages: int
    total_elements: int
    is_duplicate: bool
    created_at: datetime
    message: str = "Document successfully ingested"


class IngestionService:
    """Coordinates parsing, deduplication, object storage, and canonical database persistence."""

    def __init__(
        self,
        storage_service: ObjectStorageProtocol | None = None,
        format_router: FormatRouter | None = None,
        boilerplate_detector: BoilerplateDetector | None = None,
    ) -> None:
        self.storage = storage_service or get_storage_service()
        self.router = format_router or FormatRouter()
        self.boilerplate_detector = boilerplate_detector or BoilerplateDetector()

    async def ingest_document(
        self,
        session: AsyncSession,
        file_content: bytes,
        filename: str,
        metadata_dict: dict[str, Any] | None = None,
        parser_override: DocumentParser | None = None,
    ) -> IngestionResult:
        """Ingest a document file end-to-end synchronously."""
        repo = DocumentRepository(session)
        log = logger.bind(filename=filename, size=len(file_content))

        # 1. Deduplication Check (Exact File Hash)
        file_hash = compute_file_sha256(file_content)
        log = log.bind(file_hash=file_hash)
        existing_doc = await repo.get_by_hash(file_hash)

        if existing_doc is not None:
            log.info("exact_duplicate_short_circuit", existing_id=str(existing_doc.id))
            active_version = existing_doc.versions[0] if existing_doc.versions else None
            version_id = active_version.id if active_version else uuid.uuid4()
            total_pages = active_version.total_pages if active_version else 0
            total_elements = active_version.total_elements if active_version else 0

            return IngestionResult(
                document_id=existing_doc.id,
                version_id=version_id,
                filename=existing_doc.title,
                file_hash=file_hash,
                storage_key=existing_doc.storage_key,
                total_pages=total_pages,
                total_elements=total_elements,
                is_duplicate=True,
                created_at=existing_doc.created_at,
                message="Identical file already ingested; returned existing document reference.",
            )

        # 2. Upload raw source file to Object Storage under original/ prefix (ADR-003)
        storage_key = build_storage_key(StoragePrefix.ORIGINAL, file_hash, filename)
        self.storage.upload_file(
            key=storage_key,
            data=file_content,
            content_type="application/octet-stream",
            metadata={"filename": filename, "file_hash": file_hash},
        )
        log.info("uploaded_raw_source_to_storage", key=storage_key)

        # 3. Write temp file and execute parser
        file_suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = Path(tmp_file.name)

        try:
            if parser_override is not None:
                parsed_doc: ParsedDocument = parser_override.parse(tmp_path)
            else:
                parsed_doc = self.router.route_and_parse(tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        # Update filename if parser lost it
        parsed_doc.filename = filename

        # 4. Upload extracted figure image bytes to object storage if any
        for fig in parsed_doc.all_figures:
            if fig.image_bytes:
                fig_key = f"images/{file_hash}_{fig.figure_id}.{fig.format}"
                try:
                    self.storage.upload_file(
                        key=fig_key,
                        data=fig.image_bytes,
                        content_type=f"image/{fig.format}",
                    )
                except Exception as e:
                    log.warning("failed_to_upload_extracted_figure", figure_id=fig.figure_id, error=str(e))

        # 5. Adapt ParsedDocument into Canonical Database Models
        doc, version, pages, elements, metadata_entity = CanonicalAdapter.to_canonical_models(
            parsed_doc=parsed_doc,
            file_hash=file_hash,
            storage_key=storage_key,
            file_size_bytes=len(file_content),
            metadata_dict=metadata_dict,
        )

        # 6. Flag Boilerplate elements in-place (Master Plan §10)
        self.boilerplate_detector.detect_and_flag(elements, total_pages=len(pages))

        # 7. Persist Canonical Hierarchy atomically in DB Transaction
        await repo.save_document_hierarchy(
            document=doc,
            version=version,
            pages=pages,
            elements=elements,
            metadata_record=metadata_entity,
        )

        log.info(
            "document_ingestion_complete",
            document_id=str(doc.id),
            version_id=str(version.id),
            pages=len(pages),
            elements=len(elements),
        )

        return IngestionResult(
            document_id=doc.id,
            version_id=version.id,
            filename=doc.title,
            file_hash=file_hash,
            storage_key=storage_key,
            total_pages=len(pages),
            total_elements=len(elements),
            is_duplicate=False,
            created_at=doc.created_at or datetime.now(UTC),
            message="Document successfully parsed, adapted, and persisted.",
        )


_ingestion_service: IngestionService | None = None


def get_ingestion_service() -> IngestionService:
    """Return singleton IngestionService."""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service

"""Document repository implementing transactional CRUD and hierarchy loading (ADR-002, ADR-005)."""

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.document import Document
from app.db.models.element import Element
from app.db.models.metadata import DocumentMetadata
from app.db.models.page import Page
from app.db.models.version import DocumentVersion


class DocumentRepository:
    """Async repository for Document and associated hierarchical entities."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_hash(self, file_hash: str) -> Document | None:
        """Find an existing document by its SHA-256 content hash."""
        stmt = (
            select(Document)
            .options(
                selectinload(Document.versions).selectinload(DocumentVersion.metadata_record),
            )
            .where(Document.file_hash == file_hash)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        """Retrieve a document by ID with versions and metadata."""
        stmt = (
            select(Document)
            .options(
                selectinload(Document.versions).selectinload(DocumentVersion.metadata_record),
                selectinload(Document.versions).selectinload(DocumentVersion.pages),
            )
            .where(Document.id == document_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_version_by_id(self, version_id: uuid.UUID) -> DocumentVersion | None:
        """Retrieve a specific document version with its metadata and pages."""
        stmt = (
            select(DocumentVersion)
            .options(
                selectinload(DocumentVersion.document),
                selectinload(DocumentVersion.metadata_record),
                selectinload(DocumentVersion.pages),
            )
            .where(DocumentVersion.id == version_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_elements_by_version(
        self,
        version_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
        include_boilerplate: bool = True,
    ) -> list[Element]:
        """Fetch canonical elements for a version with optional boilerplate filtering."""
        stmt = (
            select(Element)
            .where(Element.version_id == version_id)
            .order_by(Element.sequence_index.asc())
            .limit(limit)
            .offset(offset)
        )
        if not include_boilerplate:
            stmt = stmt.where(Element.is_boilerplate.is_(False))

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_all_elements_by_version(
        self,
        version_id: uuid.UUID,
        include_boilerplate: bool = True,
    ) -> list[Element]:
        """Fetch every canonical element for a version in reading order.

        Chunking must see the whole version, so this deliberately has no limit
        unlike :meth:`get_elements_by_version`, which backs a paginated endpoint.
        """
        stmt = (
            select(Element)
            .where(Element.version_id == version_id)
            .order_by(Element.sequence_index.asc())
        )
        if not include_boilerplate:
            stmt = stmt.where(Element.is_boilerplate.is_(False))

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_elements_by_element_ids(
        self,
        version_id: uuid.UUID,
        element_ids: list[str],
    ) -> list[Element]:
        """Resolve canonical elements by their stable string element_ids.

        Used by citation validation to prove a cited element genuinely exists.
        """
        if not element_ids:
            return []
        stmt = (
            select(Element)
            .where(Element.version_id == version_id, Element.element_id.in_(element_ids))
            .order_by(Element.sequence_index.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_elements_by_version(self, version_id: uuid.UUID) -> int:
        """Count total elements for a version."""
        stmt = select(func.count()).select_from(Element).where(Element.version_id == version_id)
        result = await self.session.execute(stmt)
        return result.scalar_one() or 0

    async def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        department: str | None = None,
        policy_type: str | None = None,
        policy_status: str | None = None,
    ) -> tuple[list[Document], int]:
        """List documents with optional metadata filtering and total count."""
        stmt = (
            select(Document)
            .options(
                selectinload(Document.versions).selectinload(DocumentVersion.metadata_record),
            )
            .order_by(Document.created_at.desc())
        )

        count_stmt = select(func.count(func.distinct(Document.id))).select_from(Document)

        if department or policy_type or policy_status:
            stmt = stmt.join(Document.versions).join(DocumentVersion.metadata_record)
            count_stmt = count_stmt.join(Document.versions).join(DocumentVersion.metadata_record)

            if department:
                stmt = stmt.where(DocumentMetadata.department == department)
                count_stmt = count_stmt.where(DocumentMetadata.department == department)
            if policy_type:
                stmt = stmt.where(DocumentMetadata.policy_type == policy_type)
                count_stmt = count_stmt.where(DocumentMetadata.policy_type == policy_type)
            if policy_status:
                stmt = stmt.where(DocumentMetadata.policy_status == policy_status)
                count_stmt = count_stmt.where(DocumentMetadata.policy_status == policy_status)

        stmt = stmt.limit(limit).offset(offset)

        docs_result = await self.session.execute(stmt)
        count_result = await self.session.execute(count_stmt)

        return list(docs_result.scalars().unique().all()), (count_result.scalar_one() or 0)

    async def save_document_hierarchy(
        self,
        document: Document,
        version: DocumentVersion,
        pages: list[Page],
        elements: list[Element],
        metadata_record: DocumentMetadata | None = None,
    ) -> Document:
        """Persist a complete document hierarchy atomically within current session."""
        self.session.add(document)
        await self.session.flush()

        version.document_id = document.id
        self.session.add(version)
        await self.session.flush()

        if metadata_record is not None:
            metadata_record.version_id = version.id
            self.session.add(metadata_record)

        for page in pages:
            page.version_id = version.id
            self.session.add(page)
        await self.session.flush()

        # Map page_number to page_id for elements
        page_num_to_id = {page.page_number: page.id for page in pages}

        for element in elements:
            element.version_id = version.id
            if element.page_number in page_num_to_id:
                element.page_id = page_num_to_id[element.page_number]
            self.session.add(element)

        await self.session.flush()
        return document

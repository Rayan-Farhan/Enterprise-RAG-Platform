"""Unit tests for Canonical SQLAlchemy models and hierarchy ancestry (Task 2.1, ADR-005, ADR-037)."""

import uuid
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import (
    Base,
    Document,
    DocumentMetadata,
    DocumentVersion,
    Element,
    Page,
    VersionStatus,
)


@pytest.fixture
async def async_session() -> AsyncGenerator[AsyncSession, None]:
    """Fixture providing an isolated in-memory SQLite database session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_canonical_document_hierarchy_persistence(async_session: AsyncSession) -> None:
    """Assert complete Document -> Version -> Page -> Element hierarchy persists and round-trips."""
    doc_id = uuid.uuid4()
    ver_id = uuid.uuid4()
    page_id = uuid.uuid4()
    el_id = uuid.uuid4()

    # 1. Document
    doc = Document(
        id=doc_id,
        external_id="EXT-1001",
        title="Global HR Policy 2026.pdf",
        mime_type="application/pdf",
        file_size_bytes=1048576,
        file_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        storage_key="original/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.pdf",
    )
    async_session.add(doc)
    await async_session.flush()

    # 2. Version
    ver = DocumentVersion(
        id=ver_id,
        document_id=doc_id,
        version_number=1,
        status=VersionStatus.ACTIVE.value,
        authority="Global HR Committee",
        total_pages=1,
        total_elements=1,
        parser_name="docling",
        parsing_duration_ms=450.0,
    )
    async_session.add(ver)
    await async_session.flush()

    # 3. Metadata
    meta = DocumentMetadata(
        id=uuid.uuid4(),
        version_id=ver_id,
        department="People & Culture",
        policy_type="Leave & Remote Work",
        policy_status="active",
        country="US",
        confidentiality="internal",
        audience="All Full-time Employees",
        custom_attributes={"cost_center": "HR-001", "reviewed_by_legal": True},
    )
    async_session.add(meta)

    # 4. Page
    page = Page(
        id=page_id,
        version_id=ver_id,
        page_number=1,
        width=612.0,
        height=792.0,
        content_hash="pagehash12345",
        page_image_key=f"pages/{ver_id}/page_1.png",
    )
    async_session.add(page)
    await async_session.flush()

    # 5. Element
    element = Element(
        id=el_id,
        version_id=ver_id,
        page_id=page_id,
        page_number=1,
        element_id="sec_1_heading",
        parent_id=None,
        element_type="heading",
        sequence_index=0,
        text_content="Section 1: Annual Paid Time Off Policy",
        content_hash="contenthash999",
        bounding_box={"x0": 54.0, "y0": 72.0, "x1": 558.0, "y1": 96.0, "unit": "pt"},
        source_uri="Global HR Policy 2026.pdf#page=1&el=sec_1_heading",
        extra_metadata={"heading_level": 1},
    )
    async_session.add(element)
    await async_session.commit()

    # Query back and verify full ancestry reconstruction path (ADR-005)
    from sqlalchemy.orm import selectinload

    stmt = (
        select(Element)
        .options(
            selectinload(Element.page),
            selectinload(Element.version).selectinload(DocumentVersion.document),
            selectinload(Element.version).selectinload(DocumentVersion.metadata_record),
        )
        .where(Element.id == el_id)
    )
    result = await async_session.execute(stmt)
    loaded_el = result.scalar_one()

    assert loaded_el.text_content == "Section 1: Annual Paid Time Off Policy"
    assert loaded_el.bounding_box is not None
    assert loaded_el.bounding_box["x0"] == 54.0
    assert loaded_el.page.page_number == 1
    assert loaded_el.version.version_number == 1
    assert loaded_el.version.document.title == "Global HR Policy 2026.pdf"
    assert loaded_el.version.metadata_record is not None
    assert loaded_el.version.metadata_record.department == "People & Culture"
    assert loaded_el.version.metadata_record.custom_attributes["cost_center"] == "HR-001"




@pytest.mark.asyncio
async def test_cascade_deletion(async_session: AsyncSession) -> None:
    """Assert deleting a Document cascades to versions, pages, elements, and metadata."""
    doc_id = uuid.uuid4()
    ver_id = uuid.uuid4()
    page_id = uuid.uuid4()

    doc = Document(
        id=doc_id,
        title="Test Document.pdf",
        mime_type="application/pdf",
        file_hash="testhash111",
        storage_key="original/testhash111.pdf",
    )
    ver = DocumentVersion(
        id=ver_id,
        document_id=doc_id,
        version_number=1,
        parser_name="pymupdf",
    )
    meta = DocumentMetadata(
        id=uuid.uuid4(),
        version_id=ver_id,
        department="Engineering",
    )
    page = Page(
        id=page_id,
        version_id=ver_id,
        page_number=1,
        content_hash="pagehash111",
    )
    el = Element(
        id=uuid.uuid4(),
        version_id=ver_id,
        page_id=page_id,
        page_number=1,
        element_id="el_1",
        element_type="paragraph",
        sequence_index=0,
        text_content="Sample paragraph text.",
        content_hash="elhash111",
    )

    async_session.add_all([doc, ver, meta, page, el])
    await async_session.commit()

    # Delete parent document
    await async_session.delete(doc)
    await async_session.commit()

    # Check that children are deleted
    ver_check = await async_session.get(DocumentVersion, ver_id)
    assert ver_check is None
    page_check = await async_session.get(Page, page_id)
    assert page_check is None

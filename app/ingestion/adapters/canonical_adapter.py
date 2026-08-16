"""Parser-to-Canonical Document Model Adapter (ADR-005)."""

import hashlib
import uuid
from typing import Any

from app.db.models.document import Document
from app.db.models.element import Element
from app.db.models.metadata import DocumentMetadata
from app.db.models.page import Page
from app.db.models.version import DocumentVersion, VersionStatus
from app.ingestion.parsers.base import (
    BoundingBox,
    ParsedDocument,
)


def compute_sha256(text: str | bytes) -> str:
    """Compute SHA-256 hex digest of text or bytes."""
    if isinstance(text, str):
        text = text.encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def bbox_to_dict(bbox: BoundingBox | None) -> dict[str, Any] | None:
    """Convert BoundingBox pydantic model to canonical JSON structure."""
    if bbox is None:
        return None
    return {
        "x0": round(float(bbox.x0), 2),
        "y0": round(float(bbox.y0), 2),
        "x1": round(float(bbox.x1), 2),
        "y1": round(float(bbox.y1), 2),
        "unit": bbox.unit,
        "page_number": bbox.page_number,
    }


class CanonicalAdapter:
    """Transforms intermediate ParsedDocument into decoupled canonical SQLAlchemy database models."""

    @staticmethod
    def to_canonical_models(
        parsed_doc: ParsedDocument,
        file_hash: str,
        storage_key: str,
        file_size_bytes: int = 0,
        metadata_dict: dict[str, Any] | None = None,
        version_number: int = 1,
        status: str = VersionStatus.ACTIVE.value,
        supersedes_id: uuid.UUID | None = None,
    ) -> tuple[Document, DocumentVersion, list[Page], list[Element], DocumentMetadata | None]:
        """Convert a ParsedDocument into canonical entities ready for persistence."""
        doc_id = uuid.uuid4()
        version_id = uuid.uuid4()

        # 1. Document Root
        document = Document(
            id=doc_id,
            external_id=metadata_dict.get("external_id") if metadata_dict else None,
            title=parsed_doc.filename,
            mime_type=parsed_doc.file_type,
            file_size_bytes=file_size_bytes,
            file_hash=file_hash,
            storage_key=storage_key,
            source_priority=metadata_dict.get("source_priority", 1) if metadata_dict else 1,
        )

        # 2. Document Version (ADR-037)
        version = DocumentVersion(
            id=version_id,
            document_id=doc_id,
            version_number=version_number,
            status=status,
            effective_from=metadata_dict.get("effective_from") if metadata_dict else None,
            effective_until=metadata_dict.get("effective_until") if metadata_dict else None,
            supersedes_id=supersedes_id,
            authority=metadata_dict.get("authority") if metadata_dict else None,
            total_pages=parsed_doc.total_pages or len(parsed_doc.pages),
            total_elements=0,  # Updated below
            parser_name=parsed_doc.parser_name,
            parsing_duration_ms=parsed_doc.parsing_duration_ms,
        )

        # 3. HR / Enterprise Metadata (Master Plan §13)
        metadata_record: DocumentMetadata | None = None
        if metadata_dict:
            metadata_record = DocumentMetadata(
                id=uuid.uuid4(),
                version_id=version_id,
                department=metadata_dict.get("department"),
                policy_type=metadata_dict.get("policy_type"),
                policy_status=metadata_dict.get("policy_status"),
                country=metadata_dict.get("country"),
                location=metadata_dict.get("location"),
                employee_type=metadata_dict.get("employee_type"),
                grade=metadata_dict.get("grade"),
                confidentiality=metadata_dict.get("confidentiality"),
                audience=metadata_dict.get("audience"),
                custom_attributes=metadata_dict.get("custom_attributes", {}),
            )

        # 4. Pages and Elements
        canonical_pages: list[Page] = []
        canonical_elements: list[Element] = []
        global_seq_idx = 0

        for page in parsed_doc.pages:
            page_id = uuid.uuid4()
            page_text_accum: list[str] = []

            # Page model
            page_entity = Page(
                id=page_id,
                version_id=version_id,
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                content_hash="",  # Updated after gathering elements
                page_image_key=f"pages/{version_id}/page_{page.page_number}.png",
            )

            # Convert standard elements
            for el in page.elements:
                page_text_accum.append(el.text)
                extra = dict(el.metadata)
                if el.level is not None:
                    extra["heading_level"] = el.level

                el_entity = Element(
                    id=uuid.uuid4(),
                    version_id=version_id,
                    page_id=page_id,
                    page_number=page.page_number,
                    element_id=el.element_id,
                    parent_id=el.parent_id,
                    element_type=str(el.element_type.value),
                    sequence_index=global_seq_idx,
                    text_content=el.text,
                    content_hash=compute_sha256(el.text),
                    bounding_box=bbox_to_dict(el.bounding_box),
                    table_data=None,
                    asset_storage_key=None,
                    source_uri=f"{parsed_doc.filename}#page={page.page_number}&el={el.element_id}",
                    extra_metadata=extra,
                )
                canonical_elements.append(el_entity)
                global_seq_idx += 1

            # Convert table elements
            for tbl in page.tables:
                table_text = tbl.markdown or "\n".join(["\t".join(row) for row in tbl.cells])
                page_text_accum.append(table_text)

                tbl_entity = Element(
                    id=uuid.uuid4(),
                    version_id=version_id,
                    page_id=page_id,
                    page_number=page.page_number,
                    element_id=tbl.table_id,
                    parent_id=None,
                    element_type="table",
                    sequence_index=global_seq_idx,
                    text_content=table_text,
                    content_hash=compute_sha256(table_text),
                    bounding_box=bbox_to_dict(tbl.bounding_box),
                    table_data={
                        "title": tbl.title,
                        "num_rows": tbl.num_rows,
                        "num_cols": tbl.num_cols,
                        "headers": tbl.headers,
                        "cells": tbl.cells,
                        "markdown": tbl.markdown,
                    },
                    asset_storage_key=f"tables/{version_id}/{tbl.table_id}.json",
                    source_uri=f"{parsed_doc.filename}#page={page.page_number}&tbl={tbl.table_id}",
                    extra_metadata={},
                )
                canonical_elements.append(tbl_entity)
                global_seq_idx += 1

            # Convert figures / images
            for fig in page.figures:
                caption = fig.caption or ""
                page_text_accum.append(caption)

                fig_entity = Element(
                    id=uuid.uuid4(),
                    version_id=version_id,
                    page_id=page_id,
                    page_number=page.page_number,
                    element_id=fig.figure_id,
                    parent_id=None,
                    element_type="figure",
                    sequence_index=global_seq_idx,
                    text_content=caption,
                    content_hash=compute_sha256(caption.encode() + (fig.image_bytes or b"")),
                    bounding_box=bbox_to_dict(fig.bounding_box),
                    table_data=None,
                    asset_storage_key=f"images/{version_id}/{fig.figure_id}.{fig.format}",
                    source_uri=f"{parsed_doc.filename}#page={page.page_number}&fig={fig.figure_id}",
                    extra_metadata={"format": fig.format},
                )
                canonical_elements.append(fig_entity)
                global_seq_idx += 1

            # Page content hash
            combined_page_text = "\n".join(page_text_accum)
            page_entity.content_hash = compute_sha256(combined_page_text)
            canonical_pages.append(page_entity)

        version.total_elements = len(canonical_elements)

        return document, version, canonical_pages, canonical_elements, metadata_record

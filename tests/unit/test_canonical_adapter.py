"""Unit tests for Parser-to-Canonical Document Model Adapter (Task 2.4, ADR-005)."""


from app.ingestion.adapters.canonical_adapter import CanonicalAdapter, compute_sha256
from app.ingestion.parsers.base import (
    BoundingBox,
    ElementType,
    ParsedDocument,
    ParsedElement,
    ParsedFigure,
    ParsedPage,
    ParsedTable,
)


def test_canonical_adapter_transformation() -> None:
    """Verify ParsedDocument correctly adapts to canonical models without leaking parser specifics."""
    # Build sample parsed document as produced by Docling or PyMuPDF
    parsed_doc = ParsedDocument(
        filename="leave_policy.pdf",
        file_type="application/pdf",
        total_pages=2,
        parser_name="docling",
        parsing_duration_ms=310.5,
        pages=[
            ParsedPage(
                page_number=1,
                width=612.0,
                height=792.0,
                elements=[
                    ParsedElement(
                        element_id="el_h1",
                        element_type=ElementType.HEADING,
                        text="Parental Leave Policy 2026",
                        page_number=1,
                        level=1,
                        bounding_box=BoundingBox(
                            x0=50.0, y0=70.0, x1=400.0, y1=100.0, page_number=1, unit="pt"
                        ),
                    ),
                    ParsedElement(
                        element_id="el_p1",
                        element_type=ElementType.PARAGRAPH,
                        text="Eligible employees receive up to 16 weeks of paid parental leave.",
                        page_number=1,
                        parent_id="el_h1",
                    ),
                ],
                tables=[
                    ParsedTable(
                        table_id="tbl_1",
                        title="Leave Accrual Table",
                        page_number=1,
                        num_rows=2,
                        num_cols=2,
                        headers=["Tenure", "Weeks"],
                        cells=[["< 1 year", "8 weeks"], [">= 1 year", "16 weeks"]],
                        markdown="| Tenure | Weeks |\n| --- | --- |\n| < 1 year | 8 weeks |\n| >= 1 year | 16 weeks |",
                    )
                ],
            ),
            ParsedPage(
                page_number=2,
                width=612.0,
                height=792.0,
                figures=[
                    ParsedFigure(
                        figure_id="fig_1",
                        caption="Leave Approval Flowchart",
                        page_number=2,
                        image_bytes=b"fake_image_bytes",
                        format="png",
                    )
                ],
            ),
        ],
    )

    file_hash = compute_sha256("fake file content")
    storage_key = f"original/{file_hash}.pdf"
    metadata_payload = {
        "department": "Human Resources",
        "policy_type": "Parental Leave",
        "country": "US",
        "authority": "Global HR VP",
        "custom_attributes": {"tiered": True},
    }

    doc, version, pages, elements, meta = CanonicalAdapter.to_canonical_models(
        parsed_doc=parsed_doc,
        file_hash=file_hash,
        storage_key=storage_key,
        file_size_bytes=4096,
        metadata_dict=metadata_payload,
    )

    # 1. Assert Document Root
    assert doc.title == "leave_policy.pdf"
    assert doc.file_hash == file_hash
    assert doc.storage_key == storage_key
    assert doc.file_size_bytes == 4096

    # 2. Assert Version
    assert version.version_number == 1
    assert version.total_pages == 2
    assert version.total_elements == 4  # 2 elements + 1 table + 1 figure
    assert version.parser_name == "docling"
    assert version.authority == "Global HR VP"

    # 3. Assert Metadata
    assert meta is not None
    assert meta.department == "Human Resources"
    assert meta.policy_type == "Parental Leave"
    assert meta.custom_attributes["tiered"] is True

    # 4. Assert Pages
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2
    assert len(pages[0].content_hash) == 64

    # 5. Assert Elements
    assert len(elements) == 4
    # Heading element
    assert elements[0].element_type == "heading"
    assert elements[0].bounding_box is not None
    assert elements[0].bounding_box["x0"] == 50.0
    assert elements[0].extra_metadata["heading_level"] == 1

    # Paragraph element
    assert elements[1].element_type == "paragraph"
    assert elements[1].parent_id == "el_h1"

    # Table element
    assert elements[2].element_type == "table"
    assert elements[2].table_data is not None
    assert elements[2].table_data["num_rows"] == 2
    assert elements[2].table_data["headers"] == ["Tenure", "Weeks"]

    # Figure element
    assert elements[3].element_type == "figure"
    assert elements[3].text_content == "Leave Approval Flowchart"
    assert elements[3].asset_storage_key is not None
    assert "images/" in elements[3].asset_storage_key


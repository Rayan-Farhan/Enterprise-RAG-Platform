"""Unit tests for multi-level deduplication and boilerplate detection (Task 2.5, Master Plan §10)."""

import uuid

from app.db.models.element import Element
from app.ingestion.dedup import (
    BoilerplateDetector,
    compute_file_sha256,
    compute_simhash,
    simhash_similarity,
)


def test_exact_file_hash_computation() -> None:
    """Verify SHA-256 hash is deterministic and exact."""
    payload1 = b"Company Travel Policy 2026"
    payload2 = b"Company Travel Policy 2026"
    payload3 = b"Company Travel Policy 2025"

    h1 = compute_file_sha256(payload1)
    h2 = compute_file_sha256(payload2)
    h3 = compute_file_sha256(payload3)

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64


def test_simhash_near_duplicate_detection() -> None:
    """Verify SimHash yields high similarity for minor text variations and low for different text."""
    text_orig = "Employees traveling on company business must book flights through the corporate travel portal at least 14 days in advance."
    text_minor_edit = "Employees traveling on company business should book flights through the corporate travel portal at least 14 days in advance."
    text_different = "Annual performance reviews occur each November and dictate merit-based compensation adjustments for engineering staff."

    hash_orig = compute_simhash(text_orig)
    hash_minor = compute_simhash(text_minor_edit)
    hash_diff = compute_simhash(text_different)

    sim_high = simhash_similarity(hash_orig, hash_minor)
    sim_low = simhash_similarity(hash_orig, hash_diff)

    assert sim_high >= 0.85, f"Expected high similarity >= 0.85, got {sim_high}"
    assert sim_low < 0.70, f"Expected low similarity < 0.70, got {sim_low}"


def test_boilerplate_detection_recurring_headers_and_footers() -> None:
    """Verify BoilerplateDetector flags recurring text on multi-page documents."""
    vid = uuid.uuid4()
    pid = uuid.uuid4()

    # Create a 4-page sequence of elements
    elements: list[Element] = []
    for page_num in range(1, 5):
        # 1. Header occurring on all 4 pages
        elements.append(
            Element(
                id=uuid.uuid4(),
                version_id=vid,
                page_id=pid,
                page_number=page_num,
                element_id=f"hdr_{page_num}",
                element_type="paragraph",
                sequence_index=page_num * 10,
                text_content="ACME CORP INTERNAL POLICY DOCUMENT",
                content_hash="h1",
                bounding_box={"x0": 50, "y0": 20, "x1": 500, "y1": 40, "unit": "pt"},
            )
        )
        # 2. Page number footer
        elements.append(
            Element(
                id=uuid.uuid4(),
                version_id=vid,
                page_id=pid,
                page_number=page_num,
                element_id=f"ftr_{page_num}",
                element_type="paragraph",
                sequence_index=page_num * 10 + 1,
                text_content=f"Page {page_num} of 4",
                content_hash="h2",
            )
        )
        # 3. Unique policy body content on each page
        elements.append(
            Element(
                id=uuid.uuid4(),
                version_id=vid,
                page_id=pid,
                page_number=page_num,
                element_id=f"body_{page_num}",
                element_type="paragraph",
                sequence_index=page_num * 10 + 2,
                text_content=f"Specific unique section paragraph content for chapter {page_num}.",
                content_hash=f"bodyhash_{page_num}",
                bounding_box={"x0": 50, "y0": 200, "x1": 500, "y1": 300, "unit": "pt"},
            )
        )

    detector = BoilerplateDetector(recurrence_threshold=0.60, min_pages_for_recurrence=3)
    flagged_elements = detector.detect_and_flag(elements, total_pages=4)

    # Check headers
    for el in flagged_elements:
        if "ACME CORP" in el.text_content:
            assert el.is_boilerplate is True
            assert el.boilerplate_reason is not None

        elif "Page " in el.text_content:
            assert el.is_boilerplate is True
            assert el.boilerplate_reason == "Matched boilerplate pattern regex"

        elif "Specific unique section" in el.text_content:
            assert el.is_boilerplate is False

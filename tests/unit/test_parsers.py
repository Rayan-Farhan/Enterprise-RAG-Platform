"""Unit tests for document parsers (Task 1.3, Task 1.4)."""

import shutil
from pathlib import Path

import docx
import openpyxl
import pptx
import pytest

from app.ingestion.parsers.base import ElementType, ParsedDocument
from app.ingestion.parsers.column_heuristic_parser import ColumnHeuristicParser
from app.ingestion.parsers.docling_parser import DoclingParser
from app.ingestion.parsers.layout_heuristic_parser import LayoutHeuristicParser
from app.ingestion.parsers.office_parser import OfficeParser
from app.ingestion.parsers.opendataloader_parser import OpenDataLoaderParser
from app.ingestion.parsers.pymupdf_parser import PyMuPDFParser

CORPUS_DIR = Path("benchmarks/corpus")


@pytest.fixture
def staff_handbook_pdf() -> Path:
    return CORPUS_DIR / "staff_handbook.pdf"


@pytest.fixture
def health_plan_pdf() -> Path:
    return CORPUS_DIR / "health_plan_at_a_glance_2026.pdf"


@pytest.fixture
def temp_docx(tmp_path: Path) -> Path:
    p = tmp_path / "sample_policy.docx"
    doc = docx.Document()
    doc.add_heading("Global Travel Policy", level=1)
    doc.add_paragraph("Employees are reimbursed for reasonable business travel.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Tier"
    table.rows[0].cells[1].text = "Allowance"
    r = table.add_row()
    r.cells[0].text = "Tier 1"
    r.cells[1].text = "$100"
    doc.save(str(p))
    return p


@pytest.fixture
def temp_xlsx(tmp_path: Path) -> Path:
    p = tmp_path / "bonus_matrix.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Bonus"
    ws.append(["Grade", "Target Bonus"])
    ws.append(["Grade 1", 0.05])
    wb.save(str(p))
    return p


@pytest.fixture
def temp_pptx(tmp_path: Path) -> Path:
    p = tmp_path / "presentation.pptx"
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "DEI Strategy"
    slide.placeholders[1].text = "2026 Roadmap"
    prs.save(str(p))
    return p


def test_layout_heuristic_parser_pdf(health_plan_pdf: Path) -> None:
    if not health_plan_pdf.exists():
        pytest.skip("Corpus file not found.")

    parser = LayoutHeuristicParser()
    doc = parser.parse(health_plan_pdf)

    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "pymupdf-layout"
    assert doc.total_pages == 11
    assert len(doc.pages) == 11
    assert len(doc.all_elements) > 0

    # Check bounding box presence
    for el in doc.all_elements[:20]:
        assert el.bounding_box is not None
        assert el.bounding_box.page_number >= 1


def test_column_heuristic_parser_pdf(staff_handbook_pdf: Path) -> None:
    if not staff_handbook_pdf.exists():
        pytest.skip("Corpus file not found.")

    parser = ColumnHeuristicParser()
    doc = parser.parse(staff_handbook_pdf)

    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "pymupdf-columns"
    assert doc.total_pages == 56
    assert len(doc.all_elements) > 0


def test_opendataloader_parser_pdf() -> None:
    pytest.importorskip("opendataloader_pdf")
    if shutil.which("java") is None:
        pytest.skip("OpenDataLoader needs a Java runtime.")
    pdf = CORPUS_DIR / "dental_plan_at_a_glance_2026.pdf"

    doc = OpenDataLoaderParser().parse(pdf)

    assert doc.parser_name == "opendataloader"
    assert doc.total_pages == 4
    headings = [e for e in doc.all_elements if e.element_type == ElementType.HEADING]
    assert headings and all(h.level for h in headings)
    # Page 3's benefit grid comes back as a real two-column table.
    table = doc.pages[2].tables[0]
    assert table.num_cols == 2 and table.cells[0][0] == "Deductible"
    # Boxes are flipped to the canonical top-left origin.
    box = headings[0].bounding_box
    assert box is not None and 0 <= box.y0 < box.y1 <= doc.pages[0].height


@pytest.mark.heavy
def test_docling_parser_pdf() -> None:
    pytest.importorskip("docling")
    pdf = CORPUS_DIR / "dental_plan_at_a_glance_2026.pdf"

    doc = DoclingParser().parse(pdf)

    assert doc.parser_name == "docling"
    assert doc.total_pages == 4
    assert any(e.element_type == ElementType.HEADING for e in doc.all_elements)
    table = doc.pages[2].tables[0]
    assert table.num_cols == 2
    box = doc.all_elements[0].bounding_box
    assert box is not None and 0 <= box.y0 < box.y1 <= doc.pages[0].height


def test_pymupdf_parser_pdf(staff_handbook_pdf: Path) -> None:
    if not staff_handbook_pdf.exists():
        pytest.skip("Corpus file not found.")

    parser = PyMuPDFParser()
    doc = parser.parse(staff_handbook_pdf)

    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "pymupdf"
    assert doc.total_pages == 56
    assert len(doc.all_elements) > 0


def test_office_parser_docx(temp_docx: Path) -> None:
    parser = OfficeParser()
    doc = parser.parse(temp_docx)

    assert isinstance(doc, ParsedDocument)
    assert doc.file_type == "docx"
    assert len(doc.all_elements) > 0
    assert len(doc.all_tables) == 1
    assert "Tier" in doc.all_tables[0].headers


def test_office_parser_xlsx(temp_xlsx: Path) -> None:
    parser = OfficeParser()
    doc = parser.parse(temp_xlsx)

    assert isinstance(doc, ParsedDocument)
    assert doc.file_type == "xlsx"
    assert len(doc.all_tables) >= 1
    assert "Grade" in doc.all_tables[0].headers


def test_office_parser_pptx(temp_pptx: Path) -> None:
    parser = OfficeParser()
    doc = parser.parse(temp_pptx)

    assert isinstance(doc, ParsedDocument)
    assert doc.file_type == "pptx"
    assert doc.total_pages == 1
    assert len(doc.all_elements) > 0


def test_opendataloader_hybrid_options_name_the_backend() -> None:
    local = OpenDataLoaderParser()
    assert local.parser_name == "opendataloader"
    assert local._hybrid_options() == {}

    hybrid = OpenDataLoaderParser(hybrid_url="http://127.0.0.1:5002", hybrid_timeout_ms=60000)
    assert hybrid.parser_name == "opendataloader-hybrid"
    assert hybrid._hybrid_options() == {
        "hybrid": "docling-fast",
        "hybrid_mode": "auto",
        "hybrid_url": "http://127.0.0.1:5002",
        "hybrid_timeout": "60000",
    }

"""Page-routed PDF parser (ADR-004)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.parsers.base import (
    BoundingBox,
    ElementType,
    ParsedDocument,
    ParsedElement,
    ParsedPage,
)
from app.ingestion.parsers.layout_heuristic_parser import LayoutHeuristicParser
from app.ingestion.parsers.routed_parser import GRID, PROSE, RoutedPdfParser, classify_pages
from app.ingestion.parsers.router import primary_pdf_parser

CORPUS = Path("benchmarks/corpus")


class _Engine:
    """Returns one element per page, labelled with the engine's name."""

    def __init__(self, name: str, pages: int, fail: bool = False) -> None:
        self.parser_name = name
        self.pages = pages
        self.fail = fail
        self.calls = 0

    def parse(self, file_path: Path | str, mime_type: str | None = None) -> ParsedDocument:
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"{self.parser_name} crashed")
        return ParsedDocument(
            filename=Path(file_path).name,
            file_type="pdf",
            total_pages=self.pages,
            parser_name=self.parser_name,
            pages=[
                ParsedPage(
                    page_number=n,
                    elements=[
                        ParsedElement(
                            element_id=f"{self.parser_name}_{n}",
                            element_type=ElementType.PARAGRAPH,
                            text=f"{self.parser_name} page {n}",
                            page_number=n,
                            bounding_box=BoundingBox(x0=0, y0=0, x1=1, y1=1, page_number=n),
                        )
                    ],
                )
                for n in range(1, self.pages + 1)
            ],
        )


def test_benefit_grid_pages_and_prose_pages_are_told_apart() -> None:
    health = classify_pages(CORPUS / "health_plan_at_a_glance_2026.pdf")
    assert [r.kind for r in health][1:3] == [GRID, GRID]  # the benefit matrix pages
    booklet = classify_pages(CORPUS / "bcbs_dental_booklet_2026.pdf")
    assert sum(r.kind == PROSE for r in booklet) / len(booklet) > 0.9


def test_each_page_comes_from_the_engine_it_was_routed_to() -> None:
    pdf = CORPUS / "dental_plan_at_a_glance_2026.pdf"  # page 3 is the grid
    prose, grid = _Engine("docling", 4), _Engine("opendataloader", 4)
    parsed = RoutedPdfParser(prose_parser=lambda ocr: prose, grid_parser=lambda: grid).parse(pdf)

    texts = [p.elements[0].text for p in parsed.pages]
    assert texts[2] == "opendataloader page 3"
    assert all(t.startswith("docling") for i, t in enumerate(texts) if i != 2)
    assert prose.calls == 1 and grid.calls == 1
    assert [e.sequence_index for p in parsed.pages for e in p.elements] == [1, 2, 3, 4]
    assert parsed.metadata["page_routes"][2]["engine"] == "opendataloader"


def test_an_engine_that_is_not_needed_is_not_run() -> None:
    pdf = CORPUS / "organizational_structure.pdf"  # one page, no grid
    grid = _Engine("opendataloader", 1)
    RoutedPdfParser(prose_parser=lambda ocr: _Engine("docling", 1), grid_parser=lambda: grid).parse(
        pdf
    )
    assert grid.calls == 0


def test_a_failed_engine_falls_back_to_the_heuristic_for_its_pages() -> None:
    pdf = CORPUS / "dental_plan_at_a_glance_2026.pdf"
    parsed = RoutedPdfParser(
        prose_parser=lambda ocr: _Engine("docling", 4, fail=True),
        grid_parser=lambda: _Engine("opendataloader", 4),
        fallback_parser=LayoutHeuristicParser,
    ).parse(pdf)
    engines = [r["engine"] for r in parsed.metadata["page_routes"]]
    assert engines[2] == "opendataloader"
    assert {engines[i] for i in (0, 1, 3)} == {"pymupdf-layout"}
    assert "docling" in parsed.metadata["engine_errors"]


@pytest.mark.parametrize(
    ("choice", "name"),
    [("heuristic", "pymupdf-layout"), ("routed", "routed"), ("docling", "docling")],
)
def test_pdf_parser_setting_selects_the_primary(choice: str, name: str) -> None:
    assert primary_pdf_parser(choice).parser_name == name

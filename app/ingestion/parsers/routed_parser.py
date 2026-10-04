"""Page-routed PDF parser: each page goes to the engine that measured best for it.

The parser benchmark (ADR-004) found no single engine best at everything on
this corpus. Docling's layout model recovers headings far better (F1 0.88 vs
0.64 for OpenDataLoader and 0.28 for the PyMuPDF heuristic), but it fragments
the banded benefit grids, where OpenDataLoader's rule-based table reader
places 100% of cells correctly. So the decision is made per page:

* **grid pages** (detected tables cover at least ``grid_threshold`` of the
  page) go to OpenDataLoader;
* **image-only pages** (almost no text layer, but images) go to Docling with
  OCR, since no text-layer parser can read them;
* **all other pages** (prose) go to Docling.

Each engine runs at most once per document, over the whole file, and only if a
page needs it. Its output is kept only for the pages routed to it. If an
engine fails, its pages fall back to ``fallback`` (the production PyMuPDF
heuristic), so a missing optional dependency or a crashed engine degrades
quality instead of failing ingestion. The routing decision for every page is
recorded in the document metadata.

Page classification uses PyMuPDF's table finder and text layer, at about
0.1-0.25 s per page.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.core.logging import get_logger
from app.ingestion.parsers.base import DocumentParser, ParsedDocument, ParsedPage

logger = get_logger("app.ingestion.parsers.routed")

GRID = "grid"
PROSE = "prose"
IMAGE = "image"


@dataclass(frozen=True)
class PageRoute:
    page_number: int
    kind: str
    table_area: float
    words: int


def classify_pages(
    path: Path | str, grid_threshold: float = 0.30, image_max_words: int = 20
) -> list[PageRoute]:
    """Route every page by how much of it is table and whether it has a text layer."""
    routes: list[PageRoute] = []
    with pymupdf.open(str(path)) as pdf:
        for index, page in enumerate(pdf):
            words = len(page.get_text("words"))
            if words < image_max_words and page.get_images():
                routes.append(PageRoute(index + 1, IMAGE, 0.0, words))
                continue
            area = page.rect.width * page.rect.height or 1.0
            covered = sum(pymupdf.Rect(t.bbox).get_area() for t in page.find_tables().tables)
            share = min(1.0, covered / area)
            kind = GRID if share >= grid_threshold else PROSE
            routes.append(PageRoute(index + 1, kind, round(share, 3), words))
    return routes


def _default_prose(do_ocr: bool) -> DocumentParser:
    from app.ingestion.parsers.docling_parser import DoclingParser

    return DoclingParser(do_ocr=do_ocr)


def _default_grid() -> DocumentParser:
    from app.ingestion.parsers.opendataloader_parser import OpenDataLoaderParser

    return OpenDataLoaderParser()


def _default_fallback() -> DocumentParser:
    from app.ingestion.parsers.layout_heuristic_parser import LayoutHeuristicParser

    return LayoutHeuristicParser()


class RoutedPdfParser:
    """Docling for prose and image pages, OpenDataLoader for grid pages, heuristic fallback."""

    parser_name: str = "routed"

    def __init__(
        self,
        prose_parser: Callable[[bool], DocumentParser] = _default_prose,
        grid_parser: Callable[[], DocumentParser] = _default_grid,
        fallback_parser: Callable[[], DocumentParser] = _default_fallback,
        grid_threshold: float = 0.30,
        image_max_words: int = 20,
    ) -> None:
        self._prose = prose_parser
        self._grid = grid_parser
        self._fallback = fallback_parser
        self.grid_threshold = grid_threshold
        self.image_max_words = image_max_words

    def parse(self, file_path: Path | str, mime_type: str | None = None) -> ParsedDocument:
        path = Path(file_path)
        started = time.perf_counter()
        routes = classify_pages(path, self.grid_threshold, self.image_max_words)
        wanted = {
            "docling": {r.page_number for r in routes if r.kind in (PROSE, IMAGE)},
            "opendataloader": {r.page_number for r in routes if r.kind == GRID},
        }
        needs_ocr = any(r.kind == IMAGE for r in routes)

        engines: dict[str, Callable[[], DocumentParser]] = {
            "docling": lambda: self._prose(needs_ocr),
            "opendataloader": self._grid,
        }
        source_by_page: dict[int, str] = {}
        pages: dict[int, ParsedPage] = {}
        errors: dict[str, str] = {}
        fallback_pages: set[int] = set()

        for engine, page_numbers in wanted.items():
            if not page_numbers:
                continue
            try:
                parsed = engines[engine]().parse(path, mime_type)
            except Exception as exc:  # noqa: BLE001 - a failed engine degrades, never fails ingest
                errors[engine] = f"{type(exc).__name__}: {exc}"
                logger.warning("routed_engine_failed", engine=engine, error=errors[engine])
                fallback_pages |= page_numbers
                continue
            for page in parsed.pages:
                if page.page_number in page_numbers:
                    pages[page.page_number] = page
                    source_by_page[page.page_number] = engine

        if fallback_pages:
            parsed = self._fallback().parse(path, mime_type)
            for page in parsed.pages:
                if page.page_number in fallback_pages:
                    pages[page.page_number] = page
                    source_by_page[page.page_number] = parsed.parser_name

        ordered = [pages[n] for n in sorted(pages)]
        sequence = 0
        for page in ordered:
            for element in page.elements:
                sequence += 1
                element.sequence_index = sequence

        return ParsedDocument(
            filename=path.name,
            file_type="pdf",
            total_pages=len(routes),
            pages=ordered,
            metadata={
                "source_path": str(path),
                "intelligence_engine": "routed",
                "page_routes": [
                    {
                        "page": r.page_number,
                        "kind": r.kind,
                        "table_area": r.table_area,
                        "engine": source_by_page.get(r.page_number),
                    }
                    for r in routes
                ],
                "ocr": needs_ocr,
                "engine_errors": errors,
            },
            parser_name=self.parser_name,
            parsing_duration_ms=(time.perf_counter() - started) * 1000.0,
        )

"""OpenDataLoader PDF parser adapter (ADR-004 candidate).

Runs the OpenDataLoader PDF engine (a Java library, driven through the
``opendataloader-pdf`` package): rule-based layout analysis that emits typed
headings with levels, paragraphs, lists, tables with row/column/span structure
and images, each with page and bounding box, as JSON. No model and no GPU.

Optional dependency (``pip install -e ".[parsers]"``) that also needs a Java
runtime (11+) on PATH; imported on first use.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pymupdf

from app.ingestion.parsers.base import (
    BoundingBox,
    ElementType,
    ParsedDocument,
    ParsedElement,
    ParsedFigure,
    ParsedPage,
    ParsedTable,
)

_TEXT_TYPES: dict[str, ElementType] = {
    "heading": ElementType.HEADING,
    "paragraph": ElementType.PARAGRAPH,
    "caption": ElementType.PARAGRAPH,
    "list item": ElementType.LIST,
    "formula": ElementType.FORMULA,
    "header": ElementType.HEADER,
    "footer": ElementType.FOOTER,
}


_FURNITURE: dict[str, ElementType] = {"header": ElementType.HEADER, "footer": ElementType.FOOTER}


def _cell_text(node: dict[str, Any]) -> str:
    """Concatenate the text under a node (table cells hold paragraphs as kids)."""
    parts = [str(node.get("content") or "").strip()]
    # Cells can hold lists, whose entries sit under "list items", not "kids".
    for kid in (node.get("kids") or []) + (node.get("list items") or []):
        parts.append(_cell_text(kid))
    return " ".join(p for p in parts if p)


def _iter_nodes(nodes: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Every node in the output tree, table cells included."""
    for node in nodes:
        yield node
        for key in ("kids", "list items", "rows", "cells"):
            yield from _iter_nodes(node.get(key) or [])


def _words(text: str) -> set[str]:
    return {w for w in text.lower().split() if len(w) > 3}


def _align_pages(tree: dict[str, Any], physical_text: list[str]) -> dict[int, int]:
    """Map OpenDataLoader page numbers onto physical page numbers.

    On some PDFs the engine reads fewer pages than the document has (89 → 85 on
    the employee policy manual), skipping some and numbering the rest
    consecutively, so its page numbers drift from the physical ones. Each of its
    pages is anchored to the physical page whose text it shares most words with,
    searching forward in order. When the counts agree the mapping is the
    identity and no search runs.
    """
    reported = int(tree.get("number of pages") or 0)
    if reported == len(physical_text):
        return {n: n for n in range(1, reported + 1)}

    text_by_page: dict[int, list[str]] = {}
    for node in _iter_nodes(tree.get("kids") or []):
        if node.get("content") and node.get("page number"):
            text_by_page.setdefault(int(node["page number"]), []).append(str(node["content"]))

    physical_words = [_words(t) for t in physical_text]
    missing = max(0, len(physical_text) - reported)
    mapping: dict[int, int] = {}
    cursor = 0  # first physical index still available
    for number in range(1, reported + 1):
        skipped = cursor - (number - 1)  # physical pages passed over so far
        window = range(cursor, min(len(physical_text), cursor + missing - skipped + 1))
        words = _words(" ".join(text_by_page.get(number, [])))
        best = max(window, key=lambda i: len(words & physical_words[i]), default=cursor)
        mapping[number] = best + 1
        cursor = best + 1
    return mapping


class OpenDataLoaderParser:
    """OpenDataLoader's JSON output mapped onto the canonical schema."""

    parser_name: str = "opendataloader"

    def parse(self, file_path: Path | str, mime_type: str | None = None) -> ParsedDocument:
        import opendataloader_pdf

        path = Path(file_path)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory() as out_dir:
            opendataloader_pdf.convert(
                input_path=str(path),
                output_dir=out_dir,
                format="json",
                quiet=True,
                include_header_footer=True,
            )
            tree = json.loads((Path(out_dir) / f"{path.stem}.json").read_text(encoding="utf-8"))

        # OpenDataLoader reports PDF user-space boxes (origin bottom-left); the
        # canonical schema is top-left like every other adapter, so page heights
        # are needed to flip them.
        with pymupdf.open(path) as pdf:
            sizes = {i + 1: (page.rect.width, page.rect.height) for i, page in enumerate(pdf)}
            physical_text = [page.get_text("text") for page in pdf]
        remap = _align_pages(tree, physical_text)
        for node in _iter_nodes(tree.get("kids") or []):
            if "page number" in node:
                node["page number"] = remap.get(int(node["page number"]), 0)
        pages = {
            number: ParsedPage(page_number=number, width=w, height=h)
            for number, (w, h) in sizes.items()
        }
        sequence = 0

        def bbox(node: dict[str, Any]) -> BoundingBox | None:
            raw = node.get("bounding box")
            number = int(node.get("page number") or 0)
            if not raw or number not in pages:
                return None
            x0, y0, x1, y1 = (float(v) for v in raw)
            height = pages[number].height
            return BoundingBox(x0=x0, y0=height - y1, x1=x1, y1=height - y0, page_number=number)

        def walk(
            nodes: list[dict[str, Any]], furniture: ElementType | None = None
        ) -> Iterator[tuple[dict[str, Any], ElementType | None]]:
            # Lists, text blocks and list items nest their content; tables are
            # read whole from their rows, so their cells are not walked as text.
            # Running headers and footers are containers whose text sits in typed
            # children (a footer holds a "heading"), so their type is inherited.
            for node in nodes:
                yield node, furniture
                inner = _FURNITURE.get(str(node.get("type")), furniture)
                if node.get("type") != "table":
                    kids = (node.get("list items") or []) + (node.get("kids") or [])
                    yield from walk(kids, inner)

        for node, furniture in walk(tree.get("kids") or []):
            kind = node.get("type")
            number = int(node.get("page number") or 0)
            if number not in pages:
                continue
            page = pages[number]
            sequence += 1

            if kind == "table":
                n_rows = int(node.get("number of rows") or 0)
                n_cols = int(node.get("number of columns") or 0)
                grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
                header_rows: set[int] = set()
                for row in node.get("rows") or []:
                    for cell in row.get("cells") or []:
                        r = int(cell.get("row number", 1)) - 1
                        c = int(cell.get("column number", 1)) - 1
                        if 0 <= r < n_rows and 0 <= c < n_cols:
                            grid[r][c] = _cell_text(cell)
                            if cell.get("is_header"):
                                header_rows.add(r)
                headers = grid[0] if 0 in header_rows and grid else []
                body = grid[1:] if headers else grid
                markdown = "\n".join("| " + " | ".join(row) + " |" for row in grid)
                page.tables.append(
                    ParsedTable(
                        table_id=f"odl_table_{number}_{sequence}",
                        page_number=number,
                        num_rows=n_rows,
                        num_cols=n_cols,
                        headers=headers,
                        cells=body,
                        bounding_box=bbox(node),
                        markdown=markdown,
                    )
                )
            elif kind == "image":
                page.figures.append(
                    ParsedFigure(
                        figure_id=f"odl_figure_{number}_{sequence}",
                        page_number=number,
                        bounding_box=bbox(node),
                    )
                )
            elif kind in _TEXT_TYPES:
                text = str(node.get("content") or "").strip()
                if not text:
                    continue
                element_type = furniture or _TEXT_TYPES[kind]
                level = (
                    int(node.get("heading level") or 1)
                    if element_type == ElementType.HEADING
                    else None
                )
                page.elements.append(
                    ParsedElement(
                        element_id=f"odl_elem_{number}_{sequence}",
                        element_type=element_type,
                        text=text,
                        page_number=number,
                        bounding_box=bbox(node),
                        level=level,
                        sequence_index=sequence,
                        metadata={"odl_type": kind, "font_size": node.get("font size")},
                    )
                )

        ordered = [pages[number] for number in sorted(pages)]
        return ParsedDocument(
            filename=path.name,
            file_type="pdf",
            total_pages=len(ordered),
            pages=ordered,
            metadata={"source_path": str(path), "intelligence_engine": "opendataloader"},
            parser_name=self.parser_name,
            parsing_duration_ms=(time.perf_counter() - started) * 1000.0,
        )

"""Docling PDF parser adapter (ADR-004 candidate).

Runs IBM's Docling layout pipeline: a layout model segments each page into typed
regions (title, section header, text, list item, table, picture, page header and
footer, ...), TableFormer recovers table structure, and the PDF's own text layer
fills the regions. OCR is off: every corpus PDF is born-digital, and leaving it
on would make the benchmark measure Tesseract-class OCR rather than Docling's
layout analysis.

Docling and its models are an optional dependency (``pip install -e
".[parsers]"``), imported on first use, so production installs and CI do not
carry PyTorch unless they choose this parser.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.ingestion.parsers.base import (
    BoundingBox,
    ElementType,
    ParsedDocument,
    ParsedElement,
    ParsedFigure,
    ParsedPage,
    ParsedTable,
)

#: Docling label -> canonical element type. Tables and pictures are handled
#: separately; labels not listed (form fields, checkboxes, ...) become paragraphs.
_LABEL_TYPES: dict[str, ElementType] = {
    "title": ElementType.HEADING,
    "section_header": ElementType.HEADING,
    "list_item": ElementType.LIST,
    "formula": ElementType.FORMULA,
    "page_header": ElementType.HEADER,
    "page_footer": ElementType.FOOTER,
}


@lru_cache(maxsize=1)
def _converter() -> Any:
    """Build the converter once; loading the layout and table models is the slow part."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _split_by_page(text: str, provenance: list[Any]) -> list[tuple[str, Any]]:
    """One text part per page an item spans.

    A paragraph that runs across a page break carries one provenance entry per
    page, each with the character span it covers. Crediting the whole text to the
    first page would make the continuation look missing from the next one.
    Falls back to the first page when spans are absent or do not fit the text.
    """
    if len(provenance) == 1:
        return [(text.strip(), provenance[0])]
    parts = []
    for prov in provenance:
        start, end = (prov.charspan or (0, 0))[:2]
        part = text[start:end].strip()
        if part:
            parts.append((part, prov))
    covered = sum(len(p) for p, _ in parts)
    if not parts or covered < 0.5 * len(text.strip()):
        return [(text.strip(), provenance[0])]
    return parts


class DoclingParser:
    """Layout-model parsing through Docling, mapped onto the canonical schema."""

    parser_name: str = "docling"

    def parse(self, file_path: Path | str, mime_type: str | None = None) -> ParsedDocument:
        from docling_core.types.doc import ContentLayer, PictureItem, TableItem

        path = Path(file_path)
        started = time.perf_counter()
        document = _converter().convert(str(path)).document

        pages: dict[int, ParsedPage] = {
            number: ParsedPage(page_number=number, width=page.size.width, height=page.size.height)
            for number, page in sorted(document.pages.items())
        }
        sequence = 0

        # Text inside pictures (an org chart's box labels) and the furniture layer
        # (running headers and footers) are both skipped by default; the other
        # adapters keep them, so this one does too and types them accordingly.
        items = document.iterate_items(
            traverse_pictures=True,
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE},
        )
        for item, _depth in items:
            provenance = getattr(item, "prov", None)
            if not provenance:
                continue
            first = provenance[0]
            page = pages.setdefault(first.page_no, ParsedPage(page_number=first.page_no))
            box = first.bbox.to_top_left_origin(page_height=page.height)
            bbox = BoundingBox(x0=box.l, y0=box.t, x1=box.r, y1=box.b, page_number=first.page_no)
            sequence += 1

            if isinstance(item, TableItem):
                grid = [[cell.text.strip() for cell in row] for row in item.data.grid]
                header_rows = sum(
                    1 for row in item.data.grid if row and all(c.column_header for c in row)
                )
                headers = grid[0] if header_rows and grid else []
                body = grid[1:] if header_rows else grid
                page.tables.append(
                    ParsedTable(
                        table_id=f"dl_table_{first.page_no}_{sequence}",
                        page_number=first.page_no,
                        num_rows=item.data.num_rows,
                        num_cols=item.data.num_cols,
                        headers=headers,
                        cells=body,
                        bounding_box=bbox,
                        markdown=item.export_to_markdown(doc=document),
                    )
                )
                continue

            if isinstance(item, PictureItem):
                caption = item.caption_text(document) or None
                page.figures.append(
                    ParsedFigure(
                        figure_id=f"dl_figure_{first.page_no}_{sequence}",
                        caption=caption,
                        page_number=first.page_no,
                        bounding_box=bbox,
                    )
                )
                continue

            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue
            label = item.label.value
            element_type = _LABEL_TYPES.get(label, ElementType.PARAGRAPH)
            level = None
            if element_type == ElementType.HEADING:
                level = 1 if label == "title" else int(getattr(item, "level", 1) or 1)
            for part, prov in _split_by_page(getattr(item, "text", "") or "", provenance):
                part_page = pages.setdefault(prov.page_no, ParsedPage(page_number=prov.page_no))
                part_box = prov.bbox.to_top_left_origin(page_height=part_page.height)
                part_page.elements.append(
                    ParsedElement(
                        element_id=f"dl_elem_{prov.page_no}_{sequence}",
                        element_type=element_type,
                        text=part,
                        page_number=prov.page_no,
                        bounding_box=BoundingBox(
                            x0=part_box.l,
                            y0=part_box.t,
                            x1=part_box.r,
                            y1=part_box.b,
                            page_number=prov.page_no,
                        ),
                        level=level,
                        sequence_index=sequence,
                        metadata={"docling_label": label},
                    )
                )

        ordered = [pages[number] for number in sorted(pages)]
        return ParsedDocument(
            filename=path.name,
            file_type="pdf",
            total_pages=len(ordered),
            pages=ordered,
            metadata={"source_path": str(path), "intelligence_engine": "docling"},
            parser_name=self.parser_name,
            parsing_duration_ms=(time.perf_counter() - started) * 1000.0,
        )

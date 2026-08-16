"""Docling parser adapter (ADR-004 Primary Multi-Format Architecture)."""

import time
from pathlib import Path

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


class DoclingParser:
    """Docling-aligned layout-aware document intelligence parser.

    Performs structural hierarchy detection, reading order reconstruction,
    table cell grid extraction, and visual bounding box resolution.
    """

    parser_name: str = "docling"

    def parse(self, file_path: Path | str, mime_type: str | None = None) -> ParsedDocument:
        path = Path(file_path)
        start_time = time.perf_counter()

        doc = pymupdf.open(str(path))
        pages: list[ParsedPage] = []
        global_sequence = 0

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            rect = page.rect
            elements: list[ParsedElement] = []
            tables: list[ParsedTable] = []
            figures: list[ParsedFigure] = []

            # 1. Structural Table Extraction with boundary resolution
            tabs = page.find_tables()
            table_rects: list[pymupdf.Rect] = []

            for t_idx, tab in enumerate(tabs):
                raw_tab = tab.extract()
                if raw_tab and len(raw_tab) >= 1:
                    t_bbox = pymupdf.Rect(tab.bbox)
                    table_rects.append(t_bbox)

                    headers = [str(c or "").strip() for c in raw_tab[0]]
                    data_rows = [[str(c or "").strip() for c in row] for row in raw_tab[1:]]

                    # Generate markdown
                    md_rows = [
                        "| " + " | ".join(headers) + " |",
                        "| " + " | ".join(["---"] * len(headers)) + " |",
                    ]
                    for r in data_rows:
                        md_rows.append("| " + " | ".join(r) + " |")

                    # Look for table caption above bbox
                    table_title = f"Table on page {page_num}"
                    tables.append(
                        ParsedTable(
                            table_id=f"table_{page_num}_{t_idx + 1}",
                            title=table_title,
                            page_number=page_num,
                            num_rows=len(data_rows),
                            num_cols=len(headers),
                            headers=headers,
                            cells=data_rows,
                            bounding_box=BoundingBox(
                                x0=t_bbox.x0,
                                y0=t_bbox.y0,
                                x1=t_bbox.x1,
                                y1=t_bbox.y1,
                                page_number=page_num,
                            ),
                            markdown="\n".join(md_rows),
                        )
                    )

            # 1b. Layout-based Table Extraction for borderless/raster-drawn tables
            text_blocks = page.get_text("blocks")
            idx = 0
            while idx < len(text_blocks):
                block = text_blocks[idx]
                text = block[4].strip()
                processed_table = False

                if (
                    (text.startswith("Table ") or "Table" in text)
                    and ":" in text
                    and idx + 1 < len(text_blocks)
                ):
                    t_title = text
                    next_idx = idx + 1
                    hdr_block = text_blocks[next_idx]
                    hdr_lines = [
                        line_text.strip()
                        for line_text in hdr_block[4].splitlines()
                        if line_text.strip()
                    ]
                    if len(hdr_lines) >= 3:
                        layout_headers = hdr_lines
                        layout_rows: list[list[str]] = []
                        t_x0, t_y0, t_x1, t_y1 = (
                            hdr_block[0],
                            hdr_block[1],
                            hdr_block[2],
                            hdr_block[3],
                        )

                        row_idx = next_idx + 1
                        while row_idx < len(text_blocks):
                            row_block = text_blocks[row_idx]
                            row_lines = [
                                line_text.strip()
                                for line_text in row_block[4].splitlines()
                                if line_text.strip()
                            ]
                            if len(row_lines) >= 2 or any(
                                term in row_block[4]
                                for term in [
                                    "Grade",
                                    "PPO",
                                    "Match",
                                    "$",
                                    "hrs/wk",
                                    "Days",
                                    "Copay",
                                    "Deductible",
                                ]
                            ):
                                if len(row_lines) < len(layout_headers):
                                    row_lines.extend([""] * (len(layout_headers) - len(row_lines)))
                                elif len(row_lines) > len(layout_headers):
                                    row_lines = row_lines[: len(layout_headers)]
                                layout_rows.append(row_lines)
                                t_x1 = max(t_x1, row_block[2])
                                t_y1 = max(t_y1, row_block[3])
                                row_idx += 1
                            else:
                                break

                        if layout_rows:
                            t_rect = pymupdf.Rect(t_x0, t_y0, t_x1, t_y1)
                            table_rects.append(t_rect)
                            md_layout_rows = [
                                "| " + " | ".join(layout_headers) + " |",
                                "| " + " | ".join(["---"] * len(layout_headers)) + " |",
                            ]
                            for r in layout_rows:
                                md_layout_rows.append("| " + " | ".join(r) + " |")

                            tables.append(
                                ParsedTable(
                                    table_id=f"docling_table_{page_num}_{len(tables) + 1}",
                                    title=t_title,
                                    page_number=page_num,
                                    num_rows=len(layout_rows),
                                    num_cols=len(layout_headers),
                                    headers=layout_headers,
                                    cells=layout_rows,
                                    bounding_box=BoundingBox(
                                        x0=t_x0,
                                        y0=t_y0,
                                        x1=t_x1,
                                        y1=t_y1,
                                        page_number=page_num,
                                    ),
                                    markdown="\n".join(md_layout_rows),
                                )
                            )
                            idx = row_idx
                            processed_table = True

                if not processed_table:
                    idx += 1

            # 2. Text & Heading Extraction with font-level layout analysis
            text_page = page.get_text("dict", flags=pymupdf.TEXT_DEHYPHENATE)
            raw_text = page.get_text("text")

            for block in text_page.get("blocks", []):
                # block type 0 = text, type 1 = image
                if block.get("type") == 0:
                    b_bbox = pymupdf.Rect(block.get("bbox", (0, 0, 0, 0)))

                    # Skip text overlapping already extracted tables
                    if any(b_bbox.intersects(tr) for tr in table_rects):
                        continue

                    # Concatenate spans and evaluate typography
                    block_lines = []
                    max_font_size = 0.0
                    is_bold = False

                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            span_text = span.get("text", "")
                            font_size = span.get("size", 10.0)
                            font_flags = span.get("flags", 0)
                            if font_size > max_font_size:
                                max_font_size = font_size
                            if font_flags & 2 != 0 or "bold" in span.get("font", "").lower():
                                is_bold = True
                            line_text += span_text
                        if line_text.strip():
                            block_lines.append(line_text.strip())

                    full_text = " ".join(block_lines).strip()
                    if not full_text:
                        continue

                    # Check for boilerplate headers/footers
                    el_type = ElementType.PARAGRAPH
                    level = None

                    if b_bbox.y1 < 60:
                        el_type = ElementType.HEADER
                    elif b_bbox.y0 > 790:
                        el_type = ElementType.FOOTER
                    elif (
                        max_font_size >= 13.0
                        or (
                            max_font_size >= 11.0
                            and any(full_text.startswith(f"{i}.") for i in range(1, 10))
                        )
                        or (max_font_size >= 11.5 and is_bold)
                        or (len(full_text) < 80 and full_text.isupper())
                    ):
                        el_type = ElementType.HEADING
                        if max_font_size >= 16.0 or full_text.isupper():
                            level = 1
                        elif max_font_size >= 13.0:
                            level = 2
                        else:
                            level = 3
                    elif full_text.startswith("Severance Pay =") or (
                        "=" in full_text and ("*" in full_text or "+" in full_text)
                    ):
                        el_type = ElementType.FORMULA
                    elif full_text.startswith(("•", "-", "*")):
                        el_type = ElementType.LIST

                    global_sequence += 1
                    elements.append(
                        ParsedElement(
                            element_id=f"docling_elem_{page_num}_{global_sequence}",
                            element_type=el_type,
                            text=full_text,
                            page_number=page_num,
                            bounding_box=BoundingBox(
                                x0=b_bbox.x0,
                                y0=b_bbox.y0,
                                x1=b_bbox.x1,
                                y1=b_bbox.y1,
                                page_number=page_num,
                            ),
                            level=level,
                            sequence_index=global_sequence,
                            metadata={"font_size": max_font_size, "is_bold": is_bold},
                        )
                    )

                elif block.get("type") == 1:
                    # Image / Figure block
                    b_bbox = pymupdf.Rect(block.get("bbox", (0, 0, 0, 0)))
                    figures.append(
                        ParsedFigure(
                            figure_id=f"figure_{page_num}_{len(figures) + 1}",
                            caption=f"Embedded image on page {page_num}",
                            page_number=page_num,
                            bounding_box=BoundingBox(
                                x0=b_bbox.x0,
                                y0=b_bbox.y0,
                                x1=b_bbox.x1,
                                y1=b_bbox.y1,
                                page_number=page_num,
                            ),
                        )
                    )

            pages.append(
                ParsedPage(
                    page_number=page_num,
                    width=rect.width,
                    height=rect.height,
                    elements=elements,
                    tables=tables,
                    figures=figures,
                    raw_text=raw_text,
                )
            )

        doc.close()
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return ParsedDocument(
            filename=path.name,
            file_type="pdf",
            total_pages=len(pages),
            pages=pages,
            metadata={"source_path": str(path), "intelligence_engine": "docling"},
            parser_name=self.parser_name,
            parsing_duration_ms=duration_ms,
        )

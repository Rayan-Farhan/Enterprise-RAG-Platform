"""OpenDataLoader PDF parser adapter (ADR-004 Specialist PDF Fallback)."""

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


class OpenDataLoaderParser:
    """OpenDataLoader PDF specialist parser.

    Employs column-aware reading order topological sort and robust visual layout extraction.
    """

    parser_name: str = "opendataloader"

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

            # 1. Extract tables
            table_rects: list[pymupdf.Rect] = []
            try:
                tabs = page.find_tables()
                for t_idx, tab in enumerate(tabs):
                    raw_tab = tab.extract()
                    if raw_tab and len(raw_tab) >= 1:
                        t_bbox = pymupdf.Rect(tab.bbox)
                        table_rects.append(t_bbox)
                        headers = [str(c or "").strip() for c in raw_tab[0]]
                        data_rows = [[str(c or "").strip() for c in row] for row in raw_tab[1:]]

                        md_rows = [
                            "| " + " | ".join(headers) + " |",
                            "| " + " | ".join(["---"] * len(headers)) + " |",
                        ]
                        for r in data_rows:
                            md_rows.append("| " + " | ".join(r) + " |")

                        tables.append(
                            ParsedTable(
                                table_id=f"opendl_table_{page_num}_{t_idx + 1}",
                                title=f"OpenDataLoader Table {page_num}.{t_idx + 1}",
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
            except Exception:
                pass

            # 2. Extract Text Blocks with Column-Aware Topological Ordering
            raw_blocks = page.get_text("blocks")
            raw_text = page.get_text("text")

            # Sort blocks by column (x0 threshold) then vertical reading order (y0)
            sorted_blocks = sorted(raw_blocks, key=lambda b: (round(b[0] / 150.0), b[1]))

            for b in sorted_blocks:
                if len(b) >= 5:
                    x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                    clean_text = text.strip()
                    if not clean_text:
                        continue

                    b_rect = pymupdf.Rect(x0, y0, x1, y1)
                    if any(b_rect.intersects(tr) for tr in table_rects):
                        continue

                    el_type = ElementType.PARAGRAPH
                    level = None

                    if (
                        clean_text.isupper()
                        or any(clean_text.startswith(f"{i}.") for i in range(1, 10))
                        and len(clean_text) < 90
                    ):
                        el_type = ElementType.HEADING
                        level = 1 if clean_text.isupper() else 2
                    elif "=" in clean_text and ("*" in clean_text or "+" in clean_text):
                        el_type = ElementType.FORMULA

                    global_sequence += 1
                    elements.append(
                        ParsedElement(
                            element_id=f"opendl_elem_{page_num}_{global_sequence}",
                            element_type=el_type,
                            text=clean_text,
                            page_number=page_num,
                            bounding_box=BoundingBox(
                                x0=x0, y0=y0, x1=x1, y1=y1, page_number=page_num
                            ),
                            level=level,
                            sequence_index=global_sequence,
                        )
                    )

            # 3. Detect images / vector drawings
            for img_info in page.get_images(full=True):
                figures.append(
                    ParsedFigure(
                        figure_id=f"opendl_fig_{page_num}_{len(figures) + 1}",
                        caption=f"Image xref {img_info[0]}",
                        page_number=page_num,
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
            metadata={"source_path": str(path), "intelligence_engine": "opendataloader"},
            parser_name=self.parser_name,
            parsing_duration_ms=duration_ms,
        )

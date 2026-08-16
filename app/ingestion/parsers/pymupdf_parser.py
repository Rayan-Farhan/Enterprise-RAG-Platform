"""PyMuPDF parser adapter (Naive fast baseline)."""

import time
from pathlib import Path

import pymupdf

from app.ingestion.parsers.base import (
    BoundingBox,
    ElementType,
    ParsedDocument,
    ParsedElement,
    ParsedPage,
    ParsedTable,
)


class PyMuPDFParser:
    """Fast baseline parser using PyMuPDF (fitz) text and layout extraction."""

    parser_name: str = "pymupdf"

    def parse(self, file_path: Path | str, mime_type: str | None = None) -> ParsedDocument:
        path = Path(file_path)
        start_time = time.perf_counter()

        doc = pymupdf.open(str(path))
        pages: list[ParsedPage] = []
        element_counter = 0

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            rect = page.rect
            elements: list[ParsedElement] = []
            tables: list[ParsedTable] = []

            # Extract text blocks with PyMuPDF
            blocks = page.get_text("blocks")
            raw_text = page.get_text("text")

            # Extract tables using PyMuPDF table finder if available
            try:
                tabs = page.find_tables()
                for t_idx, tab in enumerate(tabs):
                    table_data = tab.extract()
                    if table_data and len(table_data) > 1:
                        headers = [str(c or "").strip() for c in table_data[0]]
                        cells = [[str(c or "").strip() for c in row] for row in table_data[1:]]
                        t_bbox = tab.bbox

                        # Markdown representation
                        md_lines = [
                            "| " + " | ".join(headers) + " |",
                            "| " + " | ".join(["---"] * len(headers)) + " |",
                        ]
                        for r in cells:
                            md_lines.append("| " + " | ".join(r) + " |")

                        tables.append(
                            ParsedTable(
                                table_id=f"table_{page_num}_{t_idx + 1}",
                                title=f"Table on page {page_num}",
                                page_number=page_num,
                                num_rows=len(cells),
                                num_cols=len(headers),
                                headers=headers,
                                cells=cells,
                                bounding_box=BoundingBox(
                                    x0=t_bbox[0],
                                    y0=t_bbox[1],
                                    x1=t_bbox[2],
                                    y1=t_bbox[3],
                                    page_number=page_num,
                                ),
                                markdown="\n".join(md_lines),
                            )
                        )
            except Exception:
                pass

            for b_idx, block in enumerate(blocks):
                # block: (x0, y0, x1, y1, text, block_no, block_type)
                if len(block) >= 5:
                    x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
                    clean_text = text.strip()
                    if not clean_text:
                        continue

                    element_counter += 1
                    bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, page_number=page_num)

                    # Simple naive classification
                    el_type = ElementType.PARAGRAPH
                    level = None

                    # Check for simple heading heuristic
                    if len(clean_text) < 80 and (
                        "\n" not in clean_text or clean_text.count("\n") == 0
                    ):
                        if clean_text.isupper() or any(
                            clean_text.startswith(f"{i}.") for i in range(1, 10)
                        ):
                            el_type = ElementType.HEADING
                            level = 1 if clean_text.isupper() else 2

                    if (
                        "Formula" in clean_text
                        or "=" in clean_text
                        and ("*" in clean_text or "+" in clean_text)
                    ):
                        el_type = ElementType.FORMULA

                    elements.append(
                        ParsedElement(
                            element_id=f"elem_{page_num}_{b_idx + 1}",
                            element_type=el_type,
                            text=clean_text,
                            page_number=page_num,
                            bounding_box=bbox,
                            level=level,
                            sequence_index=element_counter,
                        )
                    )

            pages.append(
                ParsedPage(
                    page_number=page_num,
                    width=rect.width,
                    height=rect.height,
                    elements=elements,
                    tables=tables,
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
            metadata={"source_path": str(path)},
            parser_name=self.parser_name,
            parsing_duration_ms=duration_ms,
        )

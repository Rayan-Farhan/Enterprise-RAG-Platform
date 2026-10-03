"""Validate the parser benchmark's references (Task 1.2).

Checks that ``annotations.json`` is well formed and covers every corpus PDF, that
every table is rectangular and matches its declared shape, and that the OCR
full-text reference exists for every page of every PDF.

    python -m benchmarks.validate_ground_truth
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent
ANNOTATIONS = ROOT / "ground_truth" / "annotations.json"
OCR_DIR = ROOT / "ground_truth" / "ocr"
CORPUS_DIR = ROOT / "corpus"


def validate() -> list[str]:
    errors: list[str] = []
    data = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    documents = {d["filename"]: d for d in data.get("documents", [])}
    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))

    for pdf in pdfs:
        with pymupdf.open(pdf) as doc:
            pages = len(doc)
        entry = documents.get(pdf.name)
        if entry is None:
            errors.append(f"{pdf.name}: no hand annotation")
        else:
            if entry.get("total_pages") != pages:
                errors.append(f"{pdf.name}: total_pages {entry.get('total_pages')} != {pages}")
            for page in entry.get("annotations", []):
                number = page.get("page_number")
                if not isinstance(number, int) or not 1 <= number <= pages:
                    errors.append(f"{pdf.name}: page_number {number!r} out of range")
                for heading in page.get("headings", []):
                    if not heading.get("text") or not isinstance(heading.get("level"), int):
                        errors.append(f"{pdf.name} p{number}: heading needs text and an int level")
                for table in page.get("tables", []):
                    cells = table.get("cells", [])
                    widths = {len(row) for row in cells}
                    if len(widths) != 1:
                        errors.append(
                            f"{pdf.name} p{number}: table {table.get('table_id')} is ragged"
                        )
                    if table.get("num_rows") != len(cells) or table.get("num_cols") not in widths:
                        errors.append(
                            f"{pdf.name} p{number}: table {table.get('table_id')} shape mismatch"
                        )
                if page.get("complete") and not page.get("paragraphs"):
                    errors.append(f"{pdf.name} p{number}: complete page without paragraphs")

        ocr_path = OCR_DIR / f"{pdf.stem}.json"
        if not ocr_path.exists():
            errors.append(f"{pdf.name}: no OCR reference (run benchmarks.ocr_reference)")
        else:
            ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
            numbers = [p["page_number"] for p in ocr.get("pages", [])]
            if numbers != list(range(1, pages + 1)):
                errors.append(f"{pdf.name}: OCR reference covers {len(numbers)} of {pages} pages")

    for name in documents:
        if not (CORPUS_DIR / name).exists():
            errors.append(f"{name}: annotated but not in the corpus")
    return errors


def main() -> int:
    errors = validate()
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"FAILED: {len(errors)} problem(s)")
        return 1
    print("OK: every corpus PDF has hand annotation and a full-page OCR reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())

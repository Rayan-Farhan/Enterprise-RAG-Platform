"""Build the parser benchmark's full-text reference from page images (Tesseract OCR).

The benchmark grades parsers, so its reference text must not come from any of
them, nor from the PDF text layer they all read. Every page of every corpus PDF
is rendered to an image and read by Tesseract, which shares no code or model
with the parsers under test. The result is a word-level reference for the whole
corpus, not a sample.

    python -m benchmarks.ocr_reference                 # all PDFs, skips pages already done
    python -m benchmarks.ocr_reference --force

Output: ``benchmarks/ground_truth/ocr/<pdf stem>.json`` with, per page, the
paragraphs Tesseract found (in its reading order) and its mean word
confidence. OCR makes its own errors; the benchmark measures that rate against
the hand-transcribed pages in ``annotations.json`` and reports it as the
reference's error margin.

Requires the Tesseract binary (``TESSERACT_CMD`` or the default Windows install
path or ``tesseract`` on PATH).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pymupdf

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
OUT_DIR = ROOT / "ground_truth" / "ocr"

#: 300 dpi is Tesseract's recommended input resolution for body text.
DPI = 300
#: Below this confidence a "word" is almost always a rule, bullet or artefact.
MIN_WORD_CONFIDENCE = 30.0


def tesseract_cmd() -> str:
    candidates = [
        os.environ.get("TESSERACT_CMD"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        shutil.which("tesseract"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("Tesseract not found; install it or set TESSERACT_CMD")


def tesseract_version(cmd: str) -> str:
    out = subprocess.run([cmd, "--version"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()[0].strip()


def ocr_page(args: tuple[str, int, str]) -> dict[str, Any]:
    """OCR one page image and group words into Tesseract's paragraphs."""
    pdf_path, page_index, cmd = args
    with pymupdf.open(pdf_path) as doc:
        png = doc[page_index].get_pixmap(dpi=DPI).tobytes("png")

    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    completed = subprocess.run(
        [cmd, "stdin", "stdout", "-l", "eng", "--psm", "3", "tsv"],
        input=png,
        capture_output=True,
        check=True,
        env=env,
    )
    rows = csv.DictReader(
        io.StringIO(completed.stdout.decode("utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE
    )

    paragraphs: dict[tuple[int, int], list[str]] = defaultdict(list)
    confidences: list[float] = []
    for row in rows:
        text = (row.get("text") or "").strip()
        conf = float(row.get("conf") or -1)
        if not text or conf < MIN_WORD_CONFIDENCE:
            continue
        paragraphs[(int(row["block_num"]), int(row["par_num"]))].append(text)
        confidences.append(conf)

    return {
        "page_number": page_index + 1,
        "paragraphs": [" ".join(words) for _, words in sorted(paragraphs.items())],
        "words": len(confidences),
        "mean_confidence": round(sum(confidences) / len(confidences), 2) if confidences else None,
    }


def build(pdf: Path, cmd: str, workers: int, force: bool) -> Path:
    out = OUT_DIR / f"{pdf.stem}.json"
    if out.exists() and not force:
        print(f"skip {pdf.name} (exists)")
        return out
    with pymupdf.open(pdf) as doc:
        total = len(doc)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pages = list(pool.map(ocr_page, [(str(pdf), i, cmd) for i in range(total)]))
    record = {
        "filename": pdf.name,
        "total_pages": total,
        "engine": tesseract_version(cmd),
        "dpi": DPI,
        "psm": 3,
        "min_word_confidence": MIN_WORD_CONFIDENCE,
        "pages": pages,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{pdf.name}: {total} pages, {sum(p['words'] for p in pages)} words")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args(argv)
    cmd = tesseract_cmd()
    for pdf in sorted(CORPUS_DIR.glob("*.pdf")):
        build(pdf, cmd, args.workers, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())

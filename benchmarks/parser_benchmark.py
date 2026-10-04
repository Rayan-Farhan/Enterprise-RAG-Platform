"""Document intelligence benchmark (Task 1.4, ADR-004, master plan §7).

Grades every PDF parser candidate against references that none of them produced:

* **Full-text reference, every page of every PDF:** Tesseract OCR of the rendered
  page images (``ground_truth/ocr/``, built by ``benchmarks.ocr_reference``).
* **Hand annotation, 23 stress pages across all 8 PDFs:** complete heading
  inventories, every table cell by position, and, on 12 pages, every paragraph
  verbatim (``ground_truth/annotations.json``), transcribed from page images.

The hand pages also measure the OCR reference's own error rate, so text scores
can be read against it.

    python -m benchmarks.parser_benchmark                 # all available parsers
    python -m benchmarks.parser_benchmark --parsers pymupdf-layout docling
    python -m benchmarks.parser_benchmark --reuse         # re-score cached parses

Parsed output is cached in ``benchmarks/.cache/`` so scoring can be revised
without re-running the slow parsers; ``--reuse`` reads it, otherwise each parser
runs fresh and is timed (sequentially, one document at a time).

Metrics, all page-aligned (a parser's text is compared with the reference text of
the same page):

* text recall / precision: word multiset overlap after normalisation (NFKC,
  quotes and dashes folded, case folded, edge punctuation stripped). Precision
  below 1 means extra words: duplicates, hidden or overprinted text, OCR misses.
* reading order: difflib similarity of the token sequences (tables appended at
  the end of each page, for every parser alike).
* paragraphs intact: share of hand-transcribed paragraphs whose tokens appear, in
  order, at >= 95% coverage.
* headings: precision / recall / F1 on the annotated pages (a parser heading that
  matches an annotated table cell is neutral, neither right nor wrong), plus the
  share of all elements typed heading across the corpus.
* tables: detection recall / precision on annotated pages, cell recall as a
  multiset and by position (best alignment over small row/column offsets).
* provenance (page and box on every element) and speed (ms/page).

Every average is reported micro (pooled over pages) and macro (mean of
per-document values), because a 1-page PDF and a 108-page one should not weigh
the same by accident.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.ingestion.parsers.base import DocumentParser, ElementType, ParsedDocument

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
ANNOTATIONS = ROOT / "ground_truth" / "annotations.json"
OCR_DIR = ROOT / "ground_truth" / "ocr"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"

PARAGRAPH_INTACT = 0.95
HEADING_MATCH = 0.85
CELL_MATCH = 0.90


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------


def _layout() -> DocumentParser:
    from app.ingestion.parsers.layout_heuristic_parser import LayoutHeuristicParser

    return LayoutHeuristicParser()


def _columns() -> DocumentParser:
    from app.ingestion.parsers.column_heuristic_parser import ColumnHeuristicParser

    return ColumnHeuristicParser()


def _plain() -> DocumentParser:
    from app.ingestion.parsers.pymupdf_parser import PyMuPDFParser

    return PyMuPDFParser()


def _docling() -> DocumentParser:
    import docling  # noqa: F401 - fail fast when the optional dependency is absent

    from app.ingestion.parsers.docling_parser import DoclingParser

    return DoclingParser()


def _opendataloader() -> DocumentParser:
    import shutil

    import opendataloader_pdf  # noqa: F401

    if shutil.which("java") is None:
        raise ImportError("OpenDataLoader needs a Java runtime on PATH")
    from app.ingestion.parsers.opendataloader_parser import OpenDataLoaderParser

    return OpenDataLoaderParser()


#: Where the hybrid backend listens (``opendataloader-pdf-hybrid --port 5002``).
HYBRID_URL = os.environ.get("ODL_HYBRID_URL", "http://127.0.0.1:5002")


def _opendataloader_hybrid() -> DocumentParser:
    import socket
    from urllib.parse import urlparse

    _opendataloader()  # same dependency checks as local mode
    target = urlparse(HYBRID_URL)
    try:
        with socket.create_connection((target.hostname or "127.0.0.1", target.port or 80), 2):
            pass
    except OSError as exc:
        raise ImportError(
            f"hybrid backend not reachable at {HYBRID_URL}; start "
            "`opendataloader-pdf-hybrid --port 5002 --heading-hierarchy`"
        ) from exc
    from app.ingestion.parsers.opendataloader_parser import OpenDataLoaderParser

    return OpenDataLoaderParser(hybrid_url=HYBRID_URL)


def _routed() -> DocumentParser:
    _docling()  # both engines must be importable for a true routed run
    _opendataloader()
    from app.ingestion.parsers.routed_parser import RoutedPdfParser

    return RoutedPdfParser()


#: name -> (factory, one-line description for the report)
PARSERS: dict[str, tuple[Callable[[], DocumentParser], str]] = {
    "pymupdf-layout": (_layout, "PyMuPDF + typography heuristics (production primary)"),
    "pymupdf-columns": (_columns, "PyMuPDF + column-sorted blocks (production fallback)"),
    "pymupdf": (_plain, "PyMuPDF text blocks, no typing (baseline)"),
    "docling": (_docling, "Docling layout + TableFormer models, OCR off"),
    "opendataloader": (_opendataloader, "OpenDataLoader PDF, rule-based (Java)"),
    "opendataloader-hybrid": (
        _opendataloader_hybrid,
        "OpenDataLoader hybrid: local engine + Docling backend for complex pages "
        "(auto triage, heading hierarchy, OCR on image regions)",
    ),
    "routed": (
        _routed,
        "Page-routed: OpenDataLoader for grid pages, Docling for prose (OCR for image-only pages)",
    ),
}


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

_FOLD = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "–": "-",
        "—": "-",
        "‐": "-",
        "‑": "-",
        "­": "",
        "�": "",
    }
)
_EDGE = "\"'`.,;:!?()[]{}<>*•●▪◦·…|_"


def tokens(text: str) -> list[str]:
    """Comparable word tokens: NFKC, folded quotes and dashes, lower case, edges trimmed."""
    text = unicodedata.normalize("NFKC", text).translate(_FOLD).lower()
    out = []
    for raw in text.split():
        word = raw.strip(_EDGE)
        if word and any(ch.isalnum() for ch in word):
            out.append(word)
    return out


def norm(text: str) -> str:
    return " ".join(tokens(text))


def similar(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def multiset_overlap(reference: Counter[str], candidate: Counter[str]) -> int:
    return sum((reference & candidate).values())


def in_order_coverage(needle: Sequence[str], haystack: Sequence[str]) -> float:
    """Share of ``needle`` found in ``haystack`` as an order-preserving match."""
    if not needle:
        return 1.0
    blocks = SequenceMatcher(None, needle, haystack, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) / len(needle)


# --------------------------------------------------------------------------
# Parsed output, reduced to what scoring needs
# --------------------------------------------------------------------------


@dataclass
class PageView:
    stream: list[str] = field(default_factory=list)  # element text in emitted order, then tables
    headings: list[tuple[str, int | None]] = field(default_factory=list)
    tables: list[list[list[str]]] = field(default_factory=list)  # full grids incl. header row
    elements: int = 0
    heading_elements: int = 0
    with_box: int = 0


@dataclass
class ParsedView:
    parser: str
    filename: str
    total_pages: int
    seconds: float
    pages: dict[int, PageView]


def view(doc: ParsedDocument, seconds: float) -> ParsedView:
    pages: dict[int, PageView] = {}
    for page in doc.pages:
        pv = pages.setdefault(page.page_number, PageView())
        for el in sorted(page.elements, key=lambda e: e.sequence_index):
            pv.stream.append(el.text)
            pv.elements += 1
            pv.with_box += int(el.bounding_box is not None and el.page_number > 0)
            if el.element_type == ElementType.HEADING:
                pv.headings.append((el.text, el.level))
                pv.heading_elements += 1
        for table in page.tables:
            grid = ([list(table.headers)] if table.headers else []) + [list(r) for r in table.cells]
            pv.tables.append(grid)
            pv.stream.extend(cell for row in grid for cell in row if cell)
            pv.elements += 1
            pv.with_box += int(table.bounding_box is not None)
    return ParsedView(doc.parser_name, doc.filename, doc.total_pages, seconds, pages)


def run_parser(name: str, pdf: Path, reuse: bool) -> ParsedView:
    cache = CACHE_DIR / name / f"{pdf.stem}.json"
    if reuse and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        doc = ParsedDocument.model_validate(data["document"])
        return view(doc, data["seconds"])
    parser = PARSERS[name][0]()
    started = time.perf_counter()
    doc = parser.parse(pdf)
    seconds = time.perf_counter() - started
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "seconds": seconds,
                "document": doc.model_dump(
                    mode="json",
                    exclude={"pages": {"__all__": {"figures": {"__all__": {"image_bytes"}}}}},
                ),
            }
        ),
        encoding="utf-8",
    )
    return view(doc, seconds)


# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------


def load_references() -> tuple[
    dict[str, dict[int, list[str]]], dict[str, dict[int, dict[str, Any]]], dict[str, Any]
]:
    ocr: dict[str, dict[int, list[str]]] = {}
    meta: dict[str, Any] = {}
    for path in sorted(OCR_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        ocr[data["filename"]] = {
            p["page_number"]: tokens(" ".join(p["paragraphs"])) for p in data["pages"]
        }
        meta = {"engine": data["engine"], "dpi": data["dpi"]}
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    hand: dict[str, dict[int, dict[str, Any]]] = {
        d["filename"]: {a["page_number"]: a for a in d["annotations"]}
        for d in annotations["documents"]
    }
    return ocr, hand, meta


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@dataclass
class Tally:
    """Numerators and denominators, so micro and macro averages come from one pass."""

    counts: Counter[str] = field(default_factory=Counter)

    def add(self, **values: float) -> None:
        for key, value in values.items():
            self.counts[key] += value

    def ratio(self, num: str, den: str) -> float | None:
        return self.counts[num] / self.counts[den] if self.counts[den] else None


def match_headings(
    expected: list[str], found: list[str], neutral: set[str]
) -> tuple[int, int, int]:
    """Greedy one-to-one fuzzy match: (true positives, scored found, expected)."""
    remaining = list(range(len(expected)))
    tp = 0
    scored = 0
    for text in found:
        best, best_score = None, 0.0
        for i in remaining:
            score = similar(text, expected[i])
            if score > best_score:
                best, best_score = i, score
        if best is not None and best_score >= HEADING_MATCH:
            remaining.remove(best)
            tp += 1
            scored += 1
        elif any(similar(text, n) >= HEADING_MATCH or (text and text in n) for n in neutral):
            continue
        else:
            scored += 1
    return tp, scored, len(expected)


def grid_norm(grid: list[list[str]]) -> list[list[str]]:
    return [[norm(c) for c in row] for row in grid]


def positional_hits(gt: list[list[str]], cand: list[list[str]]) -> int:
    """Best count of GT cells matched at the same aligned position over small offsets."""
    best = 0
    for dr in range(-2, 3):
        for dc in range(-1, 2):
            hits = 0
            for r, row in enumerate(gt):
                for c, cell in enumerate(row):
                    if not cell:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < len(cand) and 0 <= cc < len(cand[rr]):
                        if similar(cell, cand[rr][cc]) >= CELL_MATCH:
                            hits += 1
            best = max(best, hits)
    return best


def score(
    views: dict[str, ParsedView],
    ocr: dict[str, dict[int, list[str]]],
    hand: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    by_doc: dict[str, Tally] = {}
    total = Tally()
    for filename, pv in views.items():
        doc = Tally()
        doc.add(pages=pv.total_pages, seconds=pv.seconds)
        ref_pages = ocr.get(filename, {})
        for number, ref in ref_pages.items():
            page = pv.pages.get(number, PageView())
            cand = tokens(" ".join(page.stream))
            ref_c, cand_c = Counter(ref), Counter(cand)
            overlap = multiset_overlap(ref_c, cand_c)
            order = SequenceMatcher(None, ref, cand, autojunk=False).ratio() if ref else 1.0
            doc.add(
                ref_words=len(ref), cand_words=len(cand), overlap=overlap, order_w=order * len(ref)
            )
        for page in pv.pages.values():
            doc.add(
                elements=page.elements,
                heading_elements=page.heading_elements,
                with_box=page.with_box,
            )

        for number, ann in hand.get(filename, {}).items():
            page = pv.pages.get(number, PageView())
            stream_tokens = tokens(" ".join(page.stream))
            if ann.get("complete"):
                hand_tokens = tokens(" ".join(ann["paragraphs"]))
                doc.add(
                    hand_words=len(hand_tokens),
                    hand_overlap=multiset_overlap(Counter(hand_tokens), Counter(stream_tokens)),
                )
                for para in ann["paragraphs"]:
                    ptoks = tokens(para)
                    doc.add(
                        paragraphs=1,
                        intact=int(in_order_coverage(ptoks, stream_tokens) >= PARAGRAPH_INTACT),
                    )
            neutral = {norm(c) for t in ann["tables"] for row in t["cells"] for c in row if c}
            tp, scored, expected = match_headings(
                [norm(h["text"]) for h in ann["headings"]],
                [norm(h) for h, _ in page.headings],
                neutral,
            )
            doc.add(h_tp=tp, h_found=scored, h_expected=expected)

            cand_grids = [grid_norm(g) for g in page.tables]
            used: set[int] = set()
            for table in ann["tables"]:
                gt = grid_norm(table["cells"])
                gt_cells = Counter(c for row in gt for c in row if c)
                doc.add(t_expected=1, cells=sum(gt_cells.values()))
                best_i, best_overlap = None, 0
                for i, grid in enumerate(cand_grids):
                    if i in used:
                        continue
                    cand_cells = Counter(c for row in grid for c in row if c)
                    overlap = multiset_overlap(gt_cells, cand_cells)
                    if overlap > best_overlap:
                        best_i, best_overlap = i, overlap
                if best_i is not None and best_overlap >= 0.5 * sum(gt_cells.values()):
                    used.add(best_i)
                    doc.add(
                        t_found=1,
                        cell_hits=best_overlap,
                        cell_pos_hits=positional_hits(gt, cand_grids[best_i]),
                    )
            doc.add(t_cand=len(cand_grids), t_matched=len(used))
        by_doc[filename] = doc
        total.add(**doc.counts)

    def summarise(t: Tally) -> dict[str, float | None]:
        hp, hr = t.ratio("h_tp", "h_found"), t.ratio("h_tp", "h_expected")
        return {
            "text_recall": t.ratio("overlap", "ref_words"),
            "text_precision": t.ratio("overlap", "cand_words"),
            "reading_order": t.ratio("order_w", "ref_words"),
            "hand_text_recall": t.ratio("hand_overlap", "hand_words"),
            "paragraphs_intact": t.ratio("intact", "paragraphs"),
            "heading_precision": hp,
            "heading_recall": hr,
            "heading_f1": (2 * hp * hr / (hp + hr))
            if hp and hr
            else (0.0 if hp is not None and hr is not None else None),
            "heading_share": t.ratio("heading_elements", "elements"),
            "table_recall": t.ratio("t_found", "t_expected"),
            "table_precision": t.ratio("t_matched", "t_cand"),
            "cell_recall": t.ratio("cell_hits", "cells"),
            "cell_position_recall": t.ratio("cell_pos_hits", "cells"),
            "provenance": t.ratio("with_box", "elements"),
            "ms_per_page": (1000 * t.counts["seconds"] / t.counts["pages"])
            if t.counts["pages"]
            else None,
        }

    per_doc = {name: summarise(t) for name, t in by_doc.items()}
    macro: dict[str, float | None] = {}
    for key in summarise(total):
        values = [d[key] for d in per_doc.values() if d[key] is not None]
        macro[key] = sum(values) / len(values) if values else None
    return {
        "micro": summarise(total),
        "macro": macro,
        "per_document": per_doc,
        "counts": dict(total.counts),
    }


def ocr_calibration(
    ocr: dict[str, dict[int, list[str]]], hand: dict[str, dict[int, dict[str, Any]]]
) -> dict[str, float]:
    """How well the OCR reference itself recovers the hand-transcribed pages."""
    hand_words = overlap = ocr_words = 0
    for filename, pages in hand.items():
        for number, ann in pages.items():
            if not ann.get("complete"):
                continue
            h = Counter(tokens(" ".join(ann["paragraphs"] + [x["text"] for x in ann["headings"]])))
            o = Counter(ocr[filename][number])
            hand_words += sum(h.values())
            ocr_words += sum(o.values())
            overlap += multiset_overlap(h, o)
    return {
        "recall": overlap / hand_words,
        "hand_words": hand_words,
        "pages": sum(1 for p in hand.values() for a in p.values() if a.get("complete")),
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def render(
    results: dict[str, Any],
    coverage: dict[str, Any],
    calibration: dict[str, float],
    skipped: dict[str, str],
    meta: dict[str, Any],
) -> str:
    names = list(results)
    lines = [
        "# Parser Benchmark Report (ADR-004)",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')} · `python -m benchmarks.parser_benchmark`",
        "",
        "## Coverage",
        "",
        f"* **Full-text reference:** {coverage['docs']} PDFs, {coverage['pages']} pages, "
        f"{coverage['ocr_words']:,} words — every page, OCR of rendered page images "
        f"({meta.get('engine', 'Tesseract')}, {meta.get('dpi')} dpi).",
        f"* **Hand annotation:** {coverage['hand_pages']} pages across all {coverage['docs']} PDFs: "
        f"{coverage['headings']} headings, {coverage['tables']} tables / {coverage['cells']} cells, "
        f"and {coverage['hand_paragraph_pages']} pages with every paragraph verbatim "
        f"({coverage['hand_words']:,} words).",
        f"* **Reference error margin:** the OCR reference recovers {pct(calibration['recall'])} of the "
        f"words on the {calibration['pages']} verbatim hand pages. Text recall near that ceiling is "
        "as good as this reference can show; the hand-page column has no OCR in it.",
        "",
        "## Parsers",
        "",
    ]
    for name in names:
        lines.append(f"* `{name}` — {PARSERS[name][1]}")
    for name, reason in skipped.items():
        lines.append(f"* `{name}` — **not run**: {reason}")
    rows = [
        ("Text recall vs OCR (micro)", "text_recall", "micro", pct),
        ("Text recall vs OCR (macro, per doc)", "text_recall", "macro", pct),
        ("Text recall vs hand pages", "hand_text_recall", "micro", pct),
        ("Text precision (1 − extra words)", "text_precision", "micro", pct),
        ("Reading-order similarity", "reading_order", "micro", pct),
        ("Paragraphs intact (≥95% in order)", "paragraphs_intact", "micro", pct),
        ("Heading precision", "heading_precision", "micro", pct),
        ("Heading recall", "heading_recall", "micro", pct),
        ("Heading F1", "heading_f1", "micro", pct),
        ("Elements typed heading (corpus)", "heading_share", "micro", pct),
        ("Tables found (annotated pages)", "table_recall", "micro", pct),
        ("Table precision (annotated pages)", "table_precision", "micro", pct),
        ("Cell recall (multiset)", "cell_recall", "micro", pct),
        ("Cell recall (by position)", "cell_position_recall", "micro", pct),
        ("Page + box provenance", "provenance", "micro", pct),
        ("Speed (ms/page)", "ms_per_page", "micro", lambda v: "n/a" if v is None else f"{v:,.0f}"),
    ]
    lines += [
        "",
        "## Scorecard",
        "",
        "| metric | " + " | ".join(f"`{n}`" for n in names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for label, key, scope, fmt in rows:
        lines.append(
            f"| {label} | " + " | ".join(fmt(results[n][scope][key]) for n in names) + " |"
        )
    lines += ["", "## Per document (micro within the document)", ""]
    for key, label in (
        ("text_recall", "text recall"),
        ("heading_f1", "heading F1"),
        ("cell_position_recall", "cell recall by position"),
    ):
        lines += [
            f"**{label}**",
            "",
            "| document | " + " | ".join(f"`{n}`" for n in names) + " |",
            "|---|" + "---:|" * len(names),
        ]
        for doc in sorted(next(iter(results.values()))["per_document"]):
            lines.append(
                f"| `{doc}` | "
                + " | ".join(pct(results[n]["per_document"][doc][key]) for n in names)
                + " |"
            )
        lines.append("")
    lines += [
        "## How to read this",
        "",
        "* All parsers read the same born-digital PDFs; none runs OCR. Text recall below the OCR "
        "ceiling is text the parser lost or garbled.",
        "* Text precision below 100% means words the reference does not have on that page: "
        "duplicated blocks, hidden or overprinted text in the PDF, or words OCR missed.",
        "* Heading scores cover the annotated pages only (complete inventories). A parser heading "
        "that matches an annotated table cell is neutral.",
        "* Speed is wall-clock for one sequential parse per document on the development laptop "
        "(Intel i3-1215U, 8 GB, no GPU), including model loading for the first Docling document.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parsers", nargs="+", choices=list(PARSERS), default=list(PARSERS))
    parser.add_argument(
        "--reuse", action="store_true", help="score cached parses instead of re-running"
    )
    args = parser.parse_args(argv)

    ocr, hand, meta = load_references()
    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    missing = [p.name for p in pdfs if p.name not in ocr or p.name not in hand]
    if missing:
        print(
            f"references missing for: {missing}; run benchmarks.ocr_reference and check annotations"
        )
        return 1

    results: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for name in args.parsers:
        cached = all((CACHE_DIR / name / f"{pdf.stem}.json").exists() for pdf in pdfs)
        try:
            # Rescoring a fully cached parser needs neither its engine nor its
            # backend server; only a run that will parse checks they exist.
            if not (args.reuse and cached):
                PARSERS[name][0]()
        except ImportError as exc:
            skipped[name] = str(exc)
            print(f"skip {name}: {exc}")
            continue
        views = {}
        for pdf in pdfs:
            pv = run_parser(name, pdf, args.reuse)
            views[pdf.name] = pv
            print(f"{name:<16} {pdf.name[:40]:<40} {pv.seconds:7.1f}s")
        results[name] = score(views, ocr, hand)

    calibration = ocr_calibration(ocr, hand)
    annotated = [a for pages in hand.values() for a in pages.values()]
    coverage = {
        "docs": len(pdfs),
        "pages": sum(len(p) for p in ocr.values()),
        "ocr_words": sum(len(t) for p in ocr.values() for t in p.values()),
        "hand_pages": len(annotated),
        "headings": sum(len(a["headings"]) for a in annotated),
        "tables": sum(len(a["tables"]) for a in annotated),
        "cells": sum(1 for a in annotated for t in a["tables"] for r in t["cells"] for c in r if c),
        "hand_paragraph_pages": sum(1 for a in annotated if a.get("complete")),
        "hand_words": calibration["hand_words"],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "parser_benchmark_results.json").write_text(
        json.dumps(
            {
                "coverage": coverage,
                "ocr_calibration": calibration,
                "reference": meta,
                "skipped": skipped,
                "results": results,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    (RESULTS_DIR / "parser_benchmark_report.md").write_text(
        render(results, coverage, calibration, skipped, meta), encoding="utf-8"
    )
    print(f"report: {RESULTS_DIR / 'parser_benchmark_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

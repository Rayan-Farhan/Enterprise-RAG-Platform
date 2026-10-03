"""Re-derive the Stage 3 parsing figures for one document, offline.

The Stage 3 record quotes, for the staff handbook: 56 pages and 311 elements, all
with a bounding box; 62% of elements typed heading; 137 of 194 "level 3
headings" really body text; and indexed tokens rising from 10,403 to 34,465 once
the chunker stopped trusting implausibly long headings. This recomputes each one
with the production parser, the canonical adapter, the boilerplate pass and the
fixed 512/64 chunker, with no database or network.

    python -m scripts.recheck_stage3_parsing
    python -m scripts.recheck_stage3_parsing --pdf benchmarks/corpus/staff_handbook.pdf

"Body text" for a level-3 heading means longer than the chunker's
``MAX_HEADING_TOKENS`` (the guard that fixed it), which is how the original
figure was defined.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.ingestion.adapters.canonical_adapter import CanonicalAdapter
from app.ingestion.chunking.base import estimate_tokens
from app.ingestion.chunking.fixed_size import FixedSizeChunker
from app.ingestion.dedup import BoilerplateDetector
from app.ingestion.parsers.base import ElementType
from app.ingestion.parsers.layout_heuristic_parser import LayoutHeuristicParser


class _TrustingChunker(FixedSizeChunker):
    """The chunker as it was before the heading guard: every heading is a heading."""

    MAX_HEADING_TOKENS = 10**9


def recheck(pdf: Path) -> dict[str, object]:
    parsed = LayoutHeuristicParser().parse(pdf)
    elements = parsed.all_elements
    tables = parsed.all_tables
    headings = [e for e in elements if e.element_type == ElementType.HEADING]
    level3 = [h for h in headings if h.level == 3]
    level3_body = [
        h for h in level3 if estimate_tokens(h.text) > FixedSizeChunker.MAX_HEADING_TOKENS
    ]

    _, _, pages, canonical, _ = CanonicalAdapter.to_canonical_models(
        parsed, file_hash="recheck", storage_key=str(pdf)
    )
    BoilerplateDetector().detect_and_flag(canonical, total_pages=len(pages))

    def indexed_tokens(chunker: FixedSizeChunker) -> tuple[int, int]:
        chunks = chunker.chunk(canonical)
        return len(chunks), sum(c.token_count for c in chunks)

    guarded = indexed_tokens(FixedSizeChunker(512, 64, "fixed-recheck"))
    trusting = indexed_tokens(_TrustingChunker(512, 64, "fixed-recheck"))

    return {
        "file": pdf.name,
        "parser": parsed.parser_name,
        "pages": parsed.total_pages,
        "elements": len(elements),
        "tables": len(tables),
        "canonical_elements": len(canonical),
        "elements_with_bbox": sum(1 for e in elements if e.bounding_box is not None),
        "heading_elements": len(headings),
        "heading_share": round(len(headings) / len(elements), 4) if elements else None,
        "level3_headings": len(level3),
        "level3_longer_than_guard": len(level3_body),
        "longest_level3_chars": max((len(h.text) for h in level3), default=0),
        "boilerplate_elements": sum(1 for e in canonical if e.is_boilerplate),
        "chunks_without_guard": trusting[0],
        "indexed_tokens_without_guard": trusting[1],
        "chunks_with_guard": guarded[0],
        "indexed_tokens_with_guard": guarded[1],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, default=Path("benchmarks/corpus/staff_handbook.pdf"))
    args = parser.parse_args(argv)
    print(json.dumps(recheck(args.pdf), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

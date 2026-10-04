"""Does annotated page text reach the index? (results verification, Phase 2/3)

A parser can extract a paragraph and the pipeline can still lose it: the
chunker moves heading-typed text into ``section_path``, drops boilerplate, or
truncates. Retrieval only ever sees chunk ``content``, so that is what this
checks, against the benchmark's hand-transcribed pages
(``benchmarks/ground_truth/annotations.json``, pages with ``complete: true``).

For each transcribed paragraph, the chunks of the same document whose page span
touches the page are joined in chunk order; the paragraph counts as indexed when
>= 95% of its tokens appear there in order. Each annotated heading is checked
against chunk content and ``section_path`` (where headings are meant to go).

    python -m scripts.check_chunk_coverage                        # production CHUNKING_VERSION
    python -m scripts.check_chunk_coverage --chunking-version fixed-s512-o64

Needs PostgreSQL with the corpus ingested.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models.chunk import Chunk
from app.db.models.document import Document
from app.db.session import get_session_factory
from benchmarks.parser_benchmark import ANNOTATIONS, in_order_coverage, norm, tokens

INDEXED = 0.95


async def coverage(chunking_version: str) -> dict[str, Any]:
    annotations = json.loads(Path(ANNOTATIONS).read_text(encoding="utf-8"))
    per_doc: dict[str, dict[str, int]] = {}
    lost: list[dict[str, Any]] = []
    async with get_session_factory()() as session:
        for doc in annotations["documents"]:
            document = (
                await session.execute(select(Document).where(Document.title == doc["filename"]))
            ).scalar_one_or_none()
            if document is None:
                per_doc[doc["filename"]] = {"missing_document": 1}
                continue
            chunks = (
                (
                    await session.execute(
                        select(Chunk)
                        .where(
                            Chunk.document_id == document.id,
                            Chunk.chunking_version == chunking_version,
                        )
                        .order_by(Chunk.chunk_index)
                    )
                )
                .scalars()
                .all()
            )
            by_page: dict[int, list[Chunk]] = defaultdict(list)
            for chunk in chunks:
                for page in chunk.page_span or [chunk.primary_page_number]:
                    by_page[int(page)].append(chunk)

            tally = {"paragraphs": 0, "indexed": 0, "headings": 0, "headings_found": 0}
            for page in doc["annotations"]:
                number = page["page_number"]
                nearby = sorted(
                    {
                        c.id: c
                        for n in (number - 1, number, number + 1)
                        for c in by_page.get(n, [])
                    }.values(),
                    key=lambda c: c.chunk_index,
                )
                stream = tokens(" ".join(c.content for c in nearby))
                sections = {norm(part) for c in nearby for part in (c.section_path or [])}
                content = norm(" ".join(c.content for c in nearby))
                if page.get("complete"):
                    for paragraph in page["paragraphs"]:
                        tally["paragraphs"] += 1
                        score = in_order_coverage(tokens(paragraph), stream)
                        if score >= INDEXED:
                            tally["indexed"] += 1
                        else:
                            lost.append(
                                {
                                    "file": doc["filename"],
                                    "page": number,
                                    "coverage": round(score, 3),
                                    "paragraph": paragraph[:120],
                                }
                            )
                for heading in page["headings"]:
                    tally["headings"] += 1
                    text = norm(heading["text"])
                    if text and (text in content or text in sections):
                        tally["headings_found"] += 1
            per_doc[doc["filename"]] = tally

    totals: dict[str, int] = defaultdict(int)
    for tally in per_doc.values():
        for key, value in tally.items():
            totals[key] += value
    return {
        "chunking_version": chunking_version,
        "paragraphs_indexed": f"{totals['indexed']}/{totals['paragraphs']}",
        "headings_found": f"{totals['headings_found']}/{totals['headings']}",
        "per_document": per_doc,
        "not_indexed": lost,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunking-version", default=None)
    args = parser.parse_args(argv)
    version = args.chunking_version or get_settings().CHUNKING_VERSION
    print(json.dumps(asyncio.run(coverage(version)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

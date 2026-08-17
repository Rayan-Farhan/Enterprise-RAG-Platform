"""Build the golden evaluation dataset from the ingested corpus (Task 4.1).

The questions and reference answers below are hand-authored against the real HR
corpus in ``benchmarks/corpus/``. What is *not* hand-authored is the machinery
that turns an ``element_id`` into a complete :class:`ExpectedEvidence` record:
document and version UUIDs, page numbers, and section paths are all resolved
from PostgreSQL at build time, and an ``element_id`` that does not exist aborts
the build.

That split matters. Hand-copied UUIDs rot the moment the corpus is re-ingested
into a fresh database, and a stale pointer does not announce itself — it just
makes a perfect retrieval score as a miss for the rest of the roadmap. Keeping
the human input at the level of "this fact lives in these elements of this file"
means the dataset can be regenerated against any corpus rebuild:

    python -m scripts.ingest_corpus          # rebuild the corpus
    python -m scripts.build_golden_dataset   # rebuild the dataset against it
    make eval-validate SPLIT=dev             # confirm every pointer resolves

``section_path`` is denormalised from the chunks that currently contain each
element, so it is refreshed by a rebuild. ``element_ids`` remain the
authoritative key (ADR-036) precisely because Stage 5 re-chunks the corpus.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import select

from app.core.logging import setup_logging
from app.db.models.chunk import Chunk
from app.db.models.document import Document
from app.db.models.element import Element
from app.db.models.version import DocumentVersion
from app.db.session import get_session_factory
from app.evaluation.dataset import summarize, write_split
from app.evaluation.schemas import (
    DatasetSplit,
    Difficulty,
    ExpectedEvidence,
    GoldenQuestion,
    QuestionType,
)

# Short aliases so the question table below stays readable.
STAFF = "staff_handbook.pdf"
POLICY = "university_employee_policy_manual_and_handbook.pdf"
FACULTY = "una-faculty-handbook-2026-27-initial-version.8-1-26.pdf"
HEALTH_GLANCE = "health_plan_at_a_glance_2026.pdf"
DENTAL_GLANCE = "dental_plan_at_a_glance_2026.pdf"
HEALTH_BOOK = "bcbs_health_booklet_2026.pdf"
DENTAL_BOOK = "bcbs_dental_booklet_2026.pdf"
ORG = "organizational_structure.pdf"

E = Difficulty.EASY
M = Difficulty.MEDIUM
H = Difficulty.HARD


@dataclass(frozen=True)
class Ev:
    """Authored evidence: which elements of which file contain the answer."""

    filename: str
    element_ids: tuple[str, ...]
    quote: str | None = None


@dataclass(frozen=True)
class Q:
    """One authored question, before corpus resolution."""

    question_id: str
    question: str
    question_type: QuestionType
    split: DatasetSplit
    acceptable_answer: str
    evidence: tuple[Ev, ...] = ()
    difficulty: Difficulty = M
    required_citations: int = 1
    must_contain: tuple[str, ...] = ()
    must_abstain: bool = False
    variants: tuple[str, ...] = ()
    notes: str | None = None
    fails_until: int | None = None
    extra_docs: tuple[str, ...] = field(default=())


def q(  # noqa: PLR0913 - a wide authoring signature beats a dict of magic keys
    question_id: str,
    question: str,
    question_type: QuestionType,
    split: DatasetSplit,
    acceptable_answer: str,
    evidence: Sequence[Ev] = (),
    difficulty: Difficulty = M,
    required_citations: int = 1,
    must_contain: Sequence[str] = (),
    must_abstain: bool = False,
    variants: Sequence[str] = (),
    notes: str | None = None,
    fails_until: int | None = None,
) -> Q:
    return Q(
        question_id=question_id,
        question=question,
        question_type=question_type,
        split=split,
        acceptable_answer=acceptable_answer,
        evidence=tuple(evidence),
        difficulty=difficulty,
        required_citations=required_citations,
        must_contain=tuple(must_contain),
        must_abstain=must_abstain,
        variants=tuple(variants),
        notes=notes,
        fails_until=fails_until,
    )


# --------------------------------------------------------------------------
# Corpus resolution
# --------------------------------------------------------------------------


@dataclass
class CorpusIndex:
    """Everything the builder needs to turn an ``Ev`` into ``ExpectedEvidence``."""

    document_ids: dict[str, uuid.UUID]
    version_ids: dict[str, uuid.UUID]
    titles: dict[str, str]
    pages: dict[tuple[str, str], int]
    sections: dict[tuple[str, str], list[str]]
    chunked: set[tuple[str, str]]

    def known(self, filename: str, element_id: str) -> bool:
        return (filename, element_id) in self.pages


async def load_corpus_index() -> CorpusIndex:
    """Read documents, elements, and chunk section paths into memory.

    One pass per table rather than a query per evidence pointer: the dataset has
    a few hundred pointers and the resolution step should not be the slow part of
    an authoring loop.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        document_rows = (
            await session.execute(
                select(Document.id, Document.title, DocumentVersion.id).join(
                    DocumentVersion, DocumentVersion.document_id == Document.id
                )
            )
        ).all()

        document_ids = {title: doc_id for doc_id, title, _ in document_rows}
        version_ids = {title: version_id for _, title, version_id in document_rows}
        titles = {title: title for _, title, _ in document_rows}
        version_to_title = {version_id: title for _, title, version_id in document_rows}

        element_rows = (
            await session.execute(
                select(Element.version_id, Element.element_id, Element.page_number)
            )
        ).all()
        pages = {
            (version_to_title[version_id], element_id): page
            for version_id, element_id, page in element_rows
            if version_id in version_to_title
        }

        chunk_rows = (
            await session.execute(select(Chunk.version_id, Chunk.element_ids, Chunk.section_path))
        ).all()

    # An element can sit in several chunks; union their section paths so a
    # boundary-straddling element is not scored against an arbitrary half.
    section_sets: dict[tuple[str, str], list[str]] = defaultdict(list)
    chunked: set[tuple[str, str]] = set()
    for version_id, element_ids, section_path in chunk_rows:
        title = version_to_title.get(version_id)
        if title is None:
            continue
        for element_id in element_ids or []:
            chunked.add((title, element_id))
            existing = section_sets[(title, element_id)]
            for part in section_path or []:
                if part not in existing:
                    existing.append(part)

    return CorpusIndex(
        chunked=chunked,
        document_ids=document_ids,
        version_ids=version_ids,
        titles=titles,
        pages=pages,
        sections=dict(section_sets),
    )


def resolve(authored: Q, index: CorpusIndex) -> GoldenQuestion:
    """Turn one authored question into a validated :class:`GoldenQuestion`."""
    evidence: list[ExpectedEvidence] = []

    for item in authored.evidence:
        if item.filename not in index.version_ids:
            raise SystemExit(
                f"{authored.question_id}: '{item.filename}' is not in the corpus. "
                f"Run `python -m scripts.ingest_corpus` first."
            )

        missing = [e for e in item.element_ids if not index.known(item.filename, e)]
        if missing:
            raise SystemExit(
                f"{authored.question_id}: {item.filename} has no elements {missing}. "
                f"The corpus may have been re-parsed under a different parser version."
            )

        page_numbers = sorted({index.pages[(item.filename, e)] for e in item.element_ids})
        section_path: list[str] = []
        for element_id in item.element_ids:
            for part in index.sections.get((item.filename, element_id), []):
                if part not in section_path:
                    section_path.append(part)

        evidence.append(
            ExpectedEvidence(
                document_id=index.document_ids[item.filename],
                version_id=index.version_ids[item.filename],
                element_ids=list(item.element_ids),
                page_numbers=page_numbers,
                section_path=section_path,
                document_title=index.titles[item.filename],
                quote=item.quote,
            )
        )

    return GoldenQuestion(
        question_id=authored.question_id,
        question=authored.question,
        question_type=authored.question_type,
        difficulty=authored.difficulty,
        split=authored.split,
        expected_evidence=evidence,
        acceptable_answer=authored.acceptable_answer,
        acceptable_answer_variants=list(authored.variants),
        required_citations=authored.required_citations,
        must_contain=list(authored.must_contain),
        must_abstain=authored.must_abstain,
        source="hand",
        notes=authored.notes,
        expected_to_fail_until_stage=authored.fails_until,
    )


# --------------------------------------------------------------------------


def all_questions() -> list[Q]:
    """Every authored question across the three splits."""
    from scripts.golden_questions import DEV, TEST, VALIDATION

    return [*DEV, *VALIDATION, *TEST]


async def main_async(args: argparse.Namespace) -> int:
    setup_logging()
    index = await load_corpus_index()

    authored = all_questions()
    seen: set[str] = set()
    for item in authored:
        if item.question_id in seen:
            raise SystemExit(f"Duplicate question_id: {item.question_id}")
        seen.add(item.question_id)

    resolved = [resolve(item, index) for item in authored]

    # An element that belongs to no chunk cannot be retrieved by any
    # configuration, so a question resting on one measures the chunker rather
    # than the retriever. Cover pages and tables of contents are the usual
    # culprits. Surfaced loudly here rather than discovered as an unexplained
    # zero three stages later.
    orphans = [
        (item.question_id, ev.filename, element_id)
        for item in authored
        for ev in item.evidence
        for element_id in ev.element_ids
        if (ev.filename, element_id) not in index.chunked
    ]
    if orphans:
        print(f"\n{len(orphans)} evidence elements belong to no chunk:", file=sys.stderr)
        for question_id, filename, element_id in orphans:
            print(f"  {question_id}: {filename} / {element_id}", file=sys.stderr)
        if args.strict:
            return 1

    for split in DatasetSplit:
        questions = [item for item in resolved if item.split is split]
        if not questions:
            print(f"{split.value}: no questions authored, skipping", file=sys.stderr)
            continue

        stats = summarize(questions, split=split, version=args.dataset_version)
        if not args.dry_run:
            path = write_split(questions, split=split, version=args.dataset_version)
            print(f"\nWrote {path}")
        else:
            print(f"\n{split.value} (dry run)")

        print(f"  {stats.total} questions across {stats.documents_covered} documents")
        for name, count in stats.by_type.items():
            print(f"    {name:<24}{count:>4}")
        print(f"  abstention cases: {stats.abstention_cases}")
        if stats.missing_types:
            print(f"  MISSING TYPES: {', '.join(stats.missing_types)}", file=sys.stderr)
            return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_golden_dataset", description=__doc__)
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--dry-run", action="store_true", help="resolve and report, write nothing")
    parser.add_argument(
        "--strict", action="store_true", help="fail when any evidence element belongs to no chunk"
    )
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

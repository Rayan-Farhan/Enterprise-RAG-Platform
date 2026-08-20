"""Chunking experiment matrix (Task 5.3, ADR-006).

Runs one cell per (strategy, chunk size, overlap): re-chunk the corpus, index the
chunks, then score retrieval against the dev split. Results land in
``evaluation/results/`` as ``experiment-002-<strategy>-<size>-<overlap>``.

    python -m scripts.run_chunking_sweep --phase strategies
    python -m scripts.run_chunking_sweep --phase parameters --strategy hierarchical
    python -m scripts.run_chunking_sweep --report

**Cells are scored on retrieval only.** Stage 5's exit gate is a retrieval
comparison, and generating an answer for every question of every cell would cost
a day of free-tier quota per cell to produce numbers the comparison never reads.
The winner is re-run with generation afterwards, once, to confirm the Layer 1
metrics did not move the wrong way.

**Strategies do not contaminate each other.** Chunk rows are scoped by
``chunking_version`` and vector search filters on the same field, so all cells
share one Qdrant collection without mixing. The settings validator enforces that
a version string names its strategy, which is what makes that scoping reliable
rather than a convention.

**Every step is idempotent.** Chunk IDs and vector point IDs are deterministic
(ADR-036), so re-running a cell re-chunks nothing and re-embeds nothing. A cell
whose result file already exists is skipped unless ``--force`` is given, which
matters because embedding is the metered part of this.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from sqlalchemy import select

from app.core.config import AppSettings
from app.core.logging import get_logger, setup_logging
from app.db.models.version import DocumentVersion
from app.db.session import get_session_factory
from app.evaluation import storage
from app.evaluation.dataset import load_split
from app.evaluation.runner import ExperimentRunner
from app.evaluation.schemas import DatasetSplit
from app.generation.context import ContextAssembler
from app.generation.service import GenerationService
from app.ingestion.chunking.service import ChunkingService
from app.retrieval.dense import DenseRetriever
from app.retrieval.expansion import ParentExpander
from app.retrieval.indexer import ChunkIndexer

logger = get_logger("scripts.chunking_sweep")

STRATEGIES = ("fixed", "structure_aware", "hierarchical", "contextual")

#: Phase 1 holds size and overlap at the Stage 3 values so the only thing that
#: varies is the strategy. Sweeping both at once would leave a winning cell
#: unattributable — strategy or parameters, no way to tell from the numbers.
STRATEGY_PHASE = [(s, 512, 64) for s in STRATEGIES]

#: Phase 2 sweeps size and overlap for one strategy, normally the phase 1 winner.
PARAMETER_PHASE = [(256, 32), (512, 64), (768, 96), (1024, 128)]


@dataclass(frozen=True)
class Cell:
    strategy: str
    size: int
    overlap: int

    @property
    def chunking_version(self) -> str:
        return f"{self.strategy}-s{self.size}-o{self.overlap}"

    @property
    def experiment_name(self) -> str:
        return f"experiment-002-{self.strategy}-{self.size}-{self.overlap}"

    def settings(self) -> AppSettings:
        return AppSettings(
            CHUNKING_STRATEGY=self.strategy,
            CHUNKING_VERSION=self.chunking_version,
            CHUNK_SIZE_TOKENS=self.size,
            CHUNK_OVERLAP_TOKENS=self.overlap,
            # Expansion is a Task 5.2 question, held constant here so a cell
            # measures chunking and nothing else.
            ENABLE_PARENT_EXPANSION=False,
        )


async def prepare(cell: Cell, settings: AppSettings) -> tuple[int, int]:
    """Chunk and index the whole corpus for one cell. Returns (chunks, embedded)."""
    chunking = ChunkingService(settings=settings)
    indexer = ChunkIndexer(settings=settings)
    factory = get_session_factory()

    async with factory() as session:
        version_ids = [row[0] for row in (await session.execute(select(DocumentVersion.id))).all()]

    total_chunks = 0
    total_embedded = 0
    for version_id in version_ids:
        async with factory() as session:
            chunked = await chunking.chunk_version(session=session, version_id=version_id)
            await session.commit()
        async with factory() as session:
            indexed = await indexer.index_version(session=session, version_id=version_id)
            await session.commit()
        total_chunks += chunked.total_chunks
        total_embedded += indexed.chunks_embedded

    return total_chunks, total_embedded


def build_runner(settings: AppSettings) -> ExperimentRunner:
    """Wire a runner whose retriever reads this cell's chunking version.

    Constructed explicitly rather than through the module singletons: those cache
    the process-wide settings, so a swept cell would silently retrieve against
    whichever configuration happened to be loaded first.
    """
    generation = GenerationService(
        retriever=DenseRetriever(settings=settings),
        assembler=ContextAssembler(settings=settings),
        expander=ParentExpander(settings=settings),
        settings=settings,
    )
    return ExperimentRunner(generation_service=generation, settings=settings)


async def run_cell(cell: Cell, split: DatasetSplit, force: bool) -> None:
    """Prepare and score one cell."""
    if not force and storage.load_run_file(cell.experiment_name) is not None:
        print(f"  {cell.experiment_name}: already recorded, skipping")
        return

    settings = cell.settings()
    print(f"\n=== {cell.strategy}  size={cell.size} overlap={cell.overlap}")

    chunks, embedded = await prepare(cell, settings)
    print(f"  chunked {chunks}, embedded {embedded} (rest already indexed)")

    questions = load_split(split)
    run = await build_runner(settings).run(
        name=cell.experiment_name,
        questions=questions,
        session_factory=get_session_factory(),
        split=split,
        dataset_version=settings.EVAL_DATASET_VERSION,
        description=(
            f"Task 5.3 sweep: {cell.strategy} chunking at {cell.size}/{cell.overlap} tokens. "
            f"Retrieval metrics only."
        ),
        notes=f"{chunks} chunks under chunking_version={cell.chunking_version}.",
        retrieval_only=True,
        judge_enabled=False,
    )
    storage.save_run_file(run)
    storage.clear_checkpoint(cell.experiment_name)

    m = run.metrics
    print(
        f"  recall@5={m.get('recall@5', 0):.4f}  mrr={m.get('mrr', 0):.4f}  "
        f"ndcg@5={m.get('ndcg@5', 0):.4f}  ctx_prec={m.get('context_precision', 0):.4f}"
    )


def report() -> int:
    """Print every recorded sweep cell against the Stage 4 baseline."""
    baseline = storage.load_run_file("experiment-001-baseline")
    rows = []
    for name in storage.list_run_files():
        if not name.startswith("experiment-002-"):
            continue
        run = storage.load_run_file(name)
        if run is not None:
            rows.append(run)

    if not rows:
        print("No sweep cells recorded yet.")
        return 1

    base_recall = baseline.metrics.get("recall@5", 0.0) if baseline else 0.0
    header = (
        f"{'cell':<42}{'chunks':>8}{'recall@5':>10}{'mrr':>8}"
        f"{'ndcg@5':>9}{'ctx_prec':>10}{'vs base':>9}"
    )
    print(header)
    print("-" * len(header))
    for run in sorted(rows, key=lambda r: -r.metrics.get("recall@5", 0.0)):
        m = run.metrics
        chunks = "".join(c for c in run.notes.split(" ")[0] if c.isdigit()) or "?"
        print(
            f"{run.name.replace('experiment-002-', ''):<42}{chunks:>8}"
            f"{m.get('recall@5', 0):>10.4f}{m.get('mrr', 0):>8.4f}"
            f"{m.get('ndcg@5', 0):>9.4f}{m.get('context_precision', 0):>10.4f}"
            f"{m.get('recall@5', 0) - base_recall:>+9.4f}"
        )
    if baseline:
        print(f"\nStage 4 baseline (fixed 512/64, full run): recall@5 {base_recall:.4f}")
    return 0


async def main_async(args: argparse.Namespace) -> int:
    setup_logging()

    if args.report:
        return report()

    if args.phase == "strategies":
        cells = [Cell(s, size, overlap) for s, size, overlap in STRATEGY_PHASE]
    else:
        cells = [Cell(args.strategy, size, overlap) for size, overlap in PARAMETER_PHASE]

    split = DatasetSplit(args.split)
    for cell in cells:
        try:
            await run_cell(cell, split, force=args.force)
        except Exception as exc:  # noqa: BLE001 - one bad cell must not lose the others
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            logger.exception("sweep_cell_failed", cell=cell.experiment_name)

    print()
    return report()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_chunking_sweep", description=__doc__)
    parser.add_argument("--phase", choices=["strategies", "parameters"], default="strategies")
    parser.add_argument("--strategy", choices=STRATEGIES, default="hierarchical")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--force", action="store_true", help="re-score cells already recorded")
    parser.add_argument("--report", action="store_true", help="print the table and exit")
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

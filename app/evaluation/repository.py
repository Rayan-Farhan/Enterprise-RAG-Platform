"""Persistence for experiment runs and human verdicts (Task 4.4, ADR-029)."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.db.models.evaluation import ExperimentRun as ExperimentRunRow
from app.db.models.evaluation import HumanReviewVerdict
from app.db.models.evaluation import QuestionResult as QuestionResultRow
from app.evaluation.results import ExperimentRun, HumanVerdict, QuestionResult
from app.evaluation.schemas import DatasetSplit, Difficulty, QuestionType

logger = get_logger("app.evaluation.repository")


class ExperimentRepository:
    """Reads and writes experiment records."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(self, run: ExperimentRun) -> ExperimentRunRow:
        """Persist a completed run and all of its per-question results.

        Re-saving the same ``run_id`` replaces the previous rows rather than
        merging: a partially overwritten run would silently mix metrics from two
        configurations, which is the one thing an experiment record must never do.
        """
        existing = await self.session.get(ExperimentRunRow, run.run_id)
        if existing is not None:
            await self.session.delete(existing)
            await self.session.flush()

        row = ExperimentRunRow(
            id=run.run_id,
            name=run.name,
            description=run.description,
            started_at=run.started_at,
            completed_at=run.completed_at,
            dataset_split=run.dataset_split.value,
            dataset_version=run.dataset_version,
            dataset_size=run.dataset_size,
            git_commit=run.git_commit,
            config_snapshot=run.config_snapshot,
            embedding_version=run.embedding_version,
            chunking_version=run.chunking_version,
            generator_provider=run.generator_provider,
            generator_model=run.generator_model,
            generator_model_version=run.generator_model_version,
            judge_provider=run.judge_provider,
            judge_model=run.judge_model,
            judge_model_version=run.judge_model_version,
            prompt_versions=run.prompt_versions,
            prompt_hashes=run.prompt_hashes,
            metrics=run.metrics,
            metrics_by_type=run.metrics_by_type,
            system_metrics=run.system_metrics,
            notes=run.notes,
        )
        self.session.add(row)

        for result in run.results:
            self.session.add(
                QuestionResultRow(
                    run_id=run.run_id,
                    question_id=result.question_id,
                    question=result.question,
                    question_type=result.question_type.value,
                    difficulty=result.difficulty.value,
                    answer=result.answer,
                    abstained=result.abstained,
                    rejected=result.rejected,
                    support=result.support,
                    retrieved_chunk_ids=[str(c) for c in result.retrieved_chunk_ids],
                    retrieved_element_ids=result.retrieved_element_ids,
                    context_element_ids=result.context_element_ids,
                    citation_markers=result.citation_markers,
                    cited_element_ids=result.cited_element_ids,
                    retrieval_metrics=result.retrieval_metrics,
                    deterministic_metrics=result.deterministic_metrics,
                    judge_metrics=result.judge_metrics,
                    judge_raw_scores=result.judge_raw_scores,
                    judge_stdev=result.judge_stdev,
                    judge_reasoning=result.judge_reasoning,
                    judge_errors=result.judge_errors,
                    latency_ms=result.latency_ms,
                    retrieval_latency_ms=result.retrieval_latency_ms,
                    generation_latency_ms=result.generation_latency_ms,
                    token_counts=result.token_counts,
                    error=result.error,
                    expected_to_fail_until_stage=result.expected_to_fail_until_stage,
                )
            )

        await self.session.flush()
        logger.info("experiment_saved", run_id=str(run.run_id), name=run.name)
        return row

    async def get_by_id(self, run_id: uuid.UUID) -> ExperimentRun | None:
        """Load a run and its per-question results back into the domain model."""
        row = (
            await self.session.execute(
                select(ExperimentRunRow)
                .where(ExperimentRunRow.id == run_id)
                .options(selectinload(ExperimentRunRow.question_results))
            )
        ).scalar_one_or_none()
        return self._to_domain(row) if row is not None else None

    async def get_latest_by_name(
        self,
        name: str,
        split: DatasetSplit | None = None,
    ) -> ExperimentRun | None:
        """Load the most recent run with a given name.

        Names are not unique on purpose — re-running ``experiment-001-baseline``
        after a bug fix should keep the old record for comparison rather than
        destroy it — so "latest" is how the CLI resolves a name to a run.
        """
        statement = (
            select(ExperimentRunRow)
            .where(ExperimentRunRow.name == name)
            .options(selectinload(ExperimentRunRow.question_results))
            .order_by(ExperimentRunRow.started_at.desc())
            .limit(1)
        )
        if split is not None:
            statement = statement.where(ExperimentRunRow.dataset_split == split.value)

        row = (await self.session.execute(statement)).scalar_one_or_none()
        return self._to_domain(row) if row is not None else None

    async def list_runs(self, limit: int = 50) -> list[ExperimentRunRow]:
        """List runs newest first, without loading their per-question results."""
        return list(
            (
                await self.session.execute(
                    select(ExperimentRunRow)
                    .order_by(ExperimentRunRow.started_at.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )

    async def save_human_verdicts(self, verdicts: list[HumanVerdict]) -> int:
        """Upsert Layer 3 verdicts, keyed by (run, question, reviewer)."""
        saved = 0
        for verdict in verdicts:
            existing = (
                await self.session.execute(
                    select(HumanReviewVerdict).where(
                        HumanReviewVerdict.run_id == verdict.run_id,
                        HumanReviewVerdict.question_id == verdict.question_id,
                        HumanReviewVerdict.reviewer == verdict.reviewer,
                    )
                )
            ).scalar_one_or_none()

            if existing is None:
                existing = HumanReviewVerdict(
                    run_id=verdict.run_id,
                    question_id=verdict.question_id,
                    reviewer=verdict.reviewer,
                )
                self.session.add(existing)

            existing.reviewed_at = verdict.reviewed_at
            existing.is_correct = verdict.is_correct
            existing.is_faithful = verdict.is_faithful
            existing.citations_are_correct = verdict.citations_are_correct
            existing.abstention_was_right = verdict.abstention_was_right
            existing.severity = verdict.severity
            existing.comment = verdict.comment
            saved += 1

        await self.session.flush()
        logger.info("human_verdicts_saved", count=saved)
        return saved

    async def get_human_verdicts(self, run_id: uuid.UUID) -> list[HumanVerdict]:
        """Load every reviewer verdict recorded against a run."""
        rows = (
            (
                await self.session.execute(
                    select(HumanReviewVerdict).where(HumanReviewVerdict.run_id == run_id)
                )
            )
            .scalars()
            .all()
        )
        return [
            HumanVerdict(
                run_id=row.run_id,
                question_id=row.question_id,
                reviewer=row.reviewer,
                reviewed_at=row.reviewed_at,
                is_correct=row.is_correct,
                is_faithful=row.is_faithful,
                citations_are_correct=row.citations_are_correct,
                abstention_was_right=row.abstention_was_right,
                severity=row.severity,
                comment=row.comment,
            )
            for row in rows
        ]

    @staticmethod
    def _to_domain(row: ExperimentRunRow) -> ExperimentRun:
        return ExperimentRun(
            run_id=row.id,
            name=row.name,
            description=row.description,
            started_at=row.started_at,
            completed_at=row.completed_at,
            dataset_split=DatasetSplit(row.dataset_split),
            dataset_version=row.dataset_version,
            dataset_size=row.dataset_size,
            git_commit=row.git_commit,
            config_snapshot=row.config_snapshot,
            embedding_version=row.embedding_version,
            chunking_version=row.chunking_version,
            generator_provider=row.generator_provider,
            generator_model=row.generator_model,
            generator_model_version=row.generator_model_version,
            judge_provider=row.judge_provider,
            judge_model=row.judge_model,
            judge_model_version=row.judge_model_version,
            prompt_versions=row.prompt_versions,
            prompt_hashes=row.prompt_hashes,
            metrics=row.metrics,
            metrics_by_type=row.metrics_by_type,
            system_metrics=row.system_metrics,
            notes=row.notes,
            results=[
                QuestionResult(
                    question_id=r.question_id,
                    question=r.question,
                    question_type=QuestionType(r.question_type),
                    difficulty=Difficulty(r.difficulty),
                    answer=r.answer,
                    abstained=r.abstained,
                    rejected=r.rejected,
                    support=r.support,
                    retrieved_chunk_ids=[uuid.UUID(c) for c in r.retrieved_chunk_ids],
                    retrieved_element_ids=r.retrieved_element_ids,
                    context_element_ids=r.context_element_ids,
                    citation_markers=r.citation_markers,
                    cited_element_ids=r.cited_element_ids,
                    retrieval_metrics=r.retrieval_metrics,
                    deterministic_metrics=r.deterministic_metrics,
                    judge_metrics=r.judge_metrics,
                    judge_raw_scores=r.judge_raw_scores,
                    judge_stdev=r.judge_stdev,
                    judge_reasoning=r.judge_reasoning,
                    judge_errors=r.judge_errors,
                    latency_ms=r.latency_ms,
                    retrieval_latency_ms=r.retrieval_latency_ms,
                    generation_latency_ms=r.generation_latency_ms,
                    token_counts=r.token_counts,
                    error=r.error,
                    expected_to_fail_until_stage=r.expected_to_fail_until_stage,
                )
                for r in sorted(row.question_results, key=lambda r: r.question_id)
            ],
        )

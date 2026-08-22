"""Experiment runner (Task 4.4, ADR-029).

Executes a named pipeline configuration against a dataset split and records
everything needed to interpret the result later: the config snapshot, the
dataset version, the model and prompt versions with their hashes, every metric,
per-question results, latency percentiles, token usage, and failure rate.

Two design points that are easy to get wrong and expensive to discover late:

**A question that raises is recorded, not dropped.** It counts toward
``failure_rate`` and contributes no metrics. Dropping it would make a run that
crashed on its hardest third look like a run that scored well.

**Concurrency is bounded and low by default.** The hosted free tiers rate-limit
aggressively, and a 429 storm mid-run produces a set of numbers that measure the
rate limiter rather than the retriever.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.evaluation.judge import JudgeService
from app.evaluation.metrics import retrieval as retrieval_metrics
from app.evaluation.metrics.generation import score_deterministic
from app.evaluation.metrics.system import SystemMetrics
from app.evaluation.results import ExperimentRun, QuestionResult
from app.evaluation.schemas import DatasetSplit, GoldenQuestion
from app.generation.service import AnswerResult, GenerationService, get_generation_service

logger = get_logger("app.evaluation.runner")

#: Consecutive quota refusals after which the run stops rather than grinding
#: through the rest of the split recording failures. On a metered free tier the
#: daily budget does not come back within a run, so continuing turns a partial
#: measurement into a bad one — a 60% failure rate that then gets committed and
#: cited. Two in a row is enough: a single 429 is already retried six times with
#: provider-stated backoff before it ever reaches here.
QUOTA_ABORT_THRESHOLD = 2


class QuotaExhausted(RuntimeError):
    """Raised when the provider's budget is gone and the run should be resumed later.

    Carries the partial results so the caller can checkpoint and report progress
    rather than losing the questions already paid for.
    """

    def __init__(self, completed: int, total: int, detail: str) -> None:
        super().__init__(f"Provider quota exhausted after {completed}/{total} questions: {detail}")
        self.completed = completed
        self.total = total
        self.detail = detail


def current_git_commit() -> str | None:
    """Return the HEAD commit, or None outside a repository.

    Recorded on every run because "which code produced this number" is the first
    question anyone asks of a six-month-old experiment.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def config_snapshot(settings: AppSettings) -> dict[str, object]:
    """Capture every setting that can move a metric.

    Deliberately explicit rather than dumping all of ``AppSettings``: a full dump
    would include credentials, and would also churn the recorded snapshot on
    every unrelated setting added in a later stage, making diffs unreadable.
    """
    return {
        "inference_profile": settings.INFERENCE_PROFILE,
        "chunking_strategy": settings.CHUNKING_STRATEGY,
        "chunking_version": settings.CHUNKING_VERSION,
        "chunk_size_tokens": settings.CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": settings.CHUNK_OVERLAP_TOKENS,
        "embedding_version": settings.effective_embedding_version,
        "embedding_dimensions": settings.EMBEDDING_DIMENSIONS,
        "retrieval_top_k": settings.RETRIEVAL_TOP_K,
        "retrieval_min_score": settings.RETRIEVAL_MIN_SCORE,
        "retrieval_candidate_limit": settings.RETRIEVAL_CANDIDATE_LIMIT,
        "reranking_enabled": settings.ENABLE_RERANKING,
        "generation_max_context_tokens": settings.GENERATION_MAX_CONTEXT_TOKENS,
        "generation_temperature": settings.GENERATION_TEMPERATURE,
        "generation_max_tokens": settings.GENERATION_MAX_TOKENS,
        "abstention_min_evidence_chunks": settings.ABSTENTION_MIN_EVIDENCE_CHUNKS,
        "judge_enabled": settings.EVAL_JUDGE_ENABLED,
        "judge_provider": settings.EVAL_JUDGE_PROVIDER,
        "judge_model": settings.EVAL_JUDGE_MODEL or None,
        "judge_samples": settings.EVAL_JUDGE_SAMPLES,
    }


class ExperimentRunner:
    """Runs one named configuration against one dataset split."""

    def __init__(
        self,
        generation_service: GenerationService | None = None,
        judge: JudgeService | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.generation = generation_service or get_generation_service()
        self.judge = judge or JudgeService(settings=self.settings)

    async def run(
        self,
        name: str,
        questions: Sequence[GoldenQuestion],
        session_factory: object,
        split: DatasetSplit,
        dataset_version: str,
        description: str = "",
        notes: str = "",
        judge_enabled: bool | None = None,
        retrieval_only: bool = False,
        completed: Mapping[str, QuestionResult] | None = None,
        on_result: Callable[[QuestionResult], None] | None = None,
    ) -> ExperimentRun:
        """Evaluate every question and return the completed run record.

        ``session_factory`` is an async context manager factory yielding an
        ``AsyncSession``. Each question gets its own session: sharing one across
        concurrent questions would serialise them on the connection and make the
        recorded latencies a measurement of the pool rather than the pipeline.

        ``completed`` carries results recovered from a checkpoint; those
        questions are not re-evaluated. ``on_result`` is called as each new
        question finishes, so a caller can persist progress before the next one
        is attempted. Together they let one experiment span several days of a
        metered provider's budget.

        Raises :class:`QuotaExhausted` when the provider's budget is gone. The
        alternative — grinding through the remaining questions and recording each
        as a failure — produces a run that looks like a catastrophic quality
        regression and is really an accounting limit.
        """
        run = ExperimentRun(
            run_id=uuid.uuid4(),
            name=name,
            description=description,
            notes=notes,
            dataset_split=split,
            dataset_version=dataset_version,
            dataset_size=len(questions),
            retrieval_only=retrieval_only,
            git_commit=current_git_commit(),
            config_snapshot=config_snapshot(self.settings),
            embedding_version=self.settings.effective_embedding_version,
            chunking_version=self.settings.CHUNKING_VERSION,
        )

        use_judge = self.settings.EVAL_JUDGE_ENABLED if judge_enabled is None else judge_enabled
        if retrieval_only:
            # Layer 1 needs an answer to score and Layer 2 needs a model to score
            # it with. Neither exists here, so asking for the judge would be a
            # silent no-op rather than an error.
            use_judge = False
        system = SystemMetrics(total_questions=len(questions))

        recovered = dict(completed or {})
        wanted = {question.question_id for question in questions}
        results: list[QuestionResult] = [
            result for question_id, result in recovered.items() if question_id in wanted
        ]
        for result in results:
            self._replay(result, system)

        pending = [q for q in questions if q.question_id not in recovered]

        logger.info(
            "experiment_started",
            name=name,
            split=split.value,
            questions=len(questions),
            resumed=len(results),
            pending=len(pending),
            concurrency=self.settings.EVAL_CONCURRENCY,
            judge=use_judge,
        )

        # A plain `gather` over every question cannot stop early, and stopping
        # early is the whole point once a provider's daily budget is finite. A
        # small worker pool gives the same bounded concurrency and can abandon
        # the remaining queue the moment the budget is gone.
        queue: asyncio.Queue[GoldenQuestion] = asyncio.Queue()
        for question in pending:
            queue.put_nowait(question)

        quota_failures = 0
        exhausted: QuotaExhausted | None = None
        lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal quota_failures, exhausted
            while True:
                if exhausted is not None:
                    return
                try:
                    question = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                result = await self._evaluate_question(
                    question=question,
                    session_factory=session_factory,
                    run=run,
                    system=system,
                    use_judge=use_judge,
                    retrieval_only=retrieval_only,
                )

                async with lock:
                    if result.failed_on_quota:
                        quota_failures += 1
                        if quota_failures >= QUOTA_ABORT_THRESHOLD and exhausted is None:
                            exhausted = QuotaExhausted(
                                completed=len(results),
                                total=len(questions),
                                detail=result.error or "provider refused with a rate limit",
                            )
                        continue

                    quota_failures = 0
                    results.append(result)
                    if on_result is not None:
                        on_result(result)

        await asyncio.gather(*(worker() for _ in range(max(1, self.settings.EVAL_CONCURRENCY))))

        if exhausted is not None:
            logger.warning(
                "experiment_quota_exhausted",
                name=name,
                completed=len(results),
                total=len(questions),
            )
            raise exhausted

        run.results = sorted(results, key=lambda r: r.question_id)

        run.metrics = self._aggregate(run.results)
        run.metrics_by_type = self._aggregate_by_type(run.results)
        run.system_metrics = system.as_metrics()
        run.completed_at = datetime.now(UTC)

        logger.info(
            "experiment_completed",
            name=name,
            duration_s=round(run.duration_seconds, 1),
            failures=system.failures,
            evaluation_days=run.evaluation_days,
        )
        return run

    @staticmethod
    def _replay(result: QuestionResult, system: SystemMetrics) -> None:
        """Fold a checkpointed result back into the system metrics.

        Resumed questions were measured on an earlier day; their latencies and
        token counts belong in the run's aggregates exactly as if they had just
        been evaluated, otherwise the percentiles describe only the final day.
        """
        if result.failed:
            system.record_failure()
            return

        system.record(
            total_ms=result.latency_ms,
            retrieval_ms=result.retrieval_latency_ms,
            generation_ms=result.generation_latency_ms,
            token_counts=result.token_counts,
            evidence_tokens=result.evidence_tokens,
        )

    # -- one question ------------------------------------------------------

    async def _evaluate_question(
        self,
        question: GoldenQuestion,
        session_factory: object,
        run: ExperimentRun,
        system: SystemMetrics,
        use_judge: bool,
        retrieval_only: bool = False,
    ) -> QuestionResult:
        result = QuestionResult(
            question_id=question.question_id,
            question=question.question,
            question_type=question.question_type,
            difficulty=question.difficulty,
            expected_to_fail_until_stage=question.expected_to_fail_until_stage,
        )

        started = time.perf_counter()
        try:
            async with session_factory() as session:  # type: ignore[operator]
                if retrieval_only:
                    return await self._evaluate_retrieval_only(
                        question=question,
                        session=session,
                        result=result,
                        system=system,
                        started=started,
                    )
                answer = await self.generation.answer(query=question.question, session=session)
        except Exception as exc:  # noqa: BLE001 - a failed question is data, not a crash
            result.error = f"{type(exc).__name__}: {exc}"
            result.latency_ms = (time.perf_counter() - started) * 1000
            system.record_failure()
            logger.warning("question_failed", question_id=question.question_id, error=result.error)
            return result

        self._record_answer(result, answer)
        system.record(
            total_ms=answer.total_latency_ms,
            retrieval_ms=answer.retrieval_latency_ms,
            generation_ms=answer.generation_latency_ms,
            token_counts=answer.token_counts,
            evidence_tokens=answer.evidence_tokens,
        )

        # Layer 0 — retrieval, deterministic.
        result.retrieval_metrics = retrieval_metrics.compute_case_metrics(
            self._retrieval_case(question, answer)
        )

        # Layer 1 — deterministic generation checks.
        result.deterministic_metrics = score_deterministic(question, answer).as_metrics()

        # Layer 2 — LLM judge.
        if use_judge:
            verdict = await self.judge.judge(question, answer, answer.evidence_block)
            result.judge_metrics = verdict.as_metrics()
            result.judge_raw_scores = verdict.raw_scores
            result.judge_stdev = verdict.score_stdev
            result.judge_reasoning = verdict.reasoning
            result.judge_errors = verdict.errors
            if verdict.judge_model and not run.judge_model:
                run.judge_provider = verdict.judge_provider
                run.judge_model = verdict.judge_model
                run.judge_model_version = verdict.judge_model_version
                run.prompt_versions.update(verdict.prompt_versions)
                run.prompt_hashes.update(verdict.prompt_hashes)

        if not run.generator_model and answer.model_name:
            run.generator_provider = answer.provider
            run.generator_model = answer.model_name
            run.generator_model_version = answer.model_version
            run.prompt_versions.update(answer.prompt_versions)
            run.prompt_hashes.update(answer.prompt_hashes)

        return result

    async def _evaluate_retrieval_only(
        self,
        question: GoldenQuestion,
        session: AsyncSession,
        result: QuestionResult,
        system: SystemMetrics,
        started: float,
    ) -> QuestionResult:
        """Score Layer 0 without generating an answer.

        Task 5.3 sweeps four chunking strategies across a parameter grid, and its
        exit gate is a *retrieval* comparison. Generating an answer for every
        question of every configuration would cost a day of free-tier quota per
        cell to produce numbers the comparison does not read.

        Context assembly still runs. It makes no model call, and skipping it
        would leave ``context_precision`` and ``context_recall`` measuring the
        raw candidate list rather than what would actually reach a prompt — a
        different quantity, silently named the same thing.
        """
        retrieval = await self.generation.retriever.retrieve(
            query=question.question,
            session=session,
        )
        expansion = await self.generation.expander.expand(retrieval.chunks, session=session)
        context = self.generation.assembler.assemble(
            query=question.question, chunks=expansion.chunks
        )

        context_ids = {c.chunk_id for c in context.included_chunks}
        result.retrieved_chunk_ids = [c.chunk_id for c in expansion.chunks]
        result.retrieved_element_ids = sorted(
            {eid for chunk in expansion.chunks for eid in chunk.element_ids}
        )
        result.context_element_ids = sorted(
            {
                eid
                for chunk in expansion.chunks
                if chunk.chunk_id in context_ids
                for eid in chunk.element_ids
            }
        )
        result.latency_ms = (time.perf_counter() - started) * 1000
        result.retrieval_latency_ms = retrieval.latency_ms
        result.evidence_tokens = context.evidence_tokens

        result.retrieval_metrics = retrieval_metrics.compute_case_metrics(
            retrieval_metrics.RetrievalCase(
                question_id=question.question_id,
                retrieved_element_sets=[frozenset(chunk.element_ids) for chunk in expansion.chunks],
                relevant_element_ids=frozenset(question.expected_element_ids()),
                context_element_ids=frozenset(result.context_element_ids),
            )
        )

        system.record(
            total_ms=result.latency_ms,
            retrieval_ms=retrieval.latency_ms,
            generation_ms=0.0,
            token_counts={},
            evidence_tokens=context.evidence_tokens,
        )
        return result

    @staticmethod
    def _record_answer(result: QuestionResult, answer: AnswerResult) -> None:
        """Copy the pipeline's output onto the result row."""
        result.answer = answer.answer
        result.abstained = answer.abstained
        result.rejected = answer.rejected
        result.support = str(answer.support)
        result.retrieved_chunk_ids = list(answer.retrieved_chunk_ids)
        result.retrieved_element_ids = sorted(
            {eid for chunk in answer.retrieved_chunks for eid in chunk.element_ids}
        )
        # Drawn from context_chunks, not retrieved_chunks: after parent expansion
        # the chunks that reached the prompt are not the ones retrieval ranked.
        context_ids = set(answer.context_chunk_ids)
        context_source = answer.context_chunks or answer.retrieved_chunks
        result.context_element_ids = sorted(
            {
                eid
                for chunk in context_source
                if chunk.chunk_id in context_ids
                for eid in chunk.element_ids
            }
        )
        result.citation_markers = [c.marker for c in answer.citations]
        result.cited_element_ids = sorted({eid for c in answer.citations for eid in c.element_ids})
        result.latency_ms = answer.total_latency_ms
        result.retrieval_latency_ms = answer.retrieval_latency_ms
        result.generation_latency_ms = answer.generation_latency_ms
        result.token_counts = dict(answer.token_counts)
        result.evidence_tokens = answer.evidence_tokens

    @staticmethod
    def _retrieval_case(
        question: GoldenQuestion,
        answer: AnswerResult,
    ) -> retrieval_metrics.RetrievalCase:
        context_ids = set(answer.context_chunk_ids)
        context_source = answer.context_chunks or answer.retrieved_chunks
        return retrieval_metrics.RetrievalCase(
            question_id=question.question_id,
            # Layer 0 scores what the retriever ranked, so this stays on the
            # pre-expansion hits even when generation read the parents.
            retrieved_element_sets=[
                frozenset(chunk.element_ids) for chunk in answer.retrieved_chunks
            ],
            relevant_element_ids=frozenset(question.expected_element_ids()),
            context_element_ids=frozenset(
                eid
                for chunk in context_source
                if chunk.chunk_id in context_ids
                for eid in chunk.element_ids
            ),
        )

    # -- aggregation -------------------------------------------------------

    @staticmethod
    def _aggregate(results: Sequence[QuestionResult]) -> dict[str, float]:
        """Macro-average every metric over the questions that reported it."""
        return retrieval_metrics.aggregate([r.all_metrics() for r in results if not r.failed])

    @staticmethod
    def _aggregate_by_type(
        results: Sequence[QuestionResult],
    ) -> dict[str, dict[str, float]]:
        """Per-question-type breakdown — the slice `make eval-diff` reports on.

        Aggregate movement is almost never the useful signal. A reranker that
        lifts exact-retrieval by 20 points while costing multi-hop 10 shows up as
        a modest overall gain and as two clear, actionable findings here.
        """
        grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
        for result in results:
            if not result.failed:
                grouped[result.question_type.value].append(result.all_metrics())

        return {
            question_type: retrieval_metrics.aggregate(metrics)
            for question_type, metrics in sorted(grouped.items())
        }


async def persist_run(run: ExperimentRun, session: AsyncSession) -> None:
    """Save a completed run through the repository and commit."""
    from app.evaluation.repository import ExperimentRepository

    await ExperimentRepository(session).save(run)
    await session.commit()

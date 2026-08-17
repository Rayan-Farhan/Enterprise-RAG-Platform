"""Layer 2 — LLM-as-judge generation metrics (Task 4.3, master §52, ADR-029).

Three things make this a measurement rather than a vibe:

**The judge is a different model than the generator.** Under the hosted profile
the answers come from Gemini and the judge runs on Groq, pinned through the
gateway's explicit ``provider`` routing. A model scoring its own output has a
well-documented self-preference bias, and the golden dataset here is partly
LLM-drafted, so a shared model would be marking its own homework twice over.

**Judge prompts are versioned artifacts with content hashes**, loaded through the
same registry as generation prompts. Every score records the judge model, its
version, the prompt version, and the prompt hash, so two runs are only ever
compared when they were judged the same way.

**Failure is recorded, not defaulted.** A judge call that fails or returns
unparseable output yields no score at all rather than a neutral 3. Substituting a
midpoint would let a run in which the judge was down look like a run in which the
system was mediocre.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.evaluation.schemas import GoldenQuestion
from app.generation.prompts.registry import PromptTemplate, load_prompt
from app.generation.service import AnswerResult
from app.models.gateway import ModelGateway, get_model_gateway
from app.models.schemas import ModelMetadata

logger = get_logger("app.evaluation.judge")

JUDGE_TEMPLATE_DIR = Path(__file__).parent / "prompts" / "templates"

#: Scores are authored on a 1-5 Likert scale because models are markedly more
#: consistent on a small ordinal scale than on 0-100, but every metric is
#: reported in 0-1 so it sits on the same axis as recall and precision.
_SCALE_MIN = 1
_SCALE_MAX = 5

_ANSWER_DIMENSIONS = (
    "faithfulness",
    "groundedness",
    "answer_correctness",
    "relevance",
    "completeness",
)
_CITATION_DIMENSIONS = ("citation_correctness", "citation_completeness")
_ABSTENTION_DIMENSIONS = ("abstention_accuracy",)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class JudgeParseError(ValueError):
    """Raised when a judge response contains no usable JSON object."""


def normalize_score(raw: float) -> float:
    """Map a 1-5 Likert score onto 0-1."""
    clamped = max(_SCALE_MIN, min(_SCALE_MAX, raw))
    return (clamped - _SCALE_MIN) / (_SCALE_MAX - _SCALE_MIN)


def parse_judge_json(text: str) -> dict[str, Any]:
    """Extract the JSON object from a judge response.

    Models wrap JSON in markdown fences and prose despite instructions not to, and
    reasoning models prepend commentary. Rather than fail the whole evaluation on
    formatting, take the outermost brace-delimited span — but never invent a
    default, so a genuinely unparseable response still surfaces as a failure.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(stripped)
        if not match:
            raise JudgeParseError(f"No JSON object in judge response: {text[:200]!r}") from None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise JudgeParseError(f"Malformed JSON in judge response: {exc}") from exc

    if not isinstance(parsed, dict):
        raise JudgeParseError(f"Judge response is {type(parsed).__name__}, expected an object")
    return parsed


@dataclass(frozen=True)
class _PromptOutcome:
    """What one judge prompt produced across its samples."""

    scores: dict[str, list[float]]
    reasonings: list[str]
    extras: dict[str, Any]
    errors: list[str]
    metadata: ModelMetadata | None


@dataclass
class JudgeVerdict:
    """One judged question: normalised scores plus the provenance of the judging."""

    question_id: str
    scores: dict[str, float] = field(default_factory=dict)
    raw_scores: dict[str, float] = field(default_factory=dict)
    score_stdev: dict[str, float] = field(default_factory=dict)
    reasoning: dict[str, str] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    judge_provider: str | None = None
    judge_model: str | None = None
    judge_model_version: str | None = None
    prompt_versions: dict[str, str] = field(default_factory=dict)
    prompt_hashes: dict[str, str] = field(default_factory=dict)
    samples: int = 1

    errors: list[str] = field(default_factory=list)

    def as_metrics(self) -> dict[str, float]:
        """Flatten to the ``judge_*`` metric mapping the runner aggregates."""
        return {f"judge_{name}": value for name, value in self.scores.items()}


class JudgeService:
    """Runs the Layer 2 judge prompts against one answered question."""

    def __init__(
        self,
        gateway: ModelGateway | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.gateway = gateway or get_model_gateway()

    # -- prompt access -----------------------------------------------------

    def _template(self, version: str) -> PromptTemplate:
        return load_prompt(JUDGE_TEMPLATE_DIR, version)

    # -- single judged dimension group -------------------------------------

    async def _run_prompt(
        self,
        prompt_version: str,
        values: dict[str, str],
        dimensions: tuple[str, ...],
    ) -> _PromptOutcome:
        """Sample one judge prompt N times, returning per-dimension raw scores.

        Sampling more than once is how the variance band in the Stage 4 exit gate
        is measured rather than assumed. With temperature at 0 the samples are
        usually identical, and where they are not, the spread is recorded.
        """
        template = self._template(prompt_version)
        rendered = template.render(**values)

        collected: dict[str, list[float]] = {d: [] for d in dimensions}
        reasonings: list[str] = []
        extras: dict[str, Any] = {}
        errors: list[str] = []
        metadata: ModelMetadata | None = None

        for sample in range(max(1, self.settings.EVAL_JUDGE_SAMPLES)):
            try:
                response = await self.gateway.generate(
                    prompt=rendered,
                    temperature=self.settings.EVAL_JUDGE_TEMPERATURE,
                    max_tokens=self.settings.EVAL_JUDGE_MAX_TOKENS,
                    prompt_version=prompt_version,
                    provider=self.settings.EVAL_JUDGE_PROVIDER or None,
                    model_name=self.settings.EVAL_JUDGE_MODEL or None,
                )
                parsed = parse_judge_json(response.text)
            except Exception as exc:  # noqa: BLE001 - any judge failure is a recorded gap
                errors.append(f"{prompt_version}[sample={sample}]: {exc}")
                logger.warning("judge_call_failed", prompt_version=prompt_version, error=str(exc))
                continue

            metadata = response.metadata
            for dimension in dimensions:
                value = parsed.get(dimension)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    collected[dimension].append(float(value))
                else:
                    errors.append(
                        f"{prompt_version}[sample={sample}]: '{dimension}' missing or "
                        f"non-numeric ({value!r})"
                    )

            if isinstance(parsed.get("reasoning"), str):
                reasonings.append(parsed["reasoning"])
            for key in ("uncited_claims", "evidence_was_sufficient", "behaviour"):
                if key in parsed:
                    extras[key] = parsed[key]

        return _PromptOutcome(
            scores=collected,
            reasonings=reasonings,
            extras=extras,
            errors=errors,
            metadata=metadata,
        )

    # -- public API --------------------------------------------------------

    async def judge(
        self,
        question: GoldenQuestion,
        result: AnswerResult,
        evidence_text: str,
    ) -> JudgeVerdict:
        """Score one answered question across all Layer 2 dimensions."""
        verdict = JudgeVerdict(question_id=question.question_id)
        verdict.samples = max(1, self.settings.EVAL_JUDGE_SAMPLES)

        if not self.settings.EVAL_JUDGE_ENABLED:
            verdict.errors.append("judge_disabled")
            return verdict

        if self.settings.INFERENCE_PROFILE == "stub":
            # The stub gateway returns canned text. Scoring it would produce
            # numbers that look like evaluation data and are not.
            verdict.errors.append("judge_skipped_stub_profile")
            logger.warning("judge_skipped_stub_profile", question_id=question.question_id)
            return verdict

        citations_text = (
            "\n\n".join(
                f"[{c.marker.strip('[]')}] document='{c.document_title or c.document_id}' "
                f"page={c.page_number} section='{c.section_label}'\n{c.quote or ''}"
                for c in result.citations
            )
            or "(the answer carried no citations)"
        )

        answer_prompt = self.settings.PROMPT_VERSION_JUDGE_ANSWER
        citation_prompt = self.settings.PROMPT_VERSION_JUDGE_CITATION
        abstention_prompt = self.settings.PROMPT_VERSION_JUDGE_ABSTENTION

        # The three prompts are independent, but on the hosted profile the
        # binding constraint is tokens per minute, not round trips. Firing them
        # concurrently spends ~7k tokens in one instant against an 8k window, so
        # one or two get rate-limited, wait, and then *re-spend* their tokens on
        # retry — the run thrashes and slows to a third of its throughput.
        # Issuing them one at a time spends the same tokens per question without
        # the wasted retries. `EVAL_JUDGE_PARALLEL_PROMPTS` restores the
        # concurrent form for a local profile with no such window.
        answer_task = self._run_prompt(
            answer_prompt,
            {
                "question": question.question,
                "reference_answer": question.acceptable_answer,
                "evidence": evidence_text or "(no evidence was retrieved)",
                "system_answer": result.answer,
            },
            _ANSWER_DIMENSIONS,
        )
        citation_task = self._run_prompt(
            citation_prompt,
            {
                "question": question.question,
                "system_answer": result.answer,
                "citations": citations_text,
            },
            _CITATION_DIMENSIONS,
        )
        abstention_task = self._run_prompt(
            abstention_prompt,
            {
                "question": question.question,
                "expected_answerable": "no" if question.is_abstention_case else "yes",
                "evidence": evidence_text or "(no evidence was retrieved)",
                "system_answer": result.answer,
                "system_abstained": "yes" if result.abstained else "no",
            },
            _ABSTENTION_DIMENSIONS,
        )

        tasks = (answer_task, citation_task, abstention_task)
        if self.settings.EVAL_JUDGE_PARALLEL_PROMPTS:
            outcomes = list(await asyncio.gather(*tasks))
        else:
            outcomes = [await task for task in tasks]

        for prompt_version, outcome in zip(
            (answer_prompt, citation_prompt, abstention_prompt), outcomes, strict=True
        ):
            template = self._template(prompt_version)
            verdict.prompt_versions[prompt_version] = template.version
            verdict.prompt_hashes[prompt_version] = template.content_hash
            verdict.errors.extend(outcome.errors)
            verdict.extras.update(outcome.extras)
            if outcome.reasonings:
                verdict.reasoning[prompt_version] = outcome.reasonings[0]

            for dimension, samples in outcome.scores.items():
                if not samples:
                    continue
                mean = statistics.fmean(samples)
                verdict.raw_scores[dimension] = mean
                verdict.scores[dimension] = normalize_score(mean)
                verdict.score_stdev[dimension] = (
                    statistics.stdev(samples) if len(samples) > 1 else 0.0
                )

            if outcome.metadata is not None and verdict.judge_model is None:
                verdict.judge_provider = outcome.metadata.provider
                verdict.judge_model = outcome.metadata.model_name
                verdict.judge_model_version = outcome.metadata.model_version

        logger.info(
            "question_judged",
            question_id=question.question_id,
            dimensions=len(verdict.scores),
            errors=len(verdict.errors),
        )
        return verdict


def get_judge_service() -> JudgeService:
    """Construct a JudgeService bound to the active gateway."""
    return JudgeService()

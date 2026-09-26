"""Metadata-driven candidate narrowing (Task 6.3, master §13).

    User query -> metadata constraints -> candidate pool -> semantic/lexical search

A question that names what it is about — "the dental plan", "tenure", "the
Staff Handbook" — implies metadata the corpus already carries. Inferring that
and passing it to the channel as a filter means the engine ranks only the
matching chunks: the filter runs inside Qdrant's or OpenSearch's query, never
as a post-filter over results that were ranked against everything.

Inference is rule-based and conservative (``narrowing_vocabulary.json``): a
wrong constraint silently removes the right evidence, a missing one only costs
precision. Two conflicting values for one field cancel narrowing on it, and a
narrowed search that finds nothing falls back to the unnarrowed one. Every
decision is recorded as a trace on the result, including the candidate pool
before and after, so "did narrowing happen, and how much" is observable per
question rather than inferred.

LLM-based query understanding, including effective-period constraints, is
Stage 10 (ADR-014). The corpus carries no effective dates yet, so a temporal
rule here would have nothing to match.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.retrieval.channels import Retriever
from app.retrieval.schemas import RetrievalFilters, RetrievalResult

logger = get_logger("app.retrieval.narrowing")

VOCABULARY_PATH = Path(__file__).parent / "narrowing_vocabulary.json"

#: Fields a rule may narrow. Each exists on RetrievalFilters and is pushed into
#: both engines by their filter builders.
NARROWABLE_FIELDS = frozenset({"department", "policy_type", "employee_type"})


@dataclass(frozen=True)
class ConstraintRule:
    field: str
    value: str
    patterns: tuple[re.Pattern[str], ...]


@dataclass
class InferredConstraints:
    """What a question implies, and why."""

    constraints: dict[str, str] = field(default_factory=dict)
    matches: list[dict[str, str]] = field(default_factory=list)
    conflicts: dict[str, list[str]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.constraints)


class ConstraintExtractor:
    """Maps question text to metadata constraints through a vocabulary of rules."""

    def __init__(self, rules: list[ConstraintRule], version: str) -> None:
        self.rules = rules
        self.version = version

    @classmethod
    def from_file(cls, path: Path = VOCABULARY_PATH) -> ConstraintExtractor:
        raw = json.loads(path.read_text(encoding="utf-8"))
        rules = []
        for rule in raw["rules"]:
            if rule["field"] not in NARROWABLE_FIELDS:
                raise ValueError(f"Narrowing rule targets unsupported field '{rule['field']}'")
            rules.append(
                ConstraintRule(
                    field=rule["field"],
                    value=rule["value"],
                    patterns=tuple(re.compile(p, re.IGNORECASE) for p in rule["patterns"]),
                )
            )
        return cls(rules, version=str(raw["version"]))

    def extract(self, query: str) -> InferredConstraints:
        found: dict[str, set[str]] = {}
        matches: list[dict[str, str]] = []
        for rule in self.rules:
            for pattern in rule.patterns:
                hit = pattern.search(query)
                if hit:
                    found.setdefault(rule.field, set()).add(rule.value)
                    matches.append({"field": rule.field, "value": rule.value, "term": hit.group(0)})
                    break

        inferred = InferredConstraints(matches=matches)
        for field_name, values in found.items():
            if len(values) == 1:
                inferred.constraints[field_name] = next(iter(values))
            else:
                # "dental and health", "the Staff Handbook and the Policy Manual":
                # the question spans both, so narrowing to either would be wrong.
                inferred.conflicts[field_name] = sorted(values)
        return inferred


@lru_cache(maxsize=1)
def get_constraint_extractor() -> ConstraintExtractor:
    return ConstraintExtractor.from_file()


def merge_filters(explicit: RetrievalFilters | None, inferred: dict[str, str]) -> RetrievalFilters:
    """Add inferred constraints; a field the caller set explicitly always wins."""
    base = explicit or RetrievalFilters()
    update = {name: value for name, value in inferred.items() if getattr(base, name) is None}
    return base.model_copy(update=update)


class NarrowingRetriever:
    """Wraps any channel: infer constraints, narrow the pool, search it, trace it."""

    def __init__(
        self,
        inner: Retriever,
        extractor: ConstraintExtractor | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.inner = inner
        self.extractor = extractor or get_constraint_extractor()
        self.settings = settings or inner.settings or get_settings()

    @property
    def channel(self) -> str:
        return self.inner.channel

    async def candidate_count(self, filters: RetrievalFilters | None) -> int:
        return await self.inner.candidate_count(filters)

    async def retrieve(
        self,
        query: str,
        session: AsyncSession | None = None,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        inferred = self.extractor.extract(query)
        trace: dict[str, Any] = {
            "vocabulary": self.extractor.version,
            "matches": inferred.matches,
            "conflicts": inferred.conflicts,
            "applied": {},
            "candidates_before": None,
            "candidates_after": None,
            "fallback": False,
        }

        if not inferred:
            result = await self.inner.retrieve(query, session, top_k, filters, min_score)
            return self._with_trace(result, trace)

        narrowed = merge_filters(filters, inferred.constraints)
        trace["applied"] = {
            name: getattr(narrowed, name)
            for name in inferred.constraints
            if getattr(narrowed, name) == inferred.constraints[name]
        }
        trace["candidates_before"] = await self.inner.candidate_count(filters)
        trace["candidates_after"] = await self.inner.candidate_count(narrowed)

        result = await self.inner.retrieve(query, session, top_k, narrowed, min_score)
        if not result.chunks:
            # The scope was inferred, not stated: finding nothing inside it is
            # evidence the inference was wrong, not that the corpus is silent.
            trace["fallback"] = True
            result = await self.inner.retrieve(query, session, top_k, filters, min_score)

        logger.info(
            "retrieval_narrowed",
            channel=self.channel,
            applied=trace["applied"],
            candidates_before=trace["candidates_before"],
            candidates_after=trace["candidates_after"],
            fallback=trace["fallback"],
        )
        return self._with_trace(result, trace)

    @staticmethod
    def _with_trace(result: RetrievalResult, trace: dict[str, Any]) -> RetrievalResult:
        result.retrieval_config = {**result.retrieval_config, "narrowing": trace}
        return result

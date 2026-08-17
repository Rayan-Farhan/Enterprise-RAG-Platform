"""Post-generation citation validation and support classification (Task 3.5, ADR-025/026).

The rule from the Stage 3 exit gate is that fabricated citations are impossible.
That is enforced negatively: a marker the model emits is only allowed to survive
if it was in the context we supplied. Anything else is stripped and the answer is
downgraded or rejected — the model is never trusted to have cited honestly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from app.core.logging import get_logger
from app.generation.context import AssembledContext
from app.retrieval.schemas import Citation

logger = get_logger("app.generation.citation")

# Inline markers: [1], [2], [1, 3], [1][2]
_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SUPPORT_LINE_RE = re.compile(
    r"^\s*SUPPORT\s*:\s*(grounded|partial|insufficient)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# Sentences that assert something factual and therefore require a citation.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class SupportState(StrEnum):
    """Confidence/support state reported with every answer (master §21)."""

    GROUNDED = "grounded"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


@dataclass
class CitationValidationResult:
    """Outcome of validating a generated answer's citations."""

    answer: str
    support: SupportState
    declared_support: SupportState | None = None
    citations: list[Citation] = field(default_factory=list)
    fabricated_markers: list[str] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    rejected: bool = False
    rejection_reason: str | None = None

    @property
    def is_valid(self) -> bool:
        return not self.rejected and not self.fabricated_markers


class CitationValidator:
    """Validates that every citation in an answer resolves to supplied evidence."""

    def __init__(self, strip_fabricated: bool = True) -> None:
        # Stripping is the default because a partially-correct answer with the bad
        # marker removed is more useful than a hard failure. The fabrication is
        # still reported, and an answer left with no valid citations is rejected.
        self.strip_fabricated = strip_fabricated

    def validate(
        self,
        raw_answer: str,
        context: AssembledContext,
    ) -> CitationValidationResult:
        """Validate and clean a generated answer against its own context."""
        answer, declared_support = self._extract_support(raw_answer)
        allowed = context.allowed_markers

        used = self._extract_markers(answer)
        fabricated = sorted(used - allowed, key=self._marker_sort_key)
        valid_used = used & allowed

        if fabricated:
            logger.warning(
                "fabricated_citations_detected",
                fabricated=fabricated,
                allowed=sorted(allowed, key=self._marker_sort_key),
            )
            if self.strip_fabricated:
                answer = self._strip_markers(answer, fabricated)

        citations = [c for c in context.citations if c.marker in valid_used]
        uncited = self._uncited_factual_sentences(answer, allowed)

        support = self._resolve_support(
            declared=declared_support,
            has_evidence=context.has_evidence,
            valid_citations=len(citations),
            uncited_count=len(uncited),
            fabricated_count=len(fabricated),
        )

        result = CitationValidationResult(
            answer=answer.strip(),
            support=support,
            declared_support=declared_support,
            citations=citations,
            fabricated_markers=fabricated,
            uncited_sentences=uncited,
        )

        # An answer that asserts policy content while resolving to no real evidence
        # is exactly the confabulation this stage exists to prevent. A model that
        # explicitly declared `insufficient` is abstaining honestly, which is the
        # desired behaviour rather than a failure — so only an *asserted* answer
        # with zero resolvable citations is rejected.
        asserted = declared_support is not SupportState.INSUFFICIENT
        if context.has_evidence and not citations and asserted:
            result.rejected = True
            result.rejection_reason = (
                "Answer contains no citation resolving to supplied evidence"
            )
            logger.warning(
                "answer_rejected_no_resolvable_citations",
                fabricated=fabricated,
                declared=str(declared_support) if declared_support else None,
            )

        return result

    # ---- internals -----------------------------------------------------

    @staticmethod
    def _marker_sort_key(marker: str) -> tuple[int, str]:
        """Sort markers numerically so [12] follows [3] rather than preceding it."""
        return (int(marker), marker) if marker.isdigit() else (10**9, marker)

    @staticmethod
    def _extract_markers(text: str) -> set[str]:
        """Collect every marker referenced in the answer, expanding `[1, 3]`."""
        markers: set[str] = set()
        for match in _MARKER_RE.finditer(text):
            markers.update(part.strip() for part in match.group(1).split(","))
        return markers

    @staticmethod
    def _strip_markers(text: str, fabricated: list[str]) -> str:
        """Remove fabricated markers while keeping any valid ones in the group."""
        bad = set(fabricated)

        def replace(match: re.Match[str]) -> str:
            kept = [p.strip() for p in match.group(1).split(",") if p.strip() not in bad]
            return f"[{', '.join(kept)}]" if kept else ""

        # Collapse whitespace left behind by a fully removed marker group.
        return re.sub(r"[ \t]{2,}", " ", _MARKER_RE.sub(replace, text))

    @staticmethod
    def _extract_support(raw: str) -> tuple[str, SupportState | None]:
        """Split the SUPPORT control line off the answer body."""
        match = _SUPPORT_LINE_RE.search(raw)
        if match is None:
            return raw.strip(), None

        declared = SupportState(match.group(1).lower())
        body = (raw[: match.start()] + raw[match.end() :]).strip()
        # Remove a fenced block left empty by lifting the SUPPORT line out of it.
        body = re.sub(r"```\s*```", "", body).strip()
        return body, declared

    @staticmethod
    def _uncited_factual_sentences(answer: str, allowed: set[str]) -> list[str]:
        """Find assertive sentences carrying no citation marker."""
        if not allowed:
            return []

        uncited: list[str] = []
        for sentence in _SENTENCE_SPLIT_RE.split(answer):
            stripped = sentence.strip()
            # Short fragments, headings, and list scaffolding are not assertions.
            if len(stripped) < 40 or stripped.startswith(("#", "-", "*", "|", ">")):
                continue
            if not _MARKER_RE.search(stripped):
                uncited.append(stripped)
        return uncited

    @staticmethod
    def _resolve_support(
        declared: SupportState | None,
        has_evidence: bool,
        valid_citations: int,
        uncited_count: int,
        fabricated_count: int,
    ) -> SupportState:
        """Decide the final support state.

        The model's own claim is an input, not the verdict: it may only be
        downgraded by the evidence, never upgraded by its own assertion.
        """
        if not has_evidence or valid_citations == 0:
            return SupportState.INSUFFICIENT

        computed = SupportState.GROUNDED
        if fabricated_count or uncited_count:
            computed = SupportState.PARTIAL

        if declared is None:
            return computed

        severity = {
            SupportState.GROUNDED: 0,
            SupportState.PARTIAL: 1,
            SupportState.INSUFFICIENT: 2,
        }
        return declared if severity[declared] > severity[computed] else computed


_citation_validator: CitationValidator | None = None


def get_citation_validator() -> CitationValidator:
    """Return the singleton CitationValidator."""
    global _citation_validator
    if _citation_validator is None:
        _citation_validator = CitationValidator()
    return _citation_validator

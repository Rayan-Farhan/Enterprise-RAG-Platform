"""Oversized-block splitting shared by the Stage 5 strategies (Task 5.1)."""

from __future__ import annotations

import re

from app.ingestion.chunking.base import estimate_tokens

#: Sentence-ish boundary, so an over-budget element is cut where a reader would
#: pause rather than mid-clause.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")


def split_block(text: str, budget_tokens: int) -> list[tuple[str, int]]:
    """Break text into pieces no larger than the budget, on sentence boundaries.

    Returns ``(text, tokens)`` pairs. A single sentence longer than the budget is
    emitted whole rather than cut mid-clause: an unreadable fragment retrieves no
    better than an over-long chunk and reads far worse in a citation.
    """
    tokens = estimate_tokens(text)
    if tokens <= budget_tokens:
        return [(text, tokens)]

    pieces: list[tuple[str, int]] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in (s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()):
        sentence_tokens = estimate_tokens(sentence)
        if current and current_tokens + sentence_tokens > budget_tokens:
            joined = " ".join(current)
            pieces.append((joined, estimate_tokens(joined)))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        joined = " ".join(current)
        pieces.append((joined, estimate_tokens(joined)))

    return pieces

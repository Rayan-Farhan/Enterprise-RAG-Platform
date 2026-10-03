"""Parser benchmark scoring and the OpenDataLoader page re-anchoring (ADR-004)."""

from __future__ import annotations

from typing import Any

from app.ingestion.parsers.opendataloader_parser import _align_pages
from benchmarks.parser_benchmark import (
    in_order_coverage,
    match_headings,
    norm,
    positional_hits,
    tokens,
)


def test_tokens_fold_quotes_dashes_case_and_edge_punctuation() -> None:
    assert tokens("The employee’s “Leave” — (FMLA).") == ["the", "employee's", "leave", "fmla"]
    assert tokens("• 8 hours") == ["8", "hours"]
    assert tokens("half-­time") == ["half-time"]


def test_in_order_coverage_penalises_scrambled_order() -> None:
    needle = "a b c d".split()
    assert in_order_coverage(needle, "x a b c d y".split()) == 1.0
    assert in_order_coverage(needle, "d c b a".split()) == 0.25


def test_heading_match_counts_false_positives_but_not_table_cells() -> None:
    expected = [norm("Hours of Work"), norm("Breaks")]
    found = [
        norm("Hours of Work"),
        norm("Provisional Employee: The first three months"),
        norm("SERVICE"),
    ]
    tp, scored, total = match_headings(expected, found, neutral={norm("SERVICE")})
    assert (tp, scored, total) == (1, 2, 2)


def test_positional_hits_align_over_a_missing_header_row() -> None:
    gt = [["service", "benefit"], ["restorative services", "80%"]]
    without_header = [["restorative services", "80%"]]
    assert positional_hits(gt, without_header) == 2
    # Swapped columns: a whole-grid offset can line up one cell, never both.
    assert positional_hits(gt, [["80%", "restorative services"]]) < 2


def _tree(pages: dict[int, str], reported: int) -> dict[str, Any]:
    return {
        "number of pages": reported,
        "kids": [{"type": "paragraph", "page number": n, "content": t} for n, t in pages.items()],
    }


def test_align_pages_is_identity_when_counts_agree() -> None:
    tree = _tree({1: "alpha", 2: "beta"}, reported=2)
    assert _align_pages(tree, ["alpha", "beta"]) == {1: 1, 2: 2}


def test_align_pages_re_anchors_after_skipped_pages() -> None:
    physical = [
        "cover page title",
        "section divider contents",
        "administrative privileges policy computer access",
        "background checks investigation procedure",
        "divider again listing",
        "sick leave categories bereavement",
    ]
    # The engine skipped physical pages 1, 2 and 5 and numbered the rest 1..3.
    reported = {
        1: "administrative privileges policy computer access",
        2: "background checks investigation procedure",
        3: "sick leave categories bereavement",
    }
    assert _align_pages(_tree(reported, reported=3), physical) == {1: 3, 2: 4, 3: 6}

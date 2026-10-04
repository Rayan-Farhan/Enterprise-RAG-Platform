"""The results audit and claims checker (results verification, Phase 1)."""

from __future__ import annotations

from typing import Any

from scripts.audit_results import audit_pair, audit_run, recompute_metrics
from scripts.check_claims import ClaimContext, check


def _row(
    qid: str,
    qtype: str = "factual",
    retrieval: dict[str, float] | None = None,
    deterministic: dict[str, float] | None = None,
    error: str | None = None,
    abstained: bool = False,
    provider: str | None = "groq",
) -> dict[str, Any]:
    return {
        "question_id": qid,
        "question_type": qtype,
        "retrieval_metrics": retrieval or {},
        "deterministic_metrics": deterministic or {},
        "judge_metrics": {},
        "error": error,
        "abstained": abstained,
        "generator_provider": provider,
        "generator_model": "m" if provider else None,
        "latency_ms": 100.0,
        "retrieval_latency_ms": 50.0,
        "evaluated_at": "2026-10-03T00:00:00Z",
    }


def _run(rows: list[dict[str, Any]], metrics: dict[str, float], **extra: Any) -> dict[str, Any]:
    run: dict[str, Any] = {
        "dataset_split": "dev",
        "dataset_size": len(rows),
        "retrieval_only": False,
        "git_commit": "abc",
        "chunking_version": "v",
        "config_snapshot": {"chunking_version": "v"},
        "generator_provider": "groq",
        "generator_model": "m",
        "prompt_versions": {"answer": "answer_v1"},
        "prompt_hashes": {"answer": "h"},
        "metrics": metrics,
        "metrics_by_type": {},
        "system_metrics": {},
        "results": rows,
    }
    run.update(extra)
    return run


def test_recompute_skips_failed_and_treats_absent_metrics_as_neutral() -> None:
    rows = [
        _row("a", retrieval={"recall@5": 1.0}),
        _row("b", retrieval={"recall@5": 0.0}),
        _row("c", qtype="adversarial", deterministic={"abstention_correct": 1.0}),
        _row("d", retrieval={"recall@5": 1.0}, error="boom"),
    ]
    assert recompute_metrics(rows) == {"recall@5": 0.5, "abstention_correct": 1.0}


def test_audit_reports_a_stored_aggregate_that_does_not_recompute() -> None:
    rows = [_row("a", retrieval={"recall@5": 1.0}), _row("b", retrieval={"recall@5": 0.0})]
    audit = audit_run("x", _run(rows, {"recall@5": 0.6}), tolerance=1e-6)
    assert any(m.startswith("recall@5: stored 0.600000") for m in audit.metric_mismatches)


def test_audit_flags_unattributed_and_mixed_generators() -> None:
    rows = [_row("a", provider=None), _row("b", provider="gemini")]
    audit = audit_run("x", _run(rows, {}), tolerance=1e-6)
    assert any("no per-question generator" in f for f in audit.flags)
    assert any("mixes generators" in f for f in audit.flags)


def test_pair_flags_different_question_sets_and_prompts() -> None:
    a = _run([_row("q1"), _row("q2")], {})
    b = _run([_row("q1"), _row("q3")], {}, prompt_hashes={"answer": "other"})
    runs = {"a": a, "b": b}
    audits = {name: audit_run(name, data, 1e-6) for name, data in runs.items()}
    pair = audit_pair("a", "b", runs, audits)
    assert pair.only_baseline == ["q2"] and pair.only_candidate == ["q3"]
    assert any("prompt content differs" in f for f in pair.flags)


def test_claim_is_confirmed_at_the_stated_precision_only() -> None:
    rows = [_row("a", retrieval={"recall@5": 1.0}), _row("b", retrieval={"recall@5": 0.0})]
    context = ClaimContext({"experiment-001-x": _run(rows, {"recall@5": 0.4878})})
    claim = {"id": "c", "claim": "r@5", "expr": "m('001','recall@5')"}
    assert check({**claim, "stated": "0.488"}, context).status == "confirmed"
    assert check({**claim, "stated": "0.49"}, context).status == "confirmed"
    wrong = check({**claim, "stated": "0.4851"}, context)
    assert wrong.status == "corrected" and wrong.verified == "0.4878"


def test_claim_without_expression_passes_its_status_through() -> None:
    context = ClaimContext({})
    result = check({"id": "c", "claim": "x", "status": "pending", "note": "Phase 3"}, context)
    assert result.status == "pending" and result.note == "Phase 3"


def test_a_newer_measurement_supersedes_a_matching_record() -> None:
    old = [_row("a", retrieval={"recall@5": 0.0}), _row("b", retrieval={"recall@5": 1.0})]
    new = [_row("a", retrieval={"recall@5": 1.0}), _row("b", retrieval={"recall@5": 1.0})]
    context = ClaimContext(
        {
            "experiment-002-x": _run(old, {"recall@5": 0.5}),
            "experiment-002-x-verify": _run(new, {"recall@5": 1.0}),
        }
    )
    claim = {
        "id": "c",
        "claim": "r@5",
        "stated": "0.50",
        "expr": "m('experiment-002-x','recall@5')",
        "supersede_expr": "m('experiment-002-x-verify','recall@5')",
    }
    result = check(claim, context)
    assert result.status == "corrected" and result.verified == "1.00"
    assert "re-measured" in result.note

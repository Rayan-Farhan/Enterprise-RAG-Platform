"""`eval run` refuses configurations whose numbers would not mean what they claim."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from app.core.config import AppSettings
from app.evaluation import cli


def run_args(**overrides: Any) -> argparse.Namespace:
    parsed = cli.build_parser().parse_args(["run", "--name", "experiment-test"])
    for key, value in overrides.items():
        setattr(parsed, key, value)
    return parsed


def use_settings(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    configured = AppSettings(**{"INFERENCE_PROFILE": "hosted", **overrides})
    monkeypatch.setattr(cli, "get_settings", lambda: configured)


class _Reached(Exception):
    """Raised by the stubbed dataset loader: the guard let the run through."""


class TestJudgeProviderGuard:
    """A pinned generator must not be graded by a judge on the same provider."""

    async def test_judged_run_on_the_judges_provider_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        use_settings(monkeypatch, GENERATION_PROVIDER="groq", EVAL_JUDGE_PROVIDER="groq")

        assert await cli.cmd_run(run_args()) == 2
        assert "judge would score its own" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("overrides", "arg_overrides"),
        [
            ({"GENERATION_PROVIDER": "groq"}, {"judge": False}),
            ({"GENERATION_PROVIDER": "groq"}, {"retrieval_only": True}),
            ({"GENERATION_PROVIDER": "groq", "EVAL_JUDGE_ENABLED": False}, {}),
            ({"GENERATION_PROVIDER": "groq", "EVAL_JUDGE_PROVIDER": "gemini"}, {}),
            ({}, {}),
        ],
        ids=["no-judge", "retrieval-only", "judge-disabled", "judge-elsewhere", "unpinned"],
    )
    async def test_runs_without_a_self_judging_pair_proceed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        overrides: dict[str, Any],
        arg_overrides: dict[str, Any],
    ) -> None:
        use_settings(monkeypatch, **overrides)

        def reached(*_: object, **__: object) -> None:
            raise _Reached

        monkeypatch.setattr(cli, "load_split", reached)

        with pytest.raises(_Reached):
            await cli.cmd_run(run_args(**arg_overrides))

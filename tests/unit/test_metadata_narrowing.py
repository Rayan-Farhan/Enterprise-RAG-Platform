"""Metadata-driven candidate narrowing (Task 6.3)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from app.core.config import AppSettings
from app.retrieval.channels import get_retriever
from app.retrieval.lexical_store import build_filter
from app.retrieval.narrowing import (
    ConstraintExtractor,
    NarrowingRetriever,
    merge_filters,
)
from app.retrieval.schemas import RetrievalFilters, RetrievalResult, RetrievedChunk
from app.retrieval.vector_store import QdrantVectorStore


@pytest.fixture(scope="module")
def extractor() -> ConstraintExtractor:
    return ConstraintExtractor.from_file()


class TestExtraction:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("What is the dental plan deductible?", {"policy_type": "dental_benefits"}),
            ("What do the medical plans cover?", {"policy_type": "health_benefits"}),
            (
                "What does Academic Affairs require for tenure review?",
                {"department": "academic_affairs", "employee_type": "faculty"},
            ),
            (
                "What does the Staff Handbook say about overtime?",
                {"policy_type": "handbook", "employee_type": "staff"},
            ),
            ("How many days of annual leave do I get?", {}),
        ],
    )
    def test_questions_map_to_the_constraints_they_imply(
        self, extractor: ConstraintExtractor, query: str, expected: dict[str, str]
    ) -> None:
        assert extractor.extract(query).constraints == expected

    def test_two_values_for_one_field_cancel_narrowing_on_it(
        self, extractor: ConstraintExtractor
    ) -> None:
        """'health plans' (plural) once failed to match, wrongly narrowing to dental."""
        inferred = extractor.extract("Do the dental and health plans have different deductibles?")

        assert "policy_type" not in inferred.constraints
        assert inferred.conflicts == {"policy_type": ["dental_benefits", "health_benefits"]}

    def test_hr_matches_only_as_the_acronym(self, extractor: ConstraintExtractor) -> None:
        assert extractor.extract("Ask HR about leave").constraints == {
            "department": "human_resources"
        }
        assert extractor.extract("Their hr policy").constraints == {}

    def test_every_match_is_recorded_with_its_triggering_term(
        self, extractor: ConstraintExtractor
    ) -> None:
        matches = extractor.extract("Are orthodontics covered?").matches

        assert matches == [
            {"field": "policy_type", "value": "dental_benefits", "term": "orthodontics"}
        ]

    def test_rules_may_only_target_fields_both_engines_filter_on(self, tmp_path: Path) -> None:
        path = tmp_path / "vocab.json"
        path.write_text(
            json.dumps(
                {"version": "t", "rules": [{"field": "grade", "value": "7", "patterns": ["x"]}]}
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unsupported field"):
            ConstraintExtractor.from_file(path)

    def test_vocabulary_values_exist_in_the_curated_corpus_metadata(
        self, extractor: ConstraintExtractor
    ) -> None:
        """A rule for a value no document carries would narrow every match to nothing."""
        corpus = Path(__file__).resolve().parents[2] / "benchmarks/corpus/metadata.json"
        documents = json.loads(corpus.read_text(encoding="utf-8"))["documents"].values()
        for rule in extractor.rules:
            assert any(doc.get(rule.field) == rule.value for doc in documents), rule.value


class TestFilters:
    def test_explicit_filters_win_over_inferred_ones(self) -> None:
        merged = merge_filters(
            RetrievalFilters(policy_type="health_benefits"),
            {"policy_type": "dental_benefits", "employee_type": "faculty"},
        )

        assert merged.policy_type == "health_benefits"
        assert merged.employee_type == "faculty"

    def test_employee_type_also_admits_documents_that_apply_to_everyone(self) -> None:
        """A faculty question must still see the university-wide policy manual."""
        filters = RetrievalFilters(employee_type="faculty")

        assert {"terms": {"employee_type": ["faculty", "all"]}} in build_filter(filters)
        qdrant = QdrantVectorStore.build_filter(filters)
        assert qdrant is not None
        condition = qdrant.must[0]  # type: ignore[index]
        assert condition.key == "employee_type"
        assert set(condition.match.any) == {"faculty", "all"}  # type: ignore[union-attr]


class _FakeChannel:
    channel = "fake"

    def __init__(self, empty_when_filtered: bool = False) -> None:
        self.settings = AppSettings(APP_ENV="testing")
        self.empty_when_filtered = empty_when_filtered
        self.calls: list[RetrievalFilters | None] = []

    async def candidate_count(self, filters: RetrievalFilters | None) -> int:
        return 251 if filters and filters.policy_type else 1937

    async def retrieve(
        self,
        query: str,
        session: Any = None,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        self.calls.append(filters)
        narrowed = filters is not None and filters.policy_type is not None
        chunks = (
            []
            if narrowed and self.empty_when_filtered
            else [
                RetrievedChunk(
                    chunk_id=uuid.uuid4(),
                    document_id=uuid.uuid4(),
                    version_id=uuid.uuid4(),
                    content="x",
                    score=1.0,
                )
            ]
        )
        return RetrievalResult(query=query, chunks=chunks, retrieval_config={"channels": ["fake"]})


class TestNarrowingRetriever:
    async def test_inferred_constraints_reach_the_channel_as_filters(self) -> None:
        inner = _FakeChannel()
        result = await NarrowingRetriever(inner).retrieve("dental deductible")  # type: ignore[arg-type]

        assert inner.calls[0] is not None and inner.calls[0].policy_type == "dental_benefits"
        trace = result.retrieval_config["narrowing"]
        assert trace["applied"] == {"policy_type": "dental_benefits"}
        assert (trace["candidates_before"], trace["candidates_after"]) == (1937, 251)
        assert trace["fallback"] is False
        assert result.retrieval_config["channels"] == ["fake"]  # channel config kept

    async def test_an_empty_narrowed_search_falls_back_to_the_full_corpus(self) -> None:
        inner = _FakeChannel(empty_when_filtered=True)
        result = await NarrowingRetriever(inner).retrieve("dental deductible")  # type: ignore[arg-type]

        assert result.chunks
        assert result.retrieval_config["narrowing"]["fallback"] is True
        assert inner.calls[1] is None

    async def test_a_question_implying_nothing_passes_through_with_a_trace(self) -> None:
        inner = _FakeChannel()
        result = await NarrowingRetriever(inner).retrieve("annual leave")  # type: ignore[arg-type]

        assert inner.calls == [None]
        trace = result.retrieval_config["narrowing"]
        assert trace["applied"] == {} and trace["candidates_after"] is None

    async def test_the_trace_names_the_vocabulary_version(self) -> None:
        result = await NarrowingRetriever(_FakeChannel()).retrieve("dental")  # type: ignore[arg-type]
        assert result.retrieval_config["narrowing"]["vocabulary"] == "v2"


class TestToggle:
    def test_narrowing_is_off_by_default(self) -> None:
        settings = AppSettings(APP_ENV="testing")
        assert settings.ENABLE_METADATA_NARROWING is False
        assert not isinstance(get_retriever(settings), NarrowingRetriever)

    def test_enabling_it_wraps_whichever_channel_is_selected(self) -> None:
        retriever = get_retriever(
            AppSettings(APP_ENV="testing", RETRIEVAL_MODE="bm25", ENABLE_METADATA_NARROWING=True)
        )
        assert isinstance(retriever, NarrowingRetriever)
        assert retriever.channel == "bm25"

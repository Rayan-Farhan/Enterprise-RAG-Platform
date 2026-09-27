"""The feedback loop end to end over a real ORM (Tasks 13.3–13.4, master §55).

answer recorded → feedback submitted → reviewer promotes or dismisses →
candidate written → maintainer accepts it into a split.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.testclient import TestClient

from app.db.models.base import Base
from app.db.session import get_db_session
from app.evaluation.candidates import accept_candidates, load_candidates
from app.evaluation.dataset import DatasetError, load_split
from app.evaluation.schemas import DatasetSplit
from app.feedback.service import FeedbackService, get_feedback_service
from app.generation.citation import SupportState
from app.generation.service import AnswerResult
from app.main import app
from app.retrieval.schemas import RetrievedChunk

DOC_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
VER_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def _chunk(element: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=DOC_ID,
        version_id=VER_ID,
        content=f"Document: Staff Handbook\n\nPassage on page {page}.",
        score=0.8,
        page_number=page,
        page_span=[page],
        section_path=["Leave"],
        element_ids=[element],
        document_title="Staff Handbook",
        version_number=1,
    )


def _answer() -> AnswerResult:
    chunks = [_chunk("e1", 3), _chunk("e2", 7)]
    return AnswerResult(
        query="How many days of annual leave do I get?",
        answer="You get 20 days [1].",
        support=SupportState.GROUNDED,
        context_chunks=chunks,
        context_chunk_ids=[c.chunk_id for c in chunks],
        provider="groq",
        model_name="openai/gpt-oss-120b",
        prompt_versions={"answer": "answer_v2"},
    )


@pytest.fixture
async def session_factory() -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture
def candidates_path(tmp_path: Path) -> Path:
    return tmp_path / "candidates" / "feedback_candidates_v1.jsonl"


@pytest.fixture
def service(candidates_path: Path) -> FeedbackService:
    return FeedbackService(candidates_path=candidates_path)


@pytest.fixture
def client(
    session_factory: async_sessionmaker[AsyncSession], service: FeedbackService
) -> Generator[TestClient, None, None]:
    async def _session() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session
            await session.commit()

    app.dependency_overrides[get_db_session] = _session
    app.dependency_overrides[get_feedback_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
async def answer_id(
    session_factory: async_sessionmaker[AsyncSession], service: FeedbackService
) -> uuid.UUID:
    async with session_factory() as session:
        return await service.record_answer(session, _answer())


def _rate(client: TestClient, answer_id: uuid.UUID, **fields: object) -> dict:
    body = {"answer_id": str(answer_id), "helpful": False, **fields}
    response = client.post("/api/v1/feedback", json=body)
    assert response.status_code == 201, response.text
    return response.json()


class TestAnswerRecord:
    def test_evidence_is_stored_under_the_markers_the_model_saw(
        self, client: TestClient, answer_id: uuid.UUID
    ) -> None:
        answer = client.get(f"/api/v1/answers/{answer_id}").json()

        assert [e["marker"] for e in answer["evidence"]] == ["1", "2"]
        assert answer["evidence"][1]["element_ids"] == ["e2"]
        # The contextual prefix is retrieval scaffolding, not policy text.
        assert answer["evidence"][0]["text"] == "Passage on page 3."
        assert answer["prompt_versions"] == {"answer": "answer_v2"}

    def test_unknown_answer_is_404(self, client: TestClient) -> None:
        assert client.get(f"/api/v1/answers/{uuid.uuid4()}").status_code == 404


class TestSubmitFeedback:
    def test_structured_judgements_are_kept_distinct_from_not_assessed(
        self, client: TestClient, answer_id: uuid.UUID
    ) -> None:
        feedback = _rate(
            client, answer_id, answer_correct=False, citations_correct=True, comment="  It is 25. "
        )

        assert feedback["answer_correct"] is False
        assert feedback["citations_correct"] is True
        assert feedback["answer_complete"] is None
        assert feedback["comment"] == "It is 25."
        assert feedback["status"] == "new"
        assert feedback["answer"]["query"].startswith("How many days")

    def test_feedback_on_an_unknown_answer_is_404(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/feedback", json={"answer_id": str(uuid.uuid4()), "helpful": True}
        )
        assert response.status_code == 404

    def test_queue_filters_by_status_oldest_first(
        self, client: TestClient, answer_id: uuid.UUID
    ) -> None:
        first = _rate(client, answer_id)
        second = _rate(client, answer_id, helpful=True)
        client.post(f"/api/v1/feedback/{second['id']}/dismiss", json={})

        queue = client.get("/api/v1/feedback", params={"status": "new"}).json()
        everything = client.get("/api/v1/feedback").json()

        assert [f["id"] for f in queue["items"]] == [first["id"]]
        assert queue["total"] == 1
        assert everything["total"] == 2


class TestReview:
    def test_promotion_writes_a_valid_candidate_with_element_evidence(
        self, client: TestClient, answer_id: uuid.UUID, candidates_path: Path
    ) -> None:
        feedback = _rate(client, answer_id, comment="Handbook says 25 days")

        response = client.post(
            f"/api/v1/feedback/{feedback['id']}/promote",
            json={
                "question_type": "factual",
                "acceptable_answer": "25 days of annual leave.",
                "evidence_markers": ["2"],
                "must_contain": ["25"],
                "reviewer_note": "Model cited the old version",
            },
        )

        assert response.status_code == 200, response.text
        promoted = response.json()
        assert promoted["status"] == "promoted"
        [candidate] = load_candidates(candidates_path)
        assert candidate.question_id == promoted["candidate_question_id"]
        assert candidate.split is DatasetSplit.DEV
        assert candidate.source == "feedback"
        assert candidate.expected_evidence[0].element_ids == ["e2"]
        assert candidate.expected_evidence[0].page_numbers == [7]
        assert candidate.required_citations == 1
        assert "Handbook says 25 days" in (candidate.notes or "")

    def test_an_abstention_candidate_carries_no_evidence(
        self, client: TestClient, answer_id: uuid.UUID, candidates_path: Path
    ) -> None:
        feedback = _rate(client, answer_id)

        response = client.post(
            f"/api/v1/feedback/{feedback['id']}/promote",
            json={
                "question_type": "negative_unsupported",
                "acceptable_answer": "The corpus does not cover this; the system must abstain.",
                "must_abstain": True,
            },
        )

        assert response.status_code == 200, response.text
        [candidate] = load_candidates(candidates_path)
        assert candidate.must_abstain and candidate.expected_evidence == []

    @pytest.mark.parametrize(
        "body",
        [
            # Evidence the model never saw cannot be the expected evidence.
            {"question_type": "factual", "acceptable_answer": "x", "evidence_markers": ["9"]},
            # An answerable question without evidence would score every retrieval perfect.
            {"question_type": "factual", "acceptable_answer": "x", "evidence_markers": []},
            # The locked split is opened once, at Stage 14.
            {
                "question_type": "factual",
                "acceptable_answer": "x",
                "evidence_markers": ["1"],
                "split": "test",
            },
        ],
    )
    def test_invalid_promotions_are_refused_and_leave_the_feedback_open(
        self, client: TestClient, answer_id: uuid.UUID, candidates_path: Path, body: dict
    ) -> None:
        feedback = _rate(client, answer_id)

        response = client.post(f"/api/v1/feedback/{feedback['id']}/promote", json=body)

        assert response.status_code == 422
        assert load_candidates(candidates_path) == []
        queue = client.get("/api/v1/feedback", params={"status": "new"}).json()
        assert queue["total"] == 1

    def test_reviewed_feedback_cannot_be_reviewed_twice(
        self, client: TestClient, answer_id: uuid.UUID
    ) -> None:
        feedback = _rate(client, answer_id)
        client.post(f"/api/v1/feedback/{feedback['id']}/dismiss", json={"reviewer_note": "dup"})

        again = client.post(f"/api/v1/feedback/{feedback['id']}/dismiss", json={})

        assert again.status_code == 409

    def test_reopen_withdraws_the_pending_candidate(
        self, client: TestClient, answer_id: uuid.UUID, candidates_path: Path
    ) -> None:
        feedback = _rate(client, answer_id)
        client.post(
            f"/api/v1/feedback/{feedback['id']}/promote",
            json={
                "question_type": "factual",
                "acceptable_answer": "25 days.",
                "evidence_markers": ["1"],
            },
        )

        reopened = client.post(f"/api/v1/feedback/{feedback['id']}/reopen").json()

        assert reopened["status"] == "new"
        assert reopened["candidate_question_id"] is None
        assert load_candidates(candidates_path) == []


class TestAcceptCandidates:
    def _promote(self, client: TestClient, answer_id: uuid.UUID, split: str = "dev") -> str:
        feedback = _rate(client, answer_id)
        response = client.post(
            f"/api/v1/feedback/{feedback['id']}/promote",
            json={
                "question_type": "factual",
                "acceptable_answer": "25 days.",
                "evidence_markers": ["1"],
                "split": split,
            },
        )
        return str(response.json()["candidate_question_id"])

    def _seed_split(self, base_dir: Path, split: str) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        existing = {
            "question_id": f"{split}-factual-001",
            "question": "What is the probation period?",
            "question_type": "factual",
            "split": split,
            "acceptable_answer": "Three months.",
            "expected_evidence": [
                {"document_id": str(DOC_ID), "version_id": str(VER_ID), "element_ids": ["e9"]}
            ],
        }
        (base_dir / f"golden_dataset_{split}_v1.jsonl").write_text(
            json.dumps(existing) + "\n", encoding="utf-8"
        )

    def test_accepted_candidates_join_their_split_and_leave_the_queue(
        self,
        client: TestClient,
        answer_id: uuid.UUID,
        candidates_path: Path,
        tmp_path: Path,
    ) -> None:
        base_dir = tmp_path / "datasets"
        self._seed_split(base_dir, "dev")
        question_id = self._promote(client, answer_id)

        accepted = accept_candidates(path=candidates_path, base_dir=base_dir)

        assert [c.question_id for c in accepted] == [question_id]
        dev = load_split(DatasetSplit.DEV, base_dir=base_dir)
        assert [q.question_id for q in dev] == ["dev-factual-001", question_id]
        assert load_candidates(candidates_path) == []

    def test_accepting_an_id_that_is_not_pending_fails_without_writing(
        self,
        client: TestClient,
        answer_id: uuid.UUID,
        candidates_path: Path,
        tmp_path: Path,
    ) -> None:
        base_dir = tmp_path / "datasets"
        self._seed_split(base_dir, "dev")
        self._promote(client, answer_id)

        with pytest.raises(DatasetError):
            accept_candidates(["dev-feedback-nope"], path=candidates_path, base_dir=base_dir)

        assert len(load_split(DatasetSplit.DEV, base_dir=base_dir)) == 1
        assert len(load_candidates(candidates_path)) == 1

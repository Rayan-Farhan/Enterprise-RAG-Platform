"""Unit tests for domain exceptions and RFC-7807 problem details."""

from app.core.exceptions import (
    AuthenticationException,
    AuthorizationException,
    CircuitBreakerOpenException,
    ConflictException,
    ModelProviderException,
    NotFoundException,
    ProblemDetail,
    RateLimitExceededException,
    ValidationException,
)


def test_problem_detail_serialization() -> None:
    problem = ProblemDetail(
        title="Resource Not Found",
        status=404,
        detail="Document with ID 123 was not found.",
        instance="/api/v1/documents/123",
        code="NOT_FOUND",
    )
    dumped = problem.model_dump(exclude_none=True)
    assert dumped["status"] == 404
    assert dumped["code"] == "NOT_FOUND"
    assert dumped["detail"] == "Document with ID 123 was not found."


def test_exception_status_codes() -> None:
    assert NotFoundException().status_code == 404
    assert ValidationException().status_code == 422
    assert AuthenticationException().status_code == 401
    assert AuthorizationException().status_code == 403
    assert ConflictException().status_code == 409
    assert RateLimitExceededException().status_code == 429
    assert ModelProviderException("Failed", provider="gemini").status_code == 502
    assert CircuitBreakerOpenException("qdrant").status_code == 503

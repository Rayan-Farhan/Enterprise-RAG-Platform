"""Domain exception hierarchy and RFC-7807 problem details specification."""

from typing import Any

from pydantic import BaseModel, Field


class ProblemDetail(BaseModel):
    """RFC-7807 compliant problem details response."""

    type: str = Field(
        default="about:blank", description="URI reference identifying the problem type"
    )
    title: str = Field(description="Short, human-readable summary of the problem type")
    status: int = Field(description="HTTP status code")
    detail: str = Field(description="Human-readable explanation specific to this occurrence")
    instance: str | None = Field(
        default=None, description="URI reference identifying the specific occurrence"
    )
    code: str | None = Field(default=None, description="Internal error code for tracking")
    invalid_params: list[dict[str, Any]] | None = Field(
        default=None, description="Validation errors details"
    )


class AppException(Exception):
    """Base exception for all domain exceptions in Enterprise RAG Platform."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        code: str = "INTERNAL_SERVER_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details or {}


class NotFoundException(AppException):
    """Raised when a requested resource is not found."""

    def __init__(
        self, message: str = "Resource not found", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message=message, status_code=404, code="NOT_FOUND", details=details)


class ValidationException(AppException):
    """Raised when input validation fails."""

    def __init__(
        self, message: str = "Validation failed", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message=message, status_code=422, code="VALIDATION_ERROR", details=details)


class AuthenticationException(AppException):
    """Raised when authentication fails."""

    def __init__(
        self, message: str = "Authentication required", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message=message, status_code=401, code="UNAUTHORIZED", details=details)


class AuthorizationException(AppException):
    """Raised when authorization fails."""

    def __init__(
        self, message: str = "Permission denied", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message=message, status_code=403, code="FORBIDDEN", details=details)


class ConflictException(AppException):
    """Raised when an operation conflicts with current state."""

    def __init__(
        self, message: str = "Conflict occurred", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message=message, status_code=409, code="CONFLICT", details=details)


class RateLimitExceededException(AppException):
    """Raised when a rate limit is exceeded."""

    def __init__(
        self, message: str = "Rate limit exceeded", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message, status_code=429, code="RATE_LIMIT_EXCEEDED", details=details
        )


class ModelProviderException(AppException):
    """Raised when an inference provider fails."""

    def __init__(self, message: str, provider: str, details: dict[str, Any] | None = None) -> None:
        err_details = details or {}
        err_details["provider"] = provider
        super().__init__(
            message=message, status_code=502, code="MODEL_PROVIDER_ERROR", details=err_details
        )


class CircuitBreakerOpenException(AppException):
    """Raised when an operation is rejected by an open circuit breaker."""

    def __init__(self, service_name: str) -> None:
        super().__init__(
            message=f"Service '{service_name}' is currently unavailable (circuit open)",
            status_code=503,
            code="CIRCUIT_BREAKER_OPEN",
            details={"service": service_name},
        )

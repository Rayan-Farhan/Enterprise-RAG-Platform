"""Health check endpoints (ADR-001, Task 3.7, master §44)."""

from typing import Any

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.health import check_dependencies, overall_status

router = APIRouter(prefix="/health", tags=["Health"])


class HealthStatus(BaseModel):
    status: str = Field(description="Overall status (healthy, degraded, unhealthy)")
    version: str = Field(description="Application version")
    environment: str = Field(description="Current running environment")


class DependencyHealth(BaseModel):
    name: str
    status: str  # healthy, degraded, unreachable
    required: bool
    latency_ms: float | None = None
    error: str | None = None
    details: dict[str, Any] | None = None


class DetailedHealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    dependencies: list[DependencyHealth]


@router.get("/live", response_model=HealthStatus, summary="Liveness probe")
async def liveness_probe() -> HealthStatus:
    """Liveness probe indicating whether the process is alive.

    Deliberately checks nothing external: a dependency outage must not cause the
    orchestrator to restart an otherwise healthy process.
    """
    settings = get_settings()
    return HealthStatus(
        status="healthy",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
    )


@router.get("/ready", response_model=DetailedHealthResponse, summary="Readiness probe")
async def readiness_probe(response: Response) -> DetailedHealthResponse:
    """Readiness probe indicating whether the service can accept user traffic.

    Required dependencies (PostgreSQL, Qdrant, MinIO) failing yields 503 and takes
    the instance out of rotation. Optional dependencies failing reports `degraded`
    and keeps serving.
    """
    settings = get_settings()
    reports = await check_dependencies(settings)
    overall = overall_status(reports)

    if overall == "unhealthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return DetailedHealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        dependencies=[
            DependencyHealth(
                name=report.name,
                status=str(report.status),
                required=report.required,
                latency_ms=report.latency_ms,
                error=report.error,
            )
            for report in reports
        ],
    )


@router.get(
    "/dependencies",
    response_model=DetailedHealthResponse,
    summary="Per-dependency health detail",
)
async def dependency_health() -> DetailedHealthResponse:
    """Report every dependency's health without affecting the HTTP status.

    Always returns 200 so it stays usable as an operator diagnostic while
    readiness is failing.
    """
    settings = get_settings()
    reports = await check_dependencies(settings)

    return DetailedHealthResponse(
        status=overall_status(reports),
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        dependencies=[
            DependencyHealth(
                name=report.name,
                status=str(report.status),
                required=report.required,
                latency_ms=report.latency_ms,
                error=report.error,
            )
            for report in reports
        ],
    )

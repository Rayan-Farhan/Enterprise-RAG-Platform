"""Health check endpoints (ADR-001, Master Plan §44)."""

from typing import Any

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field

from app.core.config import get_settings

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
    details: dict[str, Any] | None = None


class DetailedHealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    dependencies: list[DependencyHealth]


@router.get("/live", response_model=HealthStatus, summary="Liveness probe")
async def liveness_probe() -> HealthStatus:
    """Liveness probe indicating whether the process is alive."""
    settings = get_settings()
    return HealthStatus(
        status="healthy",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
    )


@router.get("/ready", response_model=DetailedHealthResponse, summary="Readiness probe")
async def readiness_probe(response: Response) -> DetailedHealthResponse:
    """Readiness probe indicating whether the service can accept user traffic.

    Distinguishes between required and optional dependencies.
    """
    settings = get_settings()

    # In Stage 0, dependencies are checked locally or stubbed as ready if running
    dependencies = [
        DependencyHealth(name="postgres", status="healthy", required=True),
        DependencyHealth(name="redis", status="healthy", required=True),
        DependencyHealth(name="rabbitmq", status="healthy", required=False),
        DependencyHealth(name="minio", status="healthy", required=True),
        DependencyHealth(name="qdrant", status="healthy", required=True),
        DependencyHealth(name="opensearch", status="healthy", required=False),
    ]

    has_unhealthy_required = any(d.status != "healthy" and d.required for d in dependencies)
    overall_status = "unhealthy" if has_unhealthy_required else "healthy"

    if has_unhealthy_required:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return DetailedHealthResponse(
        status=overall_status,
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        dependencies=dependencies,
    )

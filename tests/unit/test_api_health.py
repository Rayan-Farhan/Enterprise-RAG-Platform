"""Unit tests for FastAPI root and health check endpoints (Task 3.7, master §44)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import app.core.health as health_module
from app.core.health import DependencyReport, DependencyStatus, overall_status


def test_root_endpoint(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Enterprise Multimodal RAG Platform"
    assert data["status"] == "running"
    assert "version" in data
    assert "inference_profile" in data


def test_health_live_endpoint(client: TestClient) -> None:
    """Liveness must not depend on any external service."""
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.fixture
def stub_probes(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, DependencyStatus]]:
    """Replace the real dependency probes with a controllable status map."""
    statuses: dict[str, DependencyStatus] = {
        name: DependencyStatus.HEALTHY for name, _, _ in health_module._PROBES
    }

    async def fake_check(settings: object = None) -> list[DependencyReport]:
        return [
            DependencyReport(
                name=name,
                status=statuses[name],
                required=required,
                latency_ms=1.0,
                error=None if statuses[name] is DependencyStatus.HEALTHY else "induced failure",
            )
            for name, required, _ in health_module._PROBES
        ]

    monkeypatch.setattr("app.api.v1.health.check_dependencies", fake_check)
    yield statuses


class TestReadiness:
    def test_all_healthy_returns_200(
        self, client: TestClient, stub_probes: dict[str, DependencyStatus]
    ) -> None:
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        names = {d["name"] for d in data["dependencies"]}
        assert {"postgres", "qdrant", "minio", "redis", "rabbitmq", "opensearch"} <= names

    def test_required_dependencies_are_marked_required(
        self, client: TestClient, stub_probes: dict[str, DependencyStatus]
    ) -> None:
        data = client.get("/api/v1/health/ready").json()
        required = {d["name"] for d in data["dependencies"] if d["required"]}
        assert required == {"postgres", "qdrant", "minio"}

    def test_qdrant_down_flips_readiness_to_503(
        self, client: TestClient, stub_probes: dict[str, DependencyStatus]
    ) -> None:
        """The Task 3.7 done-when: stopping Qdrant must remove the instance."""
        stub_probes["qdrant"] = DependencyStatus.UNREACHABLE
        response = client.get("/api/v1/health/ready")

        assert response.status_code == 503
        assert response.json()["status"] == "unhealthy"

    @pytest.mark.parametrize("dependency", ["postgres", "minio"])
    def test_each_required_dependency_down_fails_readiness(
        self, client: TestClient, stub_probes: dict[str, DependencyStatus], dependency: str
    ) -> None:
        stub_probes[dependency] = DependencyStatus.UNREACHABLE
        assert client.get("/api/v1/health/ready").status_code == 503

    @pytest.mark.parametrize("dependency", ["redis", "rabbitmq", "opensearch"])
    def test_optional_dependency_down_keeps_serving(
        self, client: TestClient, stub_probes: dict[str, DependencyStatus], dependency: str
    ) -> None:
        """The other half of the done-when: an optional outage must not evict us."""
        stub_probes[dependency] = DependencyStatus.UNREACHABLE
        response = client.get("/api/v1/health/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "degraded"

    def test_failure_reason_is_surfaced(
        self, client: TestClient, stub_probes: dict[str, DependencyStatus]
    ) -> None:
        stub_probes["qdrant"] = DependencyStatus.UNREACHABLE
        data = client.get("/api/v1/health/ready").json()
        qdrant = next(d for d in data["dependencies"] if d["name"] == "qdrant")

        assert qdrant["status"] == "unreachable"
        assert qdrant["error"] == "induced failure"


class TestDependenciesEndpoint:
    def test_returns_200_even_when_required_dependency_is_down(
        self, client: TestClient, stub_probes: dict[str, DependencyStatus]
    ) -> None:
        """Stays usable as a diagnostic while readiness is failing."""
        stub_probes["postgres"] = DependencyStatus.UNREACHABLE
        response = client.get("/api/v1/health/dependencies")

        assert response.status_code == 200
        assert response.json()["status"] == "unhealthy"


class TestOverallStatus:
    def test_healthy_when_everything_passes(self) -> None:
        reports = [
            DependencyReport("postgres", DependencyStatus.HEALTHY, required=True),
            DependencyReport("redis", DependencyStatus.HEALTHY, required=False),
        ]
        assert overall_status(reports) == "healthy"

    def test_degraded_when_only_optional_fails(self) -> None:
        reports = [
            DependencyReport("postgres", DependencyStatus.HEALTHY, required=True),
            DependencyReport("redis", DependencyStatus.UNREACHABLE, required=False),
        ]
        assert overall_status(reports) == "degraded"

    def test_unhealthy_when_required_fails(self) -> None:
        reports = [
            DependencyReport("postgres", DependencyStatus.UNREACHABLE, required=True),
            DependencyReport("redis", DependencyStatus.HEALTHY, required=False),
        ]
        assert overall_status(reports) == "unhealthy"

    def test_blocks_readiness_only_for_required(self) -> None:
        assert DependencyReport("qdrant", DependencyStatus.UNREACHABLE, True).blocks_readiness
        assert not DependencyReport("redis", DependencyStatus.UNREACHABLE, False).blocks_readiness


class TestProbeIsolation:
    async def test_a_raising_probe_becomes_an_unreachable_report(self) -> None:
        """A probe must never propagate an exception into the health response."""

        async def exploding() -> None:
            raise ConnectionError("connection refused")

        report = await health_module._probe("thing", True, exploding)

        assert report.status is DependencyStatus.UNREACHABLE
        assert report.error is not None
        assert "connection refused" in report.error

    async def test_a_hanging_probe_times_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio

        monkeypatch.setattr(health_module, "PROBE_TIMEOUT_SECONDS", 0.05)

        async def hanging() -> None:
            await asyncio.sleep(10)

        report = await health_module._probe("slow", True, hanging)

        assert report.status is DependencyStatus.UNREACHABLE
        assert report.error is not None
        assert "timeout" in report.error

    async def test_a_passing_probe_reports_healthy_with_latency(self) -> None:
        async def ok() -> None:
            return None

        report = await health_module._probe("fine", False, ok)

        assert report.status is DependencyStatus.HEALTHY
        assert report.error is None
        assert report.latency_ms is not None

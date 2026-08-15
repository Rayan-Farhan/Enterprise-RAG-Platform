"""Unit tests for FastAPI root and health check endpoints."""

from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Enterprise Multimodal RAG Platform"
    assert data["status"] == "running"
    assert "version" in data
    assert "inference_profile" in data


def test_health_live_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_health_ready_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "unhealthy")
    assert len(data["dependencies"]) >= 6
    names = [d["name"] for d in data["dependencies"]]
    assert "postgres" in names
    assert "qdrant" in names
    assert "opensearch" in names
    assert "minio" in names

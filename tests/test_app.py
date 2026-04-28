"""Test FastAPI app endpoints."""

from fastapi.testclient import TestClient

from legal_agent.api.app import create_app


def test_root_endpoint() -> None:
    """Root should return service info."""
    client = TestClient(create_app())
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "legal-agent"
    assert "version" in data


def test_health_endpoint() -> None:
    """Health check should return ok."""
    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

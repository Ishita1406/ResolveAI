import pytest
from fastapi.testclient import TestClient

from app.api import health
from app.config import get_settings
from app.main import create_app


async def _ok(_client) -> None:
    return None


async def _down(_client) -> None:
    raise ConnectionError("refused")


@pytest.mark.parametrize(
    ("redis_probe", "expected_code", "expected_status", "expected_redis"),
    [
        (_ok, 200, "ok", "ok"),
        (_down, 503, "degraded", "error"),
    ],
)
def test_health_reports_component_status(
    monkeypatch, redis_probe, expected_code, expected_status, expected_redis
):
    # Probes are stubbed so the test needs no running Postgres/Redis (CI-friendly).
    monkeypatch.setenv("POSTGRES_USER", "test")
    get_settings.cache_clear()
    monkeypatch.setattr(health, "ping_postgres", _ok)
    monkeypatch.setattr(health, "ping_redis", redis_probe)

    app = create_app()
    # TestClient is not used as a context manager, so lifespan (real connections) is skipped.
    app.state.engine = object()
    app.state.redis = object()

    response = TestClient(app).get("/health")

    assert response.status_code == expected_code
    body = response.json()
    assert body["status"] == expected_status
    assert body["checks"]["postgres"]["status"] == "ok"
    assert body["checks"]["redis"]["status"] == expected_redis

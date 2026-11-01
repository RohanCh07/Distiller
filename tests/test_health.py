import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from distiller.gateway.app import create_app  # noqa: E402
from distiller.gateway.config import GatewayConfig  # noqa: E402


def test_health(tmp_path):
    app = create_app(GatewayConfig(database=tmp_path / "d.db"))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"ok": True, "students": []}

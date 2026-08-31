from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_and_meta():
    health = client.get("/health")
    assert health.status_code in (200, 503)
    body = health.json()
    assert "disclaimer" in body
    assert body["data_vendor"] == "Yahoo Finance"
    meta = client.get("/api/meta")
    assert meta.status_code == 200
    assert meta.json()["estimate_kind"] == "next_session_close_from_last_close"


def test_resolve_endpoint():
    res = client.get("/api/resolve", params={"q": "outlook for Apple"})
    assert res.status_code == 200
    assert res.json()["ticker"] == "AAPL"


def test_home_renders_disclaimer():
    res = client.get("/")
    assert res.status_code == 200
    assert b"Not investment advice" in res.content or b"not investment advice" in res.content.lower()

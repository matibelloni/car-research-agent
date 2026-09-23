from fastapi.testclient import TestClient

import api
from agent import AgentResult

client = TestClient(api.app)
AUTH = {"X-API-Key": api.API_KEY}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_requires_api_key():
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 401


def test_ask_rejects_missing_question():
    response = client.post("/ask", json={}, headers=AUTH)
    assert response.status_code == 422


def test_ask_returns_result(monkeypatch):
    monkeypatch.setattr(
        api, "run_agent", lambda q: AgentResult(answer="Golf", tokens=10)
    )

    response = client.post("/ask", json={"question": "hi"}, headers=AUTH)

    assert response.status_code == 200
    assert response.json()["answer"] == "Golf"


def test_ask_maps_provider_error_to_http_status(monkeypatch):
    monkeypatch.setattr(
        api, "run_agent", lambda q: AgentResult(error="provider_unavailable")
    )

    response = client.post("/ask", json={"question": "hi"}, headers=AUTH)

    assert response.status_code == 503


def test_stream_requires_api_key():
    response = client.post("/ask/stream", json={"question": "hi"})
    assert response.status_code == 401


def test_stream_emits_sse_events(monkeypatch):
    def fake_stream(question):
        yield {"type": "searching", "query": "golf"}
        yield {
            "type": "result",
            "answer": "ok",
            "error": None,
            "tokens": 1,
            "searches": 1,
        }

    monkeypatch.setattr(api, "run_agent_stream", fake_stream)

    response = client.post("/ask/stream", json={"question": "hi"}, headers=AUTH)

    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"type": "searching", "query": "golf"}\n\n' in response.text

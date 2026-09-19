from __future__ import annotations

from fastapi.testclient import TestClient

from infer.sampling import SamplingParams
from infer.scheduler import Scheduler
from infer.server.api import create_app


def test_health_and_models(engine) -> None:
    app = create_app(engine, Scheduler(engine))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        models = client.get("/v1/models")
        assert models.status_code == 200
        assert models.json()["data"][0]["id"] == "tiny-test"


def test_chat_completions_non_stream(engine) -> None:
    engine.stop_token_ids = []
    app = create_app(engine, Scheduler(engine))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tiny-test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
                "temperature": 0,
                "stream": False,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)
    assert body["usage"]["completion_tokens"] == 4


def test_chat_completions_stream(engine) -> None:
    engine.stop_token_ids = []
    app = create_app(engine, Scheduler(engine))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tiny-test",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 3,
                "temperature": 0,
                "stream": True,
            },
        )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    assert "data:" in response.text
    assert "[DONE]" in response.text


def test_sampling_params_roundtrip() -> None:
    params = SamplingParams(max_tokens=3, temperature=0.0)
    assert params.max_tokens == 3

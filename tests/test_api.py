"""API tests with TestClient, a fake model, and a fake FAISS retriever."""

import pytest
import torch

from tests.helpers import TAGS, FakeRetriever, TinyEmbedModel


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from audiotag.api.engines import TorchEngine
    from audiotag.api.main import app
    from audiotag.config import Config

    fake_state = {
        "cfg": Config(),
        "device": torch.device("cpu"),
        "engine": TorchEngine(TinyEmbedModel(n_classes=len(TAGS), dim=8), torch.device("cpu")),
        "retriever": FakeRetriever(ntotal=5, dim=8),
        "tags": TAGS,
        "top_k": 5,
        "backend": "torch",
        "checkpoint_sha256": "0123456789abcdef",
    }
    monkeypatch.setattr("audiotag.api.main._load_models", lambda: fake_state)
    with TestClient(app) as test_client:
        yield test_client


def test_analyze_happy_path(client, wav_bytes):
    response = client.post("/analyze", files={"file": ("clip.wav", wav_bytes, "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert len(body["tags"]) == 5
    assert {"tag", "probability"} <= set(body["tags"][0])
    assert all(0.0 <= t["probability"] <= 1.0 for t in body["tags"])
    assert len(body["neighbors"]) == 5
    assert {"clip_id", "score", "tags"} <= set(body["neighbors"][0])
    assert "x-stage-forward-ms" in response.headers


def test_tag_endpoint(client, wav_bytes):
    response = client.post("/tag", files={"file": ("clip.wav", wav_bytes, "audio/wav")})
    assert response.status_code == 200
    assert "neighbors" not in response.json()
    assert len(response.json()["tags"]) == 5


def test_similar_endpoint(client, wav_bytes):
    response = client.post("/similar", files={"file": ("clip.wav", wav_bytes, "audio/wav")})
    assert response.status_code == 200
    assert "tags" not in response.json()
    assert len(response.json()["neighbors"]) == 5


def test_stereo_upload_works(client, stereo_wav_bytes):
    response = client.post("/analyze", files={"file": ("clip.wav", stereo_wav_bytes, "audio/wav")})
    assert response.status_code == 200


def test_unsupported_content_type_returns_415(client, wav_bytes):
    response = client.post("/analyze", files={"file": ("clip.txt", wav_bytes, "text/plain")})
    assert response.status_code == 415
    assert "unsupported content type" in response.json()["detail"]


def test_oversized_upload_returns_413(client, monkeypatch):
    monkeypatch.setattr("audiotag.api.main.MAX_UPLOAD_BYTES", 64)
    payload = b"\x00" * 512
    response = client.post("/analyze", files={"file": ("clip.wav", payload, "audio/wav")})
    assert response.status_code == 413


def test_empty_upload_returns_400(client):
    response = client.post("/analyze", files={"file": ("clip.wav", b"", "audio/wav")})
    assert response.status_code == 400


def test_healthz_shape(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "model",
        "index_type",
        "index_vectors",
        "device",
        "backend",
        "checkpoint_sha256",
    }
    assert body["index_vectors"] == 5
    assert body["device"] == "cpu"
    assert body["backend"] == "torch"
    assert body["checkpoint_sha256"] == "0123456789abcdef"


def test_responses_carry_request_id(client, wav_bytes):
    response = client.post("/analyze", files={"file": ("clip.wav", wav_bytes, "audio/wav")})
    assert "x-request-id" in response.headers
    assert len(response.headers["x-request-id"]) == 12


def _metric_value(text: str, metric: str, endpoint: str) -> int:
    prefix = f'{metric}{{endpoint="{endpoint}"}} '
    for line in text.splitlines():
        if line.startswith(prefix):
            return int(line[len(prefix) :])
    raise AssertionError(f"{metric} not found for {endpoint}")


def test_metrics_exposes_expected_names(client, wav_bytes):
    client.post("/analyze", files={"file": ("clip.wav", wav_bytes, "audio/wav")})
    text = client.get("/metrics").text
    assert 'api_requests_total{endpoint="/analyze"}' in text
    assert 'api_errors_total{endpoint="/analyze"}' in text
    assert 'api_request_duration_seconds_bucket{endpoint="/analyze",le="0.1"}' in text
    assert 'api_request_duration_seconds_bucket{endpoint="/analyze",le="+Inf"}' in text
    assert "api_request_duration_seconds_count" in text
    assert "api_request_duration_seconds_sum" in text
    assert _metric_value(text, "api_requests_total", "/analyze") >= 1


def test_metrics_counts_errors(client, monkeypatch):
    monkeypatch.setattr("audiotag.api.main.MAX_UPLOAD_BYTES", 16)
    before_errors = _metric_value(client.get("/metrics").text, "api_errors_total", "/analyze")
    client.post("/analyze", files={"file": ("c.wav", b"\x00" * 64, "audio/wav")})
    after_errors = _metric_value(client.get("/metrics").text, "api_errors_total", "/analyze")
    assert after_errors == before_errors + 1


def test_metrics_middleware_ignores_other_paths(client):
    client.get("/healthz")
    text = client.get("/metrics").text
    assert 'endpoint="/healthz"' not in text

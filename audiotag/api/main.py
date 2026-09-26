"""FastAPI serving layer: audio tagging + embedding similarity search.

Endpoints:
    POST /tag       upload audio, return top-k tags with sigmoid probabilities
    POST /similar   upload audio, return top-k nearest tracks from FAISS
    POST /analyze   both, computed in a single model pass
    GET  /healthz   model name, index type, index vector count, device
    GET  /metrics   Prometheus: request/error counters + latency histogram

The tagger and FAISS index are loaded once in the lifespan handler and reused
for every request. Inference runs under ``torch.inference_mode()`` on a
worker thread pool (via ``run_in_threadpool``) so the event loop is not
blocked. Uploads are capped at 20 MB and unsupported content types get 415.

Env vars: AUDIOTAG_MODEL (cnn), AUDIOTAG_INDEX_TYPE (flat),
AUDIOTAG_SERVE_DEVICE (auto), AUDIOTAG_TOP_K (20),
AUDIOTAG_MAX_UPLOAD_MB (20), AUDIOTAG_ARTIFACTS (artifacts),
AUDIOTAG_TORCH_THREADS (2).
"""

from __future__ import annotations

import asyncio
import io
import os
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from audiotag.config import Config
from audiotag.features.melspec import crop_or_pad, log_mel
from audiotag.index.search import Retriever
from audiotag.models import build_model
from audiotag.train import get_device

MODEL_NAME = os.environ.get("AUDIOTAG_MODEL", "cnn")
INDEX_TYPE = os.environ.get("AUDIOTAG_INDEX_TYPE", "flat")
SERVE_DEVICE = os.environ.get("AUDIOTAG_SERVE_DEVICE", "auto")
TOP_K = int(os.environ.get("AUDIOTAG_TOP_K", "20"))
MAX_UPLOAD_BYTES = int(float(os.environ.get("AUDIOTAG_MAX_UPLOAD_MB", "20")) * 1024 * 1024)
ARTIFACTS_DIR = Path(os.environ.get("AUDIOTAG_ARTIFACTS", "artifacts"))
TORCH_THREADS = int(os.environ.get("AUDIOTAG_TORCH_THREADS", "2"))
MAX_CONCURRENT = int(os.environ.get("AUDIOTAG_MAX_CONCURRENT", "8"))

ALLOWED_CONTENT_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/flac",
    "audio/x-flac",
}


# ---------------------------------------------------------------- schemas


class TagScore(BaseModel):
    tag: str
    probability: float = Field(..., description="sigmoid probability")


class Neighbor(BaseModel):
    clip_id: str
    score: float = Field(..., description="cosine similarity")
    tags: list[str] = Field(..., description="top ground-truth tags of the track")


class TagResponse(BaseModel):
    tags: list[TagScore]


class SimilarResponse(BaseModel):
    neighbors: list[Neighbor]


class AnalyzeResponse(BaseModel):
    tags: list[TagScore]
    neighbors: list[Neighbor]


class HealthResponse(BaseModel):
    model: str
    index_type: str
    index_vectors: int
    device: str


# ---------------------------------------------------------------- metrics


class Metrics:
    """Minimal Prometheus registry: counters + latency histogram per endpoint."""

    BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: dict[str, int] = defaultdict(int)
        self.errors: dict[str, int] = defaultdict(int)
        self.buckets: dict[str, dict[float, int]] = defaultdict(lambda: defaultdict(int))
        self.count: dict[str, int] = defaultdict(int)
        self.sum: dict[str, float] = defaultdict(float)

    def observe(self, endpoint: str, seconds: float, is_error: bool) -> None:
        with self._lock:
            self.requests[endpoint] += 1
            self.sum[endpoint] += seconds
            self.count[endpoint] += 1
            if is_error:
                self.errors[endpoint] += 1
            for bound in self.BUCKETS:
                if seconds <= bound:
                    self.buckets[endpoint][bound] += 1

    def render(self) -> str:
        with self._lock:
            endpoints = sorted(self.requests)
        lines = [
            "# HELP api_requests_total Total requests by endpoint.",
            "# TYPE api_requests_total counter",
            "# HELP api_errors_total Requests with HTTP status >= 400 by endpoint.",
            "# TYPE api_errors_total counter",
            "# HELP api_request_duration_seconds Request latency by endpoint.",
            "# TYPE api_request_duration_seconds histogram",
        ]
        for endpoint in endpoints:
            lines.append(f'api_requests_total{{endpoint="{endpoint}"}} {self.requests[endpoint]}')
            lines.append(f'api_errors_total{{endpoint="{endpoint}"}} {self.errors[endpoint]}')
            for bound in self.BUCKETS:
                lines.append(
                    f'api_request_duration_seconds_bucket{{endpoint="{endpoint}",le="{bound}"}} '
                    f"{self.buckets[endpoint][bound]}"
                )
            lines.append(
                f'api_request_duration_seconds_bucket{{endpoint="{endpoint}",le="+Inf"}} '
                f"{self.count[endpoint]}"
            )
            lines.append(
                f'api_request_duration_seconds_sum{{endpoint="{endpoint}"}} '
                f"{self.sum[endpoint]:.6f}"
            )
            lines.append(
                f'api_request_duration_seconds_count{{endpoint="{endpoint}"}} '
                f"{self.count[endpoint]}"
            )
        return "\n".join(lines) + "\n"


metrics = Metrics()
COUNTED_PATHS = {"/tag", "/similar", "/analyze"}
_inference_slots = asyncio.Semaphore(MAX_CONCURRENT)


# ---------------------------------------------------------------- inference


DECODE_BACKENDS = ("torchcodec", "soundfile", "ffmpeg")
_decode_backend: str | None = None
_decode_lock = threading.Lock()


def _probe_decode_backend() -> str:
    """Pick a working decode path once (torchcodec needs CUDA libs on linux).

    torchaudio 2.11 routes every load backend through torchcodec, which on
    Linux links CUDA libraries that a CPU-only torch lacks. In that case we
    fall back to spawning the ffmpeg CLI to decode to raw f32le mono 16 kHz.
    """
    global _decode_backend
    if _decode_backend is not None:
        return _decode_backend
    with _decode_lock:
        if _decode_backend is not None:
            return _decode_backend
        import struct

        wav = (
            b"RIFF"
            + struct.pack("<I", 36 + 320)
            + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
            + b"data"
            + struct.pack("<I", 320)
            + b"\x00" * 320
        )
        for backend in DECODE_BACKENDS:
            try:
                torchaudio.load(io.BytesIO(wav), backend=backend)
                _decode_backend = backend
                return backend
            except Exception:  # noqa: BLE001
                continue
        try:
            import soundfile as sf

            sf.read(io.BytesIO(wav), dtype="float32")
            _decode_backend = "libsndfile"
            return _decode_backend
        except Exception:  # noqa: BLE001
            pass
        if shutil.which("ffmpeg"):
            _decode_backend = "ffmpeg-cli"
            return _decode_backend
        raise RuntimeError(
            f"no working audio decode backend among {DECODE_BACKENDS}, "
            "libsndfile, and no ffmpeg binary"
        )


def _decode_libsndfile(data: bytes) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(audio.T)), int(sr)


def _decode_ffmpeg_cli(data: bytes) -> tuple[torch.Tensor, int]:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode(errors='replace')[:200]}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return torch.from_numpy(audio).unsqueeze(0), 16000


def _decode(data: bytes) -> tuple[torch.Tensor, int]:
    backend = _probe_decode_backend()
    if backend == "ffmpeg-cli":
        return _decode_ffmpeg_cli(data)
    if backend == "libsndfile":
        return _decode_libsndfile(data)
    return torchaudio.load(io.BytesIO(data), backend=backend)


def analyze_audio(
    data: bytes, *, need_neighbors: bool
) -> tuple[list[TagScore], list[Neighbor] | None, dict[str, float]]:
    """Decode -> log-mel -> single model pass (embedding + head) -> optional FAISS."""
    app = _app_state
    stages: dict[str, float] = {}

    t0 = time.perf_counter()
    waveform, sr = _decode(data)
    stages["decode_ms"] = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    mel = log_mel(
        waveform,
        sr,
        target_sr=app["cfg"].sample_rate,
        n_fft=app["cfg"].n_fft,
        win_length=app["cfg"].win_length,
        hop_length=app["cfg"].hop_length,
        n_mels=app["cfg"].n_mels,
    )
    mel = crop_or_pad(mel, app["cfg"].n_frames).unsqueeze(0).to(app["device"])
    stages["mel_ms"] = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    with torch.inference_mode():
        embedding = app["model"].embed(mel)  # [1, D]
        logits = app["model"].head(embedding)  # [1, C]
    probs = torch.sigmoid(logits[0]).float().cpu().numpy()
    stages["forward_ms"] = (time.perf_counter() - t2) * 1000.0

    neighbors: list[Neighbor] | None = None
    if need_neighbors:
        t3 = time.perf_counter()
        hits = app["retriever"].search(embedding.float().cpu().numpy(), k=app["top_k"])[0]
        neighbors = [
            Neighbor(clip_id=h["clip_id"], score=h["score"], tags=h["top_tags"]) for h in hits
        ]
        stages["faiss_ms"] = (time.perf_counter() - t3) * 1000.0

    k = min(app["top_k"], len(app["tags"]))
    order = np.argsort(-probs)[:k]
    tags = [TagScore(tag=app["tags"][i], probability=float(probs[i])) for i in order]
    return tags, neighbors, stages


# ---------------------------------------------------------------- app state


_app_state: dict = {}


def _load_models() -> dict:
    cfg = Config()
    device = get_device(SERVE_DEVICE)
    torch.set_num_threads(TORCH_THREADS)

    checkpoint = ARTIFACTS_DIR / MODEL_NAME / "best.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"checkpoint not found: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model"], n_classes=len(ckpt["tags"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    index_path = ARTIFACTS_DIR / "index" / f"{INDEX_TYPE}.faiss"
    retriever = Retriever(index_path, ARTIFACTS_DIR / "index" / "id_map.parquet")

    with torch.inference_mode():
        probe = model.embed(torch.zeros(1, 1, cfg.n_mels, cfg.n_frames, device=device))
    if probe.shape[-1] != retriever.dim:
        raise RuntimeError(
            f"model embedding dim {probe.shape[-1]} does not match index dim {retriever.dim}; "
            "the served model must be the one the index was built with"
        )

    return {
        "cfg": cfg,
        "device": device,
        "model": model,
        "retriever": retriever,
        "tags": list(ckpt["tags"]),
        "top_k": min(TOP_K, retriever.index.ntotal),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    _app_state.update(_load_models())
    yield
    _app_state.clear()


app = FastAPI(title="audio-tag-retrieval", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    if request.url.path in COUNTED_PATHS:
        start = time.perf_counter()
        response = await call_next(request)
        metrics.observe(request.url.path, time.perf_counter() - start, response.status_code >= 400)
    else:
        response = await call_next(request)
    return response


# ---------------------------------------------------------------- endpoints


async def _read_and_validate(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415, detail=f"unsupported content type: {file.content_type}"
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
    return data


def _add_stage_headers(response, stages: dict[str, float]) -> None:
    for name, value in stages.items():
        response.headers[f"X-Stage-{name.replace('_ms', '').title()}-Ms"] = f"{value:.2f}"


async def _analyze(data: bytes, *, need_neighbors: bool):
    async with _inference_slots:
        try:
            return await run_in_threadpool(analyze_audio, data, need_neighbors=need_neighbors)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"could not process audio: {exc}") from exc


@app.post("/tag", response_model=TagResponse)
async def tag(response: Response, file: Annotated[UploadFile, File()]) -> TagResponse:
    data = await _read_and_validate(file)
    tags, _, stages = await _analyze(data, need_neighbors=False)
    _add_stage_headers(response, stages)
    return TagResponse(tags=tags)


@app.post("/similar", response_model=SimilarResponse)
async def similar(response: Response, file: Annotated[UploadFile, File()]) -> SimilarResponse:
    data = await _read_and_validate(file)
    _, neighbors, stages = await _analyze(data, need_neighbors=True)
    _add_stage_headers(response, stages)
    return SimilarResponse(neighbors=neighbors or [])


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(response: Response, file: Annotated[UploadFile, File()]) -> AnalyzeResponse:
    data = await _read_and_validate(file)
    tags, neighbors, stages = await _analyze(data, need_neighbors=True)
    _add_stage_headers(response, stages)
    return AnalyzeResponse(tags=tags, neighbors=neighbors or [])


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(
        model=MODEL_NAME,
        index_type=INDEX_TYPE,
        index_vectors=_app_state["retriever"].index.ntotal,
        device=str(_app_state["device"]),
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    return metrics.render()

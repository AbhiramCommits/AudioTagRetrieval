# Deployment guide

## Container resource sizing

Measured with Docker Desktop on an Apple Silicon Mac (VM: 8 vCPU, 8 GB RAM),
image built from `Dockerfile`, 2 uvicorn workers, flat FAISS index of 3,970
1024-dim vectors, CNN tagger:

| resource | measured |
|---|---|
| image size | 3.35 GB on disk (`docker images`; CPU torch + ffmpeg + onnxruntime + torchcodec) |
| idle container RSS | ~0.98 GB (`docker stats`, 2 workers, after warmup) |
| CPU per `/analyze` request | ~0.3 s of CPU time at low load (forward dominates) |
| cold start (start -> healthy `/healthz`) | **14 s** (polled at 1 s intervals) |

Sizing guidance: give each replica 1 vCPU + 1.5 GB RAM minimum; the memory
footprint is dominated by two torch processes (one per uvicorn worker), so a
single worker cuts idle RSS roughly in half. CPU-bound throughput scales with
cores until the GIL-serialized decode stage becomes the limit (see the
serving benchmark in `reports/serving_benchmark.md`).

## Horizontal scaling

The API is stateless: every replica loads the checkpoint + FAISS index from
its local `artifacts/` volume at startup and serves any request.

- **Index replicated per replica (current design).** The flat index is 16 MB
  (IVFPQ: 2.3 MB) and copies are loaded read-only into each process, so
  replication is cheap and there is no extra network hop. This is the right
  choice until the corpus is large enough that memory per replica matters.
- **Shared index service.** For very large corpora (or frequent index
  updates) the FAISS index can move to a dedicated sidecar (e.g. a small
  gRPC service or a managed vector DB) that every API replica queries. The
  tradeoff: one extra network hop per query and a second service to operate,
  in exchange for loading the index once and updating it without restarting
  API replicas.

## Cold start

Measured on this machine: **14 s** from `docker compose up -d` to the first
successful `/healthz` (container create + 2 workers each importing torch,
loading the checkpoint, and opening the index). Expect this to grow with
model size and worker count; set the Docker healthcheck `start_period`
accordingly (currently 30 s).

## Rebuilding the index when new tracks arrive

1. Add the new mp3 files under `data/raw/<dir>/` and add their rows to
   `annotations_final.csv` (tab-separated: `clip_id`, 188 tag columns,
   `mp3_path`).
2. Rebuild tags and manifests: `python -m audiotag.data.labels` (the tag
   vocabulary changes only if you want it to; a frozen top-50 can be pinned
   by passing the same `tags.json`).
3. Precompute features for the new clips:
   `python scripts/precompute_features.py --workers 4` (skips existing).
4. Re-embed train+val and rebuild both indexes:
   `python -m audiotag.index.build --index-type flat` and
   `... --index-type ivfpq` (embeddings are cached in
   `artifacts/index/embeddings.npy`; delete it to force re-embedding after a
   model retrain).
5. Restart the API replicas (`docker compose up -d --force-recreate`); each
   replica loads the new `.faiss` + `id_map.parquet` at startup.

The model checkpoint itself does not change during this runbook; if you also
retrain, re-export ONNX (`scripts/export_onnx.py`) before the roll.

## Model versioning

`/healthz` reports `checkpoint_sha256` — the first 16 hex chars of the SHA-256
of the checkpoint file a replica loaded — alongside `model`, `backend`,
`index_type`, `index_vectors`, and `device`:

```json
{"model":"cnn","index_type":"flat","index_vectors":3970,
 "device":"cpu","backend":"onnx","checkpoint_sha256":"be8dd22e5715d2ac"}
```

Rolling-deploy discipline: deploy new checkpoints as a new image/tag
(`artifacts` are volume-mounted, so use a release tag on the data volume or
bake artifacts into the image), then poll each replica's
`checkpoint_sha256` until all report the new hash before draining the old
ones. Tag ordering (the 50-tag vocabulary) is fixed at training time and is
stored inside the checkpoint; never serve a model against an `id_map` built
from a different vocabulary.

## Inference backend

`AUDIOTAG_BACKEND=torch` (default) or `onnx` selects the engine; the ONNX
path needs `artifacts/{model}/model.onnx` (from `scripts/export_onnx.py`).
Measured single-clip latency on this machine (`reports/onnx_comparison.md`):
PyTorch 13.0 ms vs ONNX Runtime 29.5 ms — torch's ARM kernels are faster here
and outputs agree to 5.7e-06, so the default is torch. ONNX is useful for
portability (no PyTorch dependency at inference time) or x86 deployments
where ORT's AVX/oneDNN paths usually win.

## Observability

Logs are one JSON object per line (uvicorn + app events), every request gets
a `request_id` (also returned in the `X-Request-Id` header) and an access
line with method/path/status/duration. `AUDIOTAG_LOG_LEVEL` (or uvicorn's
`--log-level`) controls verbosity: `info` (default), `debug`, `warning`.
Prometheus metrics on `/metrics` (requests, errors, latency histogram by
endpoint) as usual.

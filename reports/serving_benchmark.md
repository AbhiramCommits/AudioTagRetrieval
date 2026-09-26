# Serving benchmark

Load test of the containerized API (`POST /analyze` = decode + log-mel +
single model pass for logits/embedding + FAISS search), generated with
`scripts/load_test.py` against `docker compose up` (2 uvicorn workers,
flat IndexFlatIP over 3970 train+val clips, `AUDIOTAG_TORCH_THREADS=2`,
`AUDIOTAG_MAX_CONCURRENT=8` inference slots).

Each run: 10 s warmup (excluded) + 45 s measured, posting real
MagnaTagATune mp3 clips (~29 s, 16 kHz mono, ~115 KB) sampled randomly
from `data/raw`.

## Hardware

- Host: Apple Silicon MacBook (macOS), Docker Desktop
- VM: 8 vCPU, 8 GB RAM, arm64
- The VM is shared: ~20 co-tenant containers run alongside the API
  (Spark master + 2 workers, a k3s-style dev control plane, several
  Postgres instances), which injects latency jitter into every
  measurement below.

## Results

| concurrency | QPS | error rate | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|
| 1 | 5.92 | 0.00% | 146 | 319 | 609 |
| 8 | 9.48 | 0.00% | 758 | 1547 | 2088 |
| 32 | 4.55 | 0.00% | 5921 | 12531 | 16790 |

- At concurrency 1 the service processes ~5.9 req/s end to end.
- At concurrency 8 throughput peaks (~9.5 QPS) but each request slows:
  the 8 inference slots are busy and torch ops contend for 2 threads each.
- At concurrency 32 the semaphore queues excess requests (max 8 in
  flight); latency becomes queue-bound (~5.9 s p50) and throughput drops
  as the shared VM thrashes. Error rate stays 0.

## Where the time goes (per-stage means from X-Stage-* headers)

| stage | conc 1 | conc 8 | conc 32 |
|---|---|---|---|
| decode (libsndfile, in-process) | 13 ms | 59 ms | 592 ms |
| log-mel (torchaudio) | 16 ms | 111 ms | 985 ms |
| forward (CNN embed + head, torch 2 threads) | 122 ms | 582 ms | 3816 ms |
| FAISS search (flat, 3970 x 1024) | 7 ms | 24 ms | 186 ms |

- At low load the CNN forward dominates (~77% of the request), then
  decode, mel, and FAISS are roughly equal minor costs.
- Under concurrency all stages inflate: the forward contends for the
  2-thread torch pool across the 8 inference slots, the libsndfile decode
  serializes on the Python GIL, and the shared VM's co-tenants add CPU
  steal. decode is much cheaper than the original ffmpeg-subprocess
  approach (which cost 180 ms per request at concurrency 1 and collapsed
  at higher concurrency due to fork/exec of the large RSS worker
  processes).
- FAISS exact search on the 4k corpus is a minor cost at every level.

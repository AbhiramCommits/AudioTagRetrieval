# Results

Every number on this page is measured, and each traces to a committed
artifact or report (source in parentheses). No estimates.

## Dataset

| item | value | source |
|---|---|---|
| clips on disk (working subset) | 4,970 | 3,970 train+val index corpus + 1,000 held-out queries (`reports/index_benchmark.md`) |
| training subset | 2,000 clips | trained splits: 413 held-out test clips (`artifacts/cnn/metrics.json` `n_test`) |
| tag vocabulary | 50 tags, 0 skipped in eval | `artifacts/cnn/metrics.json` (`n_tags`, `n_skipped_tags`) |

## Training (hardware: Apple Silicon MacBook, Docker-free host runs)

| model | epochs | wall time | best val mAP | source |
|---|---|---|---|---|
| CNN (ShortChunkCNN) | 17 (early stop, patience 5) | 19.1 min | 0.3934 | `artifacts/cnn/metrics.jsonl` (sum of `elapsed_sec`), `metrics.json` |
| Transformer | 29 (early stop) | 26.1 min | 0.3883 | `artifacts/transformer/metrics.jsonl` |

Training ran on the Apple Silicon host (device auto -> MPS); hardware of the
machine is documented in `reports/serving_benchmark.md`.

## Model size

| model | parameters |
|---|---|
| CNN (ShortChunkCNN, 50 tags) | 4,700,402 |
| Transformer (d_model=256, 4 layers, 50 tags) | 3,304,242 |

Measured with
`python -c "from audiotag.models import build_model; print(sum(p.numel() for p in build_model('X', 50).parameters()))"`
against the committed model definitions.

## Tagging quality (2,000-clip training subset, 413 test clips)

| model | macro AUC | macro mAP | micro mAP | baseline macro mAP (prior-only) | lift |
|---|---|---|---|---|---|
| CNN | 0.8480 | 0.2683 | 0.2219 | 0.0515 | +0.2168 |
| Transformer | 0.7643 | 0.2028 | 0.1887 | 0.0515 | +0.1513 |

(`artifacts/cnn/metrics.json`, `artifacts/transformer/metrics.json`.)

## Retrieval quality (Jaccard overlap of neighbor tags vs random)

| k | model | random baseline |
|---|---|---|
| 1 | 0.1527 | 0.0365 |
| 5 | 0.1499 | 0.0363 |
| 10 | 0.1492 | 0.0366 |

(`reports/retrieval_quality.md`.)

## FAISS index (corpus 3,970, 1,000 queries, k=10)

| index | size | mean query latency | recall@10 vs flat |
|---|---|---|---|
| flat (IndexFlatIP) | 16.26 MB | 0.01 ms | 1.0000 (reference) |
| ivfpq (nlist=256, m=32, nbits=8, nprobe=16) | 2.26 MB | 0.26 ms | 0.7716 |

(`reports/index_benchmark.md`.)

## Serving (containerized API, Docker VM 8 vCPU / 8 GB, `/analyze`)

| concurrency | QPS | error rate | p50 | p95 |
|---|---|---|---|---|
| 1 | 5.92 | 0% | 146 ms | 319 ms |
| 8 | 9.48 | 0% | 758 ms | 1,547 ms |
| 32 | 4.55 | 0% | 5,921 ms | 12,531 ms |

Per-stage means at concurrency 1: decode 13 ms, log-mel 16 ms, forward
122 ms, FAISS 7 ms. (`reports/serving_benchmark.md`.)

## ONNX vs PyTorch (single clip, CPU)

| engine | mean latency |
|---|---|
| PyTorch (inference_mode, 2 threads) | 13.54 ms |
| ONNX Runtime (CPU provider) | 15.33 ms |

Max output difference across batch sizes 1-3 and 150-313 frames:
8.34e-06 (< 1e-4). (`reports/onnx_comparison.md`.)

## Operations

| item | value | source |
|---|---|---|
| container cold start to healthy | 14 s | `docs/DEPLOYMENT.md` (polled /healthz at 1 s) |
| idle container RSS (2 workers) | 0.98 GB | `docs/DEPLOYMENT.md` (`docker stats`) |

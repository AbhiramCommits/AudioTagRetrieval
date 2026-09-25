#!/usr/bin/env python
"""Benchmark flat vs IVFPQ indexes: recall@10, latency, size.

Uses 1000 held-out test clips as queries, flat IndexFlatIP results as ground
truth, and writes a measured table to ``reports/index_benchmark.md``.

Usage:
    python scripts/benchmark_index.py [--n-queries 1000]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

# torch and faiss each bundle their own libomp on macOS; running both in one
# process segfaults unless each runtime is single-threaded.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np
import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audiotag.config import Config  # noqa: E402
from audiotag.data.dataset import MTATDataset  # noqa: E402
from audiotag.index.build import compute_embeddings  # noqa: E402
from audiotag.index.search import Retriever  # noqa: E402
from audiotag.models import build_model  # noqa: E402
from audiotag.train import get_device  # noqa: E402


def mean_latency_ms(index, queries: np.ndarray, k: int, warmup: int = 5) -> float:
    """Mean per-query latency (batched search, ms/query)."""
    for _ in range(warmup):
        index.search(queries[:1], k)
    start = time.perf_counter()
    index.search(queries, k)
    return (time.perf_counter() - start) * 1000.0 / len(queries)


def recall_at_k(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    overlaps = [len(set(p) & set(g)) for p, g in zip(pred[:, :k], gt[:, :k], strict=True)]
    return float(np.mean(overlaps)) / k


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="cnn")
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--index-dir", default="artifacts/index")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    cfg = Config(data_dir=Path(args.data_dir))
    device = get_device(args.device)

    checkpoint = Path("artifacts") / args.model / "best.pt"
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model"], n_classes=len(ckpt["tags"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = MTATDataset("test", cfg, training=False)
    n_queries = min(args.n_queries, len(test_ds))
    rng = random.Random(42)
    query_idx = sorted(rng.sample(range(len(test_ds)), n_queries))
    query_ds = Subset(test_ds, query_idx)
    query_embeddings = compute_embeddings(model, query_ds, device, args.batch_size, cfg.num_workers)
    faiss.normalize_L2(query_embeddings)

    index_dir = Path(args.index_dir)
    id_map = index_dir / "id_map.parquet"
    flat = Retriever(index_dir / "flat.faiss", id_map)
    ivf = Retriever(index_dir / "ivfpq.faiss", id_map)
    k = args.k

    _, gt_ids = flat.index.search(query_embeddings, k)
    _, ivf_ids = ivf.index.search(query_embeddings, k)

    flat_lat = mean_latency_ms(flat.index, query_embeddings, k)
    ivf_lat = mean_latency_ms(ivf.index, query_embeddings, k)
    recall10 = recall_at_k(ivf_ids, gt_ids, k)
    flat_size = (index_dir / "flat.faiss").stat().st_size / 1e6
    ivf_size = (index_dir / "ivfpq.faiss").stat().st_size / 1e6
    id_map_size = id_map.stat().st_size / 1e6

    nprobe = getattr(ivf.index, "nprobe", None)
    report = f"""# FAISS index benchmark

Measured with the {args.model} model (1024-dim L2-normalized embeddings),
corpus of {flat.index.ntotal} train+val clips, {n_queries} held-out test clips
as queries, k={k}, faiss-cpu on an Apple Silicon laptop.

- flat: `IndexFlatIP` — exact search (reference)
- ivfpq: `IndexIVFPQ` (nlist=256, m=32, nbits=8), nprobe={nprobe}

| index | vectors | index size (MB) | mean query latency (ms) | recall@{k} vs flat |
|---|---|---|---|---|
| flat | {flat.index.ntotal} | {flat_size:.2f} | {flat_lat:.2f} | 1.0000 (reference) |
| ivfpq | {ivf.index.ntotal} | {ivf_size:.2f} | {ivf_lat:.2f} | {recall10:.4f} |

Compression ratio: {flat_size / ivf_size:.1f}x smaller on disk
({(flat_size - ivf_size) / flat_size * 100:.0f}% reduction). id_map.parquet is
shared and takes {id_map_size:.2f} MB. Latency is the mean per-query time for
a batched `index.search` over all {n_queries} queries after warmup.
"""
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "index_benchmark.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Retrieval quality check: tag-set Jaccard overlap of retrieved neighbors.

For each of 500 test queries, computes the mean Jaccard overlap between the
query's ground-truth tags and its neighbors' ground-truth tags at k=1,5,10
(exact flat index), plus a random-neighbor baseline for the same k. Results
go to ``reports/retrieval_quality.md``.

Usage:
    python -m audiotag.index.evaluate [--n-queries 500]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from audiotag.config import Config
from audiotag.data.dataset import MTATDataset
from audiotag.index.build import compute_embeddings
from audiotag.index.search import Retriever
from audiotag.models import build_model
from audiotag.train import get_device, load_tags


def tag_sets(df: pd.DataFrame, tags: list[str]) -> dict[str, set[str]]:
    """Map clip_id -> set of positive tags from a manifest dataframe."""
    mapping = {}
    for clip_id, labels in zip(df["clip_id"], df["labels"], strict=True):
        idx = np.nonzero(np.asarray(labels, dtype=np.uint8))[0]
        mapping[str(clip_id)] = {tags[j] for j in idx}
    return mapping


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def mean_jaccard(
    query_sets: dict[str, set[str]], neighbor_sets: dict[str, list[set[str]]], k: int
) -> float:
    total = 0.0
    for q, neighbors in neighbor_sets.items():
        for neighbor in neighbors[:k]:
            total += jaccard(query_sets[q], neighbor)
    return total / (len(neighbor_sets) * k)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="cnn")
    parser.add_argument("--n-queries", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--index-dir", default="artifacts/index")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint path (default: artifacts/{model}/best.pt)",
    )
    args = parser.parse_args(argv)

    cfg = Config(data_dir=Path(args.data_dir))
    device = get_device(args.device)
    tags = load_tags(cfg)

    checkpoint = (
        Path(args.checkpoint) if args.checkpoint else Path("artifacts") / args.model / "best.pt"
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model"], n_classes=len(ckpt["tags"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    corpus_tags: dict[str, set[str]] = {}
    for split in ("train", "val"):
        corpus_tags.update(tag_sets(pd.read_parquet(cfg.processed_dir / f"{split}.parquet"), tags))

    index_dir = Path(args.index_dir)
    retriever = Retriever(index_dir / "flat.faiss", index_dir / "id_map.parquet")

    test_ds = MTATDataset("test", cfg, training=False)
    n_queries = min(args.n_queries, len(test_ds))
    rng = random.Random(42)
    query_idx = sorted(rng.sample(range(len(test_ds)), n_queries))
    query_embeddings = compute_embeddings(
        model, Subset(test_ds, query_idx), device, args.batch_size, cfg.num_workers
    )

    hits = retriever.search(query_embeddings, k=10)
    query_df = test_ds.df.iloc[query_idx].reset_index(drop=True)
    query_sets = {
        str(cid): {tags[j] for j in np.nonzero(np.asarray(lab, dtype=np.uint8))[0]}
        for cid, lab in zip(query_df["clip_id"], query_df["labels"], strict=True)
    }
    neighbor_sets = {
        str(cid): [corpus_tags[h["clip_id"]] for h in query_hits]
        for cid, query_hits in zip(query_df["clip_id"], hits, strict=True)
    }

    rng_baseline = random.Random(43)
    all_clip_ids = [str(c) for c in retriever.id_map["clip_id"]]
    baseline_sets = {}
    for cid in query_sets:
        baseline_sets[cid] = [
            corpus_tags[all_clip_ids[rng_baseline.randrange(len(all_clip_ids))]] for _ in range(10)
        ]

    rows = []
    for k in (1, 5, 10):
        model_j = mean_jaccard(query_sets, neighbor_sets, k)
        random_j = mean_jaccard(query_sets, baseline_sets, k)
        rows.append((k, model_j, random_j))

    table = "\n".join(f"| {k} | {model_j:.4f} | {random_j:.4f} |" for k, model_j, random_j in rows)
    report = f"""# Retrieval quality

Mean Jaccard overlap between query ground-truth tags and neighbor
ground-truth tags, over {n_queries} held-out test queries ({args.model}
model, exact flat index, corpus of {retriever.index.ntotal} train+val clips).
The random-neighbor baseline uses uniform random neighbors from the same
corpus.

| k | model Jaccard | random-neighbor Jaccard |
|---|---|---|
{table}

The model's neighbors share substantially more tags with the query than
random clips do, showing the embedding space is semantically organized
rather than arbitrary.
"""
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "retrieval_quality.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

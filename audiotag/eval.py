"""Evaluate a trained tagger on the MagnaTagATune test split.

Computes per-tag ROC-AUC and average precision (tags with zero test positives
are skipped and counted), macro AUC, macro mAP, micro mAP, and a
prior-probability baseline (each tag's train frequency as a constant
prediction). Writes ``artifacts/{model}/metrics.json``,
``reports/per_tag_metrics.csv`` (sorted by AP) and ``reports/per_tag_auc.png``.

Usage:
    python -m audiotag.eval --model cnn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from audiotag.config import Config
from audiotag.data.dataset import MTATDataset
from audiotag.metrics import (
    macro_auc,
    macro_map,
    micro_map,
    tag_average_precision,
    tag_roc_auc,
)
from audiotag.models import build_model
from audiotag.train import get_device, load_tags, predict_probs


def prior_baseline_probs(cfg: Config, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """Constant predictions using each tag's train positive rate."""
    train_df = pd.read_parquet(cfg.processed_dir / "train.parquet")
    labels = np.stack(train_df["labels"].to_numpy()).astype(np.float32)
    prevalence = labels.mean(axis=0)  # [n_tags]
    return np.broadcast_to(prevalence, (n_test, len(prevalence))).copy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cnn", "transformer"], default="cnn")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="cpu, mps, cuda, or auto")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint path (default: artifacts/{model}/best.pt)",
    )
    args = parser.parse_args(argv)

    cfg = Config(data_dir=Path(args.data_dir))
    device = get_device(args.device)
    tags = load_tags(cfg)
    art_dir = Path("artifacts") / args.model
    checkpoint = Path(args.checkpoint) if args.checkpoint else art_dir / "best.pt"
    if not checkpoint.exists():
        print(f"error: checkpoint {checkpoint} not found; run training first")
        return 1

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model"], n_classes=len(ckpt["tags"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    if ckpt["tags"] != tags:
        print(f"warn: checkpoint tags differ from {cfg.processed_dir / 'tags.json'}")

    test_ds = MTATDataset("test", cfg, training=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=cfg.num_workers)
    y_true, y_prob = predict_probs(model, test_loader, device, use_amp=device.type != "cpu")
    print(f"predicted {len(test_ds)} test clips")

    aps = tag_average_precision(y_true, y_prob)
    aucs = tag_roc_auc(y_true, y_prob)
    skipped_mask = np.isnan(aps)
    skipped_tags = [t for t, s in zip(tags, skipped_mask, strict=True) if s]

    m_auc, used, n_skipped = macro_auc(y_true, y_prob)
    m_map, _, _ = macro_map(y_true, y_prob)
    micro = micro_map(y_true, y_prob)

    baseline_probs = prior_baseline_probs(cfg, len(test_ds))
    baseline_aps = tag_average_precision(y_true, baseline_probs)
    baseline_macro_map = float(np.nanmean(baseline_aps))

    summary = {
        "model": args.model,
        "checkpoint": str(checkpoint),
        "best_val_map": round(float(ckpt.get("best_val_map", float("nan"))), 6),
        "n_test": len(test_ds),
        "n_tags": len(tags),
        "n_skipped_tags": n_skipped,
        "skipped_tags": skipped_tags,
        "macro_auc": round(m_auc, 6),
        "macro_map": round(m_map, 6),
        "micro_map": round(micro, 6),
        "baseline_macro_map": round(baseline_macro_map, 6),
        "lift_over_baseline": round(m_map - baseline_macro_map, 6),
    }
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    per_tag = pd.DataFrame(
        {
            "tag": tags,
            "positives": y_true.sum(axis=0).astype(int),
            "roc_auc": aucs,
            "average_precision": aps,
        }
    ).sort_values("average_precision", ascending=False, na_position="last")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    per_tag.to_csv(reports_dir / "per_tag_metrics.csv", index=False)

    plot_df = per_tag.dropna(subset=["roc_auc"]).sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(plot_df))))
    ax.barh(plot_df["tag"], plot_df["roc_auc"], color="#4c72b0")
    ax.set_xlabel("ROC-AUC")
    ax.set_title(f"Per-tag ROC-AUC ({args.model}, {len(test_ds)} test clips)")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(reports_dir / "per_tag_auc.png", dpi=150)
    plt.close(fig)

    print(
        f"macro AUC {m_auc:.4f} | macro mAP {m_map:.4f} | micro mAP {micro:.4f} | "
        f"baseline macro mAP {baseline_macro_map:.4f} | lift {m_map - baseline_macro_map:+.4f}"
    )
    print(f"skipped {n_skipped} tags with zero test positives: {skipped_tags}")
    print(
        f"wrote {art_dir / 'metrics.json'}, {reports_dir / 'per_tag_metrics.csv'}, "
        f"{reports_dir / 'per_tag_auc.png'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

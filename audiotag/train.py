"""Train a multi-label audio tagger on MagnaTagATune.

Uses BCEWithLogitsLoss with per-tag pos_weight derived from train frequency,
AdamW with a cosine LR schedule and linear warmup, mixed precision via
torch.amp (CUDA/MPS), gradient clipping at 1.0, and early stopping on
validation mAP. Checkpoints go to ``artifacts/{model}/best.pt``; per-epoch
metrics are appended to ``artifacts/{model}/metrics.jsonl`` and logged to
TensorBoard.

Usage:
    python -m audiotag.train --model cnn --subset 2000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from audiotag.config import Config
from audiotag.data.dataset import MTATDataset
from audiotag.metrics import macro_map
from audiotag.models import build_model


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_tags(cfg: Config) -> list[str]:
    return json.loads((cfg.processed_dir / "tags.json").read_text())["tags"]


def compute_pos_weight(labels: np.ndarray, max_weight: float = 50.0) -> torch.Tensor:
    """pos_weight = negatives / positives per tag, clamped to [1, max_weight]."""
    n = len(labels)
    n_pos = labels.sum(axis=0)
    weights = (n - n_pos) / np.maximum(n_pos, 1)
    return torch.from_numpy(np.clip(weights, 1.0, max_weight).astype(np.float32))


@torch.no_grad()
def predict_probs(
    model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob) as float32 numpy arrays over the whole loader."""
    model.eval()
    trues, probs = [], []
    for x, y in loader:
        x = x.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
        trues.append(y.numpy())
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(trues), np.concatenate(probs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cnn", "transformer"], default="cnn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subset", type=int, default=None, help="limit train split to N clips")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto", help="cpu, mps, cuda, or auto")
    parser.add_argument("--patience", type=int, default=5, help="early-stopping patience")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    cfg = Config(data_dir=Path(args.data_dir))
    set_seed(cfg.seed)
    device = get_device(args.device)
    tags = load_tags(cfg)
    n_classes = len(tags)

    train_ds = MTATDataset("train", cfg, training=True, limit=args.subset)
    val_ds = MTATDataset("val", cfg, training=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=cfg.num_workers
    )
    print(
        f"device={device} train={len(train_ds)} val={len(val_ds)} "
        f"tags={n_classes} model={args.model}"
    )

    model = build_model(args.model, n_classes=n_classes).to(device)
    train_labels = np.stack(train_ds.df["labels"].to_numpy())
    pos_weight = compute_pos_weight(train_labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = args.amp and device.type in ("cuda", "mps")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"amp={'bf16' if use_amp else 'off'}")

    art_dir = Path("artifacts") / args.model
    art_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(art_dir / "tb")
    jsonl_path = art_dir / "metrics.jsonl"
    with open(jsonl_path, "w") as jsonl:
        best_map, best_epoch = -1.0, -1
        patience_left = args.patience
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            running_loss = 0.0
            for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running_loss += loss.item() * x.size(0)
            train_loss = running_loss / (len(train_loader) * args.batch_size)

            y_val, p_val = predict_probs(model, val_loader, device, use_amp)
            val_map, used, skipped = macro_map(y_val, p_val)
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_map": round(val_map, 6),
                "val_tags_used": used,
                "val_tags_skipped": skipped,
                "lr": lr_now,
                "elapsed_sec": round(elapsed, 1),
            }
            jsonl.write(json.dumps(record) + "\n")
            jsonl.flush()
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/mAP", val_map, epoch)
            writer.add_scalar("train/lr", lr_now, epoch)
            print(
                f"epoch {epoch:3d} | loss {train_loss:.4f} | val mAP {val_map:.4f} "
                f"| lr {lr_now:.2e} | {elapsed:.0f}s"
            )

            if val_map > best_map:
                best_map, best_epoch = val_map, epoch
                patience_left = args.patience
                torch.save(
                    {
                        "model": args.model,
                        "model_state_dict": model.state_dict(),
                        "tags": tags,
                        "best_val_map": best_map,
                        "epoch": epoch,
                        "args": vars(args),
                    },
                    art_dir / "best.pt",
                )
            else:
                patience_left -= 1
                if patience_left == 0:
                    print(f"early stopping after {epoch} epochs (patience {args.patience})")
                    break

    writer.close()
    print(f"best val mAP {best_map:.4f} at epoch {best_epoch} -> {art_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

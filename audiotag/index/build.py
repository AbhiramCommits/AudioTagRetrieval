"""Build a FAISS index over train+val clip embeddings (offline step).

Runs ``model.embed()`` over every train+val clip in eval mode (center crop),
L2-normalizes the embeddings, and builds either an exact ``IndexFlatIP`` or a
compressed ``IndexIVFPQ`` (nlist=256, m=32, nbits=8, trained on the corpus).
Persists the index to ``artifacts/index/{type}.faiss`` and the row mapping to
``artifacts/index/id_map.parquet`` (clip_id, mp3_path, top-5 ground-truth tags).

Usage:
    python -m audiotag.index.build --index-type flat
    python -m audiotag.index.build --index-type ivfpq
"""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from audiotag.config import Config
from audiotag.data.dataset import MTATDataset
from audiotag.models import build_model
from audiotag.train import get_device, load_tags


def compute_embeddings(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> np.ndarray:
    """Embed every clip in ``dataset`` (eval mode) -> float32 [N, D]."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    chunks = []
    model.eval()
    with torch.no_grad():
        for x, _ in tqdm(loader, desc=f"embed ({len(dataset)} clips)", leave=False):
            chunks.append(model.embed(x.to(device)).float().cpu().numpy())
    embeddings = np.concatenate(chunks)
    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            "non-finite embeddings produced; retry with --device cpu if using an unstable backend"
        )
    return embeddings


def load_or_compute_embeddings(
    model: torch.nn.Module,
    cfg: Config,
    device: torch.device,
    batch_size: int,
    cache_dir: Path,
    checkpoint: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Embed train+val clips, reusing cached embeddings when fresher than the checkpoint."""
    emb_path = cache_dir / "embeddings.npy"
    meta_path = cache_dir / "embed_meta.parquet"
    if (
        emb_path.exists()
        and meta_path.exists()
        and checkpoint.stat().st_mtime <= emb_path.stat().st_mtime
    ):
        print(f"reusing cached embeddings from {emb_path}")
        return np.load(emb_path), pd.read_parquet(meta_path)

    chunks, metas = [], []
    for split in ("train", "val"):
        ds = MTATDataset(split, cfg, training=False)
        chunks.append(compute_embeddings(model, ds, device, batch_size, cfg.num_workers))
        metas.append(ds.df[["clip_id", "mp3_path", "labels"]])
    embeddings = np.concatenate(chunks).astype(np.float32)
    meta = pd.concat(metas, ignore_index=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    meta.to_parquet(meta_path, index=False)
    return embeddings, meta


def build_index(embeddings: np.ndarray, index_type: str, *, nlist: int, m: int, nbits: int):
    """Create and populate a FAISS index over L2-normalized embeddings."""
    d = embeddings.shape[1]
    if index_type == "flat":
        index = faiss.IndexFlatIP(d)
    elif index_type == "ivfpq":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        index.train(embeddings)
    else:
        raise ValueError(f"unknown index type {index_type!r}")
    index.add(embeddings)
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cnn", "transformer"], default="cnn")
    parser.add_argument("--checkpoint", default=None, help="default: artifacts/{model}/best.pt")
    parser.add_argument("--index-type", choices=["flat", "ivfpq"], default="flat")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="artifacts/index")
    parser.add_argument("--nlist", type=int, default=256)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--nbits", type=int, default=8)
    args = parser.parse_args(argv)

    cfg = Config(data_dir=Path(args.data_dir))
    device = get_device(args.device)
    tags = load_tags(cfg)

    checkpoint = (
        Path(args.checkpoint) if args.checkpoint else Path("artifacts") / args.model / "best.pt"
    )
    if not checkpoint.exists():
        print(f"error: checkpoint {checkpoint} not found; run training first")
        return 1
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model"], n_classes=len(ckpt["tags"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings, meta = load_or_compute_embeddings(
        model, cfg, device, args.batch_size, out_dir, checkpoint
    )
    assert len(embeddings) == len(meta)
    print(f"embedded {len(embeddings)} clips, dim {embeddings.shape[1]}")
    faiss.normalize_L2(embeddings)

    index = build_index(embeddings, args.index_type, nlist=args.nlist, m=args.m, nbits=args.nbits)

    faiss.write_index(index, str(out_dir / f"{args.index_type}.faiss"))

    top_tags = [
        [tags[j] for j in np.nonzero(np.asarray(row, dtype=np.uint8))[0][:5]]
        for row in meta["labels"]
    ]
    id_map = pd.DataFrame(
        {
            "clip_id": meta["clip_id"].tolist(),
            "mp3_path": meta["mp3_path"].tolist(),
            "top_tags": top_tags,
        }
    )
    id_map.to_parquet(
        out_dir / "id_map.parquet",
        engine="pyarrow",
        index=False,
    )
    index_path = out_dir / f"{args.index_type}.faiss"
    print(
        f"wrote {index_path} ({index.ntotal} vectors, "
        f"{index_path.stat().st_size / 1e6:.2f} MB) and {out_dir / 'id_map.parquet'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

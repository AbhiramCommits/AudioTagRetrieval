#!/usr/bin/env python
"""Precompute log-mel features for every clip into ``data/processed/mels/``.

Writes one ``.npy`` per clip (``<rel_path_without_ext>.npy``) plus a
memory-mapped feature matrix ``data/processed/features.dat`` and its metadata
``data/processed/index.json``.

Usage:
    python scripts/precompute_features.py --workers 4
    python scripts/precompute_features.py --subset 200
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audiotag.config import Config  # noqa: E402
from audiotag.features.melspec import log_mel  # noqa: E402

MEMORY_MAP_NAME = "features.dat"
INDEX_NAME = "index.json"


def process_one(job: tuple) -> tuple[str, int | None]:
    """Extract features for one clip; returns (rel_stem, n_frames) or (rel_stem, None) on error."""
    src, dst, rel, target_sr, n_fft, win_length, hop_length, n_mels = job
    if Path(dst).exists():
        return rel, None
    try:
        torch.set_num_threads(1)
        waveform, sr = torchaudio.load(src)
        mel = log_mel(
            waveform,
            sr,
            target_sr=target_sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, mel.numpy())
        return rel, mel.shape[-1]
    except Exception as exc:  # noqa: BLE001
        print(f"error: failed to process {src}: {exc}", file=sys.stderr)
        return rel, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="base data directory (default: data)")
    parser.add_argument(
        "--workers", type=int, default=4, help="multiprocessing workers (0 = serial)"
    )
    parser.add_argument("--subset", type=int, default=None, help="process only the first N clips")
    parser.add_argument("--force", action="store_true", help="recompute existing features")
    parser.add_argument("--no-index", action="store_true", help="skip memory-mapped index build")
    args = parser.parse_args(argv)

    cfg = Config(data_dir=Path(args.data_dir))
    cfg.ensure_dirs()

    mp3s = sorted(cfg.raw_dir.rglob("*.mp3"))
    if args.subset:
        mp3s = mp3s[: args.subset]
    if not mp3s:
        print(f"error: no mp3s under {cfg.raw_dir}; run 'python -m audiotag.data.download' first")
        return 1

    params = (cfg.sample_rate, cfg.n_fft, cfg.win_length, cfg.hop_length, cfg.n_mels)
    jobs = []
    for mp3 in mp3s:
        rel = str(mp3.relative_to(cfg.raw_dir).with_suffix(""))
        dst = cfg.mels_dir / f"{rel}.npy"
        if args.force and dst.exists():
            dst.unlink()
        jobs.append((str(mp3), str(dst), rel, *params))

    if args.workers <= 1:
        results = list(tqdm(map(process_one, jobs), total=len(jobs), desc="features"))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(tqdm(pool.map(process_one, jobs), total=len(jobs), desc="features"))

    if not args.no_index:
        entries = [(rel, n) for rel, n in results if n is not None]
        if entries:
            max_t = max(n for _, n in entries)
            memmap_path = cfg.processed_dir / MEMORY_MAP_NAME
            mmap = np.lib.format.open_memmap(
                memmap_path, mode="w+", dtype=np.float32, shape=(len(entries), cfg.n_mels, max_t)
            )
            for i, (rel, n) in enumerate(tqdm(entries, desc="index")):
                mel = np.load(cfg.mels_dir / f"{rel}.npy")  # [1, n_mels, T]
                mmap[i, :, :n] = mel[0]
            mmap.flush()
            del mmap
            index = {
                "memmap_file": MEMORY_MAP_NAME,
                "dtype": "float32",
                "shape": [len(entries), cfg.n_mels, max_t],
                "clip_ids": [rel for rel, _ in entries],
                "lengths": [n for _, n in entries],
            }
            with open(cfg.processed_dir / INDEX_NAME, "w") as fh:
                json.dump(index, fh, indent=2)
            print(f"wrote {memmap_path} ({len(entries)} x {cfg.n_mels} x {max_t})")

    n_ok = sum(1 for _, n in results if n is not None)
    print(f"done: {n_ok}/{len(jobs)} clips processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

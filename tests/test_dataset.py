"""Tests for MTATDataset with synthetic manifests and mel files."""

import numpy as np
import torch

from audiotag.config import Config
from audiotag.data.dataset import MTATDataset
from tests.helpers import write_manifest


def _make_ds(tmp_path, lengths=(350, 100), split="train"):
    processed = tmp_path / "data" / "processed"
    mels = processed / "mels"
    processed.mkdir(parents=True)
    rows = []
    for i, length in enumerate(lengths):
        cid = f"s{i}/clip_{i}"
        path = mels / f"{cid}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.arange(1 * 128 * length, dtype=np.float32).reshape(1, 128, length))
        rows.append(
            {
                "clip_id": cid,
                "mp3_path": cid + ".mp3",
                "labels": np.array([1, 0, 1, 0, 1], dtype=np.uint8),
            }
        )
    write_manifest(processed / f"{split}.parquet", rows)
    return Config(data_dir=tmp_path / "data")


def test_train_dataset_random_crop_shape(tmp_path):
    cfg = _make_ds(tmp_path)
    ds = MTATDataset("train", cfg, training=True)
    assert len(ds) == 2
    mel, labels = ds[0]
    assert mel.shape == (1, 128, cfg.n_frames)
    assert labels.numel() == 5
    assert mel.dtype == torch.float32


def test_eval_dataset_center_crop_is_deterministic(tmp_path):
    cfg = _make_ds(tmp_path)
    ds = MTATDataset("train", cfg, training=False)
    source = np.load(ds.npy_path_for(cfg.mels_dir, str(ds.df.iloc[0]["mp3_path"])))
    start = (350 - cfg.n_frames) // 2
    mel, _ = ds[0]
    assert torch.equal(mel, torch.from_numpy(source[..., start : start + cfg.n_frames]))


def test_short_clip_is_zero_padded(tmp_path):
    cfg = _make_ds(tmp_path, lengths=(100,))
    ds = MTATDataset("train", cfg, training=False)
    mel, _ = ds[0]
    assert mel.shape == (1, 128, cfg.n_frames)
    expected = torch.arange(12800, dtype=torch.float32).reshape(1, 128, 100)
    assert torch.equal(mel[..., :100], expected)
    assert (mel[..., 100:] == 0).all()


def test_limit_subsets_rows(tmp_path):
    cfg = _make_ds(tmp_path)
    ds = MTATDataset("train", cfg, training=False, limit=1)
    assert len(ds) == 1

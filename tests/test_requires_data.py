"""Tests that need the real MagnaTagATune dataset (deselected in CI)."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_data


@pytest.mark.requires_data
def test_real_dataset_manifests_exist():
    data = Path("data")
    if not (data / "processed" / "train.parquet").exists():
        pytest.skip("dataset not present; run 'make data SUBSET=2000' first")
    for split in ("train", "val", "test"):
        assert (data / "processed" / f"{split}.parquet").exists()
    assert (data / "processed" / "tags.json").exists()


@pytest.mark.requires_data
def test_real_dataset_loads_first_clip():
    from audiotag.config import Config
    from audiotag.data.dataset import MTATDataset

    cfg = Config()
    if not (cfg.processed_dir / "train.parquet").exists():
        pytest.skip("dataset not present; run 'make data SUBSET=2000' first")
    ds = MTATDataset("train", cfg, training=False, limit=4)
    mel, labels = ds[0]
    assert mel.shape == (1, cfg.n_mels, cfg.n_frames)
    assert labels.numel() == 50

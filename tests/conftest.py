"""Shared fixtures built from tests.helpers (no binary audio committed)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.helpers import TAGS, FakeRetriever, TinyEmbedModel, make_wav_bytes, write_manifest


@pytest.fixture
def wav_bytes() -> bytes:
    return make_wav_bytes()


@pytest.fixture
def stereo_wav_bytes() -> bytes:
    return make_wav_bytes(channels=2)


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Tiny on-disk dataset: tags.json, split manifests, and mel npy files."""
    data_dir = tmp_path / "data"
    processed = data_dir / "processed"
    mels_dir = processed / "mels"
    processed.mkdir(parents=True)
    (processed / "tags.json").write_text(
        json.dumps({"tags": TAGS, "positive_counts": {t: 2 for t in TAGS}})
    )
    rng = np.random.RandomState(0)
    for split, n in (("train", 4), ("val", 2), ("test", 3)):
        rows = []
        for i in range(n):
            cid = f"x{split[0]}{i}/clip_{i}"
            mel_path = mels_dir / f"{cid}.npy"
            mel_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(mel_path, rng.randn(1, 128, 350).astype(np.float32))
            labels = rng.randint(0, 2, size=len(TAGS)).astype(np.uint8)
            rows.append({"clip_id": cid, "mp3_path": cid + ".mp3", "labels": labels})
        write_manifest(processed / f"{split}.parquet", rows)
    return data_dir


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    from audiotag.models import build_model

    model = build_model("cnn", n_classes=len(TAGS))
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model": "cnn",
            "model_state_dict": model.state_dict(),
            "tags": TAGS,
            "best_val_map": 0.5,
            "epoch": 1,
        },
        path,
    )
    return path


@pytest.fixture
def tiny_embed_model() -> TinyEmbedModel:
    return TinyEmbedModel(n_classes=5, dim=8)


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever(ntotal=5, dim=8)

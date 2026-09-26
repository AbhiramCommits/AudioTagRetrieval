"""Tests for FAISS index building, search, and the retrieval-quality eval."""

import faiss
import numpy as np
import pandas as pd
import pytest
import torch

from audiotag.config import Config
from audiotag.index.build import build_index, compute_embeddings, load_or_compute_embeddings
from audiotag.index.search import Retriever
from tests.helpers import TAGS, TinyEmbedModel

D = 32


def _write_id_map(path, clip_ids, top_tags=None):
    df = pd.DataFrame(
        {
            "clip_id": clip_ids,
            "mp3_path": [cid + ".mp3" for cid in clip_ids],
            "top_tags": top_tags or [["a", "b"]] * len(clip_ids),
        }
    )
    df.to_parquet(path, index=False)


def test_flat_self_retrieval(tmp_path):
    rng = np.random.RandomState(0)
    vectors = rng.randn(64, D).astype(np.float32)
    faiss.normalize_L2(vectors)
    index = build_index(vectors, "flat", nlist=8, m=8, nbits=8)
    assert index.ntotal == 64
    scores, ids = index.search(vectors[:1], 5)
    assert ids[0][0] == 0
    assert scores[0][0] == pytest.approx(1.0, abs=1e-5)


def test_ivfpq_self_retrieval(tmp_path):
    rng = np.random.RandomState(0)
    vectors = rng.randn(256, D).astype(np.float32)
    faiss.normalize_L2(vectors)
    index = build_index(vectors, "ivfpq", nlist=8, m=8, nbits=8)
    index.nprobe = 8
    scores, ids = index.search(vectors[:1], 5)
    # With an IP quantizer, IndexIVFPQ returns internal distances sorted
    # ascending; the self-vector lands at rank 1. The ranking is what
    # retrieval relies on, not the score calibration.
    assert ids[0][0] == 0
    assert len(set(scores[0].tolist())) > 1


def test_build_index_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown index type"):
        build_index(np.zeros((8, D), dtype=np.float32), "hnsw", nlist=8, m=8, nbits=8)


def test_id_map_alignment_after_rebuild(tmp_path):
    rng = np.random.RandomState(1)
    vectors = rng.randn(16, D).astype(np.float32)
    faiss.normalize_L2(vectors)

    clip_ids = [f"clip_{i}" for i in range(len(vectors))]
    for _ in range(2):  # rebuild twice; alignment must hold
        index = build_index(vectors, "flat", nlist=8, m=8, nbits=8)
        faiss.write_index(index, str(tmp_path / "flat.faiss"))
    _write_id_map(tmp_path / "id_map.parquet", clip_ids)

    retriever = Retriever(tmp_path / "flat.faiss", tmp_path / "id_map.parquet")
    hits = retriever.search(vectors[7], k=3)
    assert hits[0][0]["clip_id"] == "clip_7"
    assert hits[0][0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert hits[0][0]["mp3_path"] == "clip_7.mp3"
    assert "top_tags" in hits[0][0]


def test_retriever_sets_nprobe_for_ivf(tmp_path):
    rng = np.random.RandomState(0)
    vectors = rng.randn(256, D).astype(np.float32)
    faiss.normalize_L2(vectors)
    index = build_index(vectors, "ivfpq", nlist=8, m=8, nbits=8)
    faiss.write_index(index, str(tmp_path / "ivfpq.faiss"))
    _write_id_map(tmp_path / "id_map.parquet", [f"c{i}" for i in range(256)])

    retriever = Retriever(tmp_path / "ivfpq.faiss", tmp_path / "id_map.parquet")
    assert retriever.index.nprobe == 16


def test_compute_embeddings_and_cache(tmp_path, synthetic_dataset, tiny_checkpoint):
    from audiotag.data.dataset import MTATDataset

    cfg = Config(data_dir=synthetic_dataset)
    model = TinyEmbedModel(n_classes=len(TAGS), dim=8)
    ds = MTATDataset("train", cfg, training=False)

    embeds = compute_embeddings(model, ds, torch.device("cpu"), batch_size=2, num_workers=0)
    assert embeds.shape == (len(ds), 8)
    assert np.isfinite(embeds).all()

    cache_dir = tmp_path / "cache"
    emb1, meta1 = load_or_compute_embeddings(
        model, cfg, torch.device("cpu"), 2, cache_dir, tiny_checkpoint
    )
    emb2, meta2 = load_or_compute_embeddings(
        model, cfg, torch.device("cpu"), 2, cache_dir, tiny_checkpoint
    )
    assert np.array_equal(emb1, emb2)
    assert list(meta1.columns) == ["clip_id", "mp3_path", "labels"]
    assert len(meta1) == len(ds) + 2  # train + val clips
    assert (cache_dir / "embeddings.npy").exists()


def test_compute_embeddings_rejects_nonfinite(tmp_path, synthetic_dataset):
    from audiotag.data.dataset import MTATDataset

    class NaNModel(TinyEmbedModel):
        def embed(self, x):
            return super().embed(x) * float("nan")

    cfg = Config(data_dir=synthetic_dataset)
    ds = MTATDataset("train", cfg, training=False)
    with pytest.raises(RuntimeError, match="non-finite"):
        compute_embeddings(NaNModel(), ds, torch.device("cpu"), 2, 0)


def test_retrieval_quality_eval_end_to_end(
    tmp_path, synthetic_dataset, tiny_checkpoint, monkeypatch
):
    from audiotag.data.dataset import MTATDataset
    from audiotag.index import evaluate as eval_mod

    monkeypatch.chdir(tmp_path)
    cfg = Config(data_dir=synthetic_dataset)
    train_clips = list(MTATDataset("train", cfg, training=False).df["clip_id"])
    val_clips = list(MTATDataset("val", cfg, training=False).df["clip_id"])

    rng = np.random.RandomState(0)
    vectors = rng.randn(len(train_clips) + len(val_clips), 1024).astype(np.float32)
    faiss.normalize_L2(vectors)
    index = build_index(vectors, "flat", nlist=8, m=32, nbits=8)

    index_dir = tmp_path / "artifacts" / "index"
    index_dir.mkdir(parents=True)
    faiss.write_index(index, str(index_dir / "flat.faiss"))
    _write_id_map(index_dir / "id_map.parquet", train_clips + val_clips)

    code = eval_mod.main(
        [
            "--model",
            "cnn",
            "--data-dir",
            str(synthetic_dataset),
            "--index-dir",
            str(index_dir),
            "--checkpoint",
            str(tiny_checkpoint),
            "--device",
            "cpu",
            "--n-queries",
            "2",
        ]
    )
    assert code == 0
    report = (tmp_path / "reports" / "retrieval_quality.md").read_text()
    assert "model Jaccard" in report
    assert "random-neighbor Jaccard" in report
    for k in ("1", "5", "10"):
        assert f"| {k} |" in report


def test_retrieval_quality_helpers():
    from audiotag.index.evaluate import jaccard, mean_jaccard, tag_sets

    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 0.0
    assert jaccard({"a"}, {"a"}) == 1.0

    query_sets = {"q1": {"a", "b"}, "q2": {"c"}}
    neighbor_sets = {"q1": [{"a", "b"}, {"a"}], "q2": [{"c", "d"}]}
    assert mean_jaccard(query_sets, neighbor_sets, 1) == pytest.approx((1.0 + 0.5) / 2)

    df = pd.DataFrame(
        {
            "clip_id": ["c1", "c2"],
            "labels": [np.array([1, 0, 1], dtype=np.uint8), np.array([0, 1, 0], dtype=np.uint8)],
        }
    )
    sets = tag_sets(df, ["a", "b", "c"])
    assert sets == {"c1": {"a", "c"}, "c2": {"b"}}

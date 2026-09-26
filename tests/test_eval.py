"""Tests for the evaluation script, end to end on a synthetic dataset."""

import json

import numpy as np

from audiotag.config import Config
from audiotag.eval import prior_baseline_probs


def test_prior_baseline_probs(synthetic_dataset):
    cfg = Config(data_dir=synthetic_dataset)
    probs = prior_baseline_probs(cfg, n_test=3)
    assert probs.shape == (3, 5)
    # rows are constant across test clips, one value per tag
    assert np.all(probs[0] == probs[1])
    assert ((probs >= 0) & (probs <= 1)).all()


def test_eval_end_to_end(tmp_path, synthetic_dataset, tiny_checkpoint, monkeypatch):
    from audiotag.eval import main as eval_main

    monkeypatch.chdir(tmp_path)
    code = eval_main(
        [
            "--model",
            "cnn",
            "--data-dir",
            str(synthetic_dataset),
            "--checkpoint",
            str(tiny_checkpoint),
            "--device",
            "cpu",
            "--batch-size",
            "8",
        ]
    )
    assert code == 0

    metrics = json.loads((tmp_path / "artifacts" / "cnn" / "metrics.json").read_text())
    for key in (
        "macro_auc",
        "macro_map",
        "micro_map",
        "baseline_macro_map",
        "n_test",
        "n_skipped_tags",
    ):
        assert key in metrics
    assert metrics["n_test"] == 3
    assert 0.0 <= metrics["baseline_macro_map"] <= 1.0

    per_tag = (tmp_path / "reports" / "per_tag_metrics.csv").read_text()
    assert "average_precision" in per_tag
    assert (tmp_path / "reports" / "per_tag_auc.png").exists()


def test_eval_missing_checkpoint_errors(tmp_path, synthetic_dataset, monkeypatch):
    from audiotag.eval import main as eval_main

    monkeypatch.chdir(tmp_path)
    code = eval_main(
        [
            "--model",
            "cnn",
            "--data-dir",
            str(synthetic_dataset),
            "--checkpoint",
            str(tmp_path / "nope.pt"),
            "--device",
            "cpu",
        ]
    )
    assert code == 1

"""Tests for training utilities (no full training runs)."""

import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from audiotag.config import Config
from audiotag.train import compute_pos_weight, get_device, load_tags, predict_probs, set_seed
from tests.helpers import TAGS, TinyEmbedModel


def test_get_device_explicit():
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_auto_returns_device():
    assert isinstance(get_device("auto"), torch.device)


def test_set_seed_deterministic():
    set_seed(0)
    a = random.random()
    b = torch.rand(3)
    set_seed(0)
    assert random.random() == a
    assert torch.equal(torch.rand(3), b)


def test_load_tags(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "tags.json").write_text(json.dumps({"tags": TAGS, "positive_counts": {}}))
    assert load_tags(Config(data_dir=tmp_path / "data")) == TAGS


def test_compute_pos_weight():
    labels = np.array([[1, 0, 0], [0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
    weights = compute_pos_weight(labels)
    # tag 0: 2 positives -> (4-2)/2 = 1.0; tag 1: 1 positive -> 3.0; tag 2: none -> 4.0
    assert weights.tolist() == [1.0, 3.0, 4.0]


def test_compute_pos_weight_clamps(tmp_path):
    labels = np.zeros((100, 1), dtype=np.float32)
    labels[0, 0] = 1.0
    weights = compute_pos_weight(labels, max_weight=50.0)
    assert weights.item() == 50.0


def test_predict_probs_shapes_and_range():
    model = TinyEmbedModel(n_classes=5, dim=8)
    x = torch.randn(7, 1, 128, 313)
    y = torch.randint(0, 2, (7, 5)).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)

    y_out, probs = predict_probs(model, loader, torch.device("cpu"), use_amp=False)
    assert y_out.shape == (7, 5)
    assert probs.shape == (7, 5)
    assert np.array_equal(y_out, y.numpy())
    assert (probs >= 0).all() and (probs <= 1).all()

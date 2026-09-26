"""Tests for tagging metrics against hand-computed values.

Expected values follow scikit-learn's interpolated average precision:

- Tag 0: y=[1,0,1,0], s=[0.8,0.6,0.4,0.2]
  sorted: 0.8(1) 0.6(0) 0.4(1) 0.2(0)
  AP = 0.5*1 + 0.5*(2/3) = 5/6; AUC = 3/4 (only 0.4 loses to 0.6)
- Tag 1: y=[0,1,0,1], s=[0.7,0.5,0.3,0.1]
  sorted: 0.7(0) 0.5(1) 0.3(0) 0.1(1)
  AP = 0.5*(1/2) + 0.5*(2/4) = 1/2; AUC = 1/4
- Tag 2: no positives -> skipped
- macro AP = (5/6 + 1/2)/2 = 2/3; macro AUC = (3/4 + 1/4)/2 = 1/2
- micro (flattened): true [1,0,0,1,1,0,0,1], s [0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
  sorted: 0.8(1) 0.7(0) 0.6(0) 0.5(1) 0.4(1) 0.3(0) 0.2(0) 0.1(1)
  AP = 0.25*(1 + 1/2 + 3/5 + 4/8) = 0.25*2.6 = 0.65
"""

import numpy as np
import pytest

from audiotag.metrics import (
    macro_auc,
    macro_map,
    micro_map,
    tag_average_precision,
    tag_roc_auc,
)


@pytest.fixture
def hand_case():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    y_scores = np.array([[0.8, 0.7, 0.5], [0.6, 0.5, 0.5], [0.4, 0.3, 0.5], [0.2, 0.1, 0.5]])
    return y_true, y_scores


def test_per_tag_ap_hand_computed(hand_case):
    y_true, y_scores = hand_case
    aps = tag_average_precision(y_true, y_scores)
    assert aps[0] == pytest.approx(5.0 / 6.0)
    assert aps[1] == pytest.approx(0.5)
    assert np.isnan(aps[2])


def test_per_tag_auc_hand_computed(hand_case):
    y_true, y_scores = hand_case
    aucs = tag_roc_auc(y_true, y_scores)
    assert aucs[0] == pytest.approx(0.75)
    assert aucs[1] == pytest.approx(0.25)
    assert np.isnan(aucs[2])


def test_macro_map_and_skip(hand_case):
    y_true, y_scores = hand_case
    score, used, skipped = macro_map(y_true, y_scores)
    assert score == pytest.approx(2.0 / 3.0)
    assert used == 2
    assert skipped == 1


def test_macro_auc(hand_case):
    y_true, y_scores = hand_case
    score, used, skipped = macro_auc(y_true, y_scores)
    assert score == pytest.approx(0.5)
    assert used == 2
    assert skipped == 1


def test_micro_map_hand_computed():
    # Flattened: true [1,0,0,1,1,0,0,1], scores [0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    # Sorted: 0.8(1) 0.7(0) 0.6(0) 0.5(1) 0.4(1) 0.3(0) 0.2(0) 0.1(1)
    # AP = 0.25*(1 + 1/2 + 3/5 + 4/8) = 0.65
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    y_scores = np.array([[0.8, 0.7], [0.6, 0.5], [0.4, 0.3], [0.2, 0.1]])
    assert micro_map(y_true, y_scores) == pytest.approx(0.65)


def test_all_tags_positive_uses_all():
    y_true = np.ones((5, 3))
    y_scores = np.arange(15, dtype=np.float32).reshape(5, 3)
    score, used, skipped = macro_map(y_true, y_scores)
    assert used == 3
    assert skipped == 0
    assert score == pytest.approx(1.0)  # all positives -> AP 1.0 for every tag

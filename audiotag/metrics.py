"""Multi-label tagging metrics built on scikit-learn."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _per_tag(y_true: np.ndarray, y_scores: np.ndarray, metric) -> np.ndarray:
    """Compute a metric per tag; NaN where the tag has no positives."""
    out = np.full(y_true.shape[1], np.nan)
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        if yt.sum() == 0:
            continue
        out[j] = metric(yt, y_scores[:, j])
    return out


def tag_average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> np.ndarray:
    return _per_tag(y_true, y_scores, average_precision_score)


def tag_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> np.ndarray:
    return _per_tag(y_true, y_scores, roc_auc_score)


def macro_map(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, int, int]:
    """Mean AP over tags with >=1 positive; returns (score, used, skipped)."""
    aps = tag_average_precision(y_true, y_scores)
    used = ~np.isnan(aps)
    return float(aps[used].mean()), int(used.sum()), int((~used).sum())


def macro_auc(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, int, int]:
    aucs = tag_roc_auc(y_true, y_scores)
    used = ~np.isnan(aucs)
    return float(aucs[used].mean()), int(used.sum()), int((~used).sum())


def micro_map(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    return float(average_precision_score(y_true.ravel(), y_scores.ravel()))

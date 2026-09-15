"""Threshold-free and thresholded metrics for a heavily imbalanced problem.

ROC-AUC is reported but not trusted: with 0.17% positives a model can post
0.98 ROC-AUC while alerting on thousands of legitimate transactions per fraud.
PR-AUC, precision at fixed recall, and recall at a fixed false-positive rate
describe what an analyst team would actually experience.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def precision_at_recall(y: np.ndarray, p: np.ndarray, target_recall: float) -> tuple[float, float]:
    """(precision, threshold) at the highest threshold whose recall is >= target."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return 0.0, 1.0
    i = ok[-1]
    return float(precision[i]), float(thresholds[i])


def recall_at_fpr(y: np.ndarray, p: np.ndarray, max_fpr: float) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y, p)
    ok = np.where(fpr <= max_fpr)[0]
    i = ok[-1]
    return float(tpr[i]), float(thresholds[i])


@dataclass(frozen=True)
class Metrics:
    threshold: float
    pr_auc: float
    roc_auc: float
    brier: float
    ece: float
    precision: float
    recall: float
    f1: float
    alerts: int
    alert_rate: float
    tn: int
    fp: int
    fn: int
    tp: int
    precision_at_recall: dict[str, float]
    recall_at_fpr_0_1pct: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def compute_metrics(
    y: np.ndarray, p: np.ndarray, threshold: float, recall_targets: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9)
) -> Metrics:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(
        threshold=float(threshold),
        pr_auc=float(average_precision_score(y, p)),
        roc_auc=float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        ece=expected_calibration_error(y, p),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        alerts=int(tp + fp),
        alert_rate=float((tp + fp) / len(y)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        precision_at_recall={f"{r:.2f}": precision_at_recall(y, p, r)[0] for r in recall_targets},
        recall_at_fpr_0_1pct=recall_at_fpr(y, p, 0.001)[0],
    )

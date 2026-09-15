"""Choose the alert threshold by expected business cost, not by F1.

Every alert costs an analyst review. Every missed fraud costs the transaction
amount. The right threshold is the one that minimises the sum on data the
model has not seen, which is the validation block of the time split. F1 treats
a ₹5 fraud and a ₹5,000 fraud as equal; the cost curve does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fraud_platform.config import CostConfig


def expected_cost(
    y: np.ndarray, p: np.ndarray, amounts: np.ndarray, threshold: float, cost: CostConfig
) -> float:
    y = np.asarray(y).astype(int)
    alert = np.asarray(p) >= threshold
    review = cost.review_cost * alert.sum()
    missed = cost.loss_multiplier * np.asarray(amounts)[(y == 1) & ~alert].sum()
    return float(review + missed)


@dataclass(frozen=True)
class CostCurvePoint:
    threshold: float
    cost: float
    alerts: int
    caught_frauds: int
    missed_amount: float


def cost_curve(
    y: np.ndarray, p: np.ndarray, amounts: np.ndarray, cost: CostConfig, n_points: int = 400
) -> list[CostCurvePoint]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    # Candidate thresholds: quantiles of the score plus the extremes, so the
    # search is dense where the scores actually are.
    qs = np.quantile(p, np.linspace(0, 1, n_points))
    candidates = np.unique(np.concatenate([qs, [0.0, 1.0 + 1e-9]]))
    out = []
    for t in candidates:
        alert = p >= t
        out.append(CostCurvePoint(
            threshold=float(t),
            cost=expected_cost(y, p, amounts, t, cost),
            alerts=int(alert.sum()),
            caught_frauds=int((alert & (y == 1)).sum()),
            missed_amount=float(amounts[(y == 1) & ~alert].sum()),
        ))
    return out


def choose_threshold_by_cost(
    y: np.ndarray, p: np.ndarray, amounts: np.ndarray, cost: CostConfig
) -> tuple[float, list[CostCurvePoint]]:
    curve = cost_curve(y, p, amounts, cost)
    best = min(curve, key=lambda c: (c.cost, -c.threshold))
    return best.threshold, curve


def do_nothing_cost(y: np.ndarray, amounts: np.ndarray, cost: CostConfig) -> float:
    """Cost of never alerting: every fraud is lost."""
    return expected_cost(y, np.zeros(len(y)), amounts, 1.0, cost)

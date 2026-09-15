"""How sure are we about the threshold, and about the cost it produces?

The threshold is chosen on a validation block that contains a few dozen
frauds. Resampling that block shows how much the choice moves, and applying
each resampled threshold to the test block shows how much that movement costs.
Point estimates without this are the most common way a fraud model gets
oversold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fraud_platform.config import CostConfig
from fraud_platform.evaluation.cost import choose_threshold_by_cost, expected_cost


@dataclass(frozen=True)
class Interval:
    p2_5: float
    p50: float
    p97_5: float
    mean: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def _interval(values: np.ndarray) -> Interval:
    if len(values) == 0:
        return Interval(float("nan"), float("nan"), float("nan"), float("nan"))
    lo, mid, hi = np.percentile(values, [2.5, 50, 97.5])
    return Interval(float(lo), float(mid), float(hi), float(np.mean(values)))


def bootstrap_threshold(
    y: np.ndarray, p: np.ndarray, amounts: np.ndarray, cost: CostConfig, *, n_boot: int = 200, seed: int = 42
) -> np.ndarray:
    """Cost-optimal threshold on ``n_boot`` resamples of the selection block."""
    rng = np.random.default_rng(seed)
    y, p, amounts = (np.asarray(a) for a in (y, p, amounts))
    n = len(y)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:  # a resample with no fraud has no useful optimum
            out[b] = np.nan
            continue
        out[b], _ = choose_threshold_by_cost(y[idx], p[idx], amounts[idx], cost)
    return out[~np.isnan(out)]


def bootstrap_cost_at_threshold(
    y: np.ndarray, p: np.ndarray, amounts: np.ndarray, threshold: float, cost: CostConfig,
    *, n_boot: int = 500, seed: int = 42,
) -> np.ndarray:
    """Expected cost on resamples of the *evaluation* block at a fixed threshold."""
    rng = np.random.default_rng(seed)
    y, p, amounts = (np.asarray(a) for a in (y, p, amounts))
    n = len(y)
    return np.array([
        expected_cost(y[idx], p[idx], amounts[idx], threshold, cost)
        for idx in (rng.integers(0, n, n) for _ in range(n_boot))
    ])


def threshold_uncertainty(
    y_sel: np.ndarray, p_sel: np.ndarray, amt_sel: np.ndarray,
    y_eval: np.ndarray, p_eval: np.ndarray, amt_eval: np.ndarray,
    threshold: float, cost: CostConfig, *, n_boot: int = 200, seed: int = 42,
) -> dict:
    """Everything the report needs, in one dict.

    * ``threshold``: interval over bootstrap thresholds from the selection block.
    * ``eval_cost_of_bootstrap_thresholds``: apply each of those thresholds to the
      evaluation block; how much does threshold instability cost?
    * ``eval_cost_at_chosen``: resample the evaluation block at the chosen threshold;
      sampling noise of the headline cost number.
    """
    thresholds = bootstrap_threshold(y_sel, p_sel, amt_sel, cost, n_boot=n_boot, seed=seed)
    eval_costs = np.array([expected_cost(y_eval, p_eval, amt_eval, t, cost) for t in thresholds])
    at_chosen = bootstrap_cost_at_threshold(y_eval, p_eval, amt_eval, threshold, cost, n_boot=max(n_boot, 500), seed=seed)
    recalls = np.array([((p_eval >= t) & (y_eval == 1)).sum() / max(1, (y_eval == 1).sum()) for t in thresholds])
    alerts = np.array([(p_eval >= t).sum() for t in thresholds])
    return {
        "n_boot": int(len(thresholds)),
        "threshold": _interval(thresholds).as_dict(),
        "eval_cost_of_bootstrap_thresholds": _interval(eval_costs).as_dict(),
        "eval_recall_of_bootstrap_thresholds": _interval(recalls).as_dict(),
        "eval_alerts_of_bootstrap_thresholds": _interval(alerts).as_dict(),
        "eval_cost_at_chosen": _interval(at_chosen).as_dict(),
    }

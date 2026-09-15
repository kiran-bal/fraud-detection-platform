"""Figures for a run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from fraud_platform.evaluation.cost import CostCurvePoint  # noqa: E402

INK, ACCENT, WARN, GOOD, MUTED = "#1B2226", "#0E6B6B", "#B2413A", "#2B7A4B", "#9AA6A9"


def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for name, (y, p) in curves.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, linewidth=1.4, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_title("Precision-recall on the test block", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost_curve(
    curve: list[CostCurvePoint], chosen: float, baseline_cost: float, path: Path, title: str
) -> None:
    ts = [c.threshold for c in curve]
    costs = [c.cost for c in curve]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(ts, costs, color=ACCENT, linewidth=1.4, label="review cost + missed fraud")
    ax.axhline(baseline_cost, color=MUTED, linestyle="--", linewidth=1, label="never alert (lose all fraud)")
    ax.axvline(chosen, color=WARN, linestyle=":", linewidth=1.2, label=f"chosen t = {chosen:.3f}")
    ax.set_xlabel("Alert threshold on model score")
    ax.set_ylabel("Expected cost on validation block")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_score_distribution(y: np.ndarray, p: np.ndarray, threshold: float, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bins = np.linspace(0, 1, 41)
    ax.hist(p[y == 0], bins=bins, color=GOOD, alpha=0.6, label="legitimate", log=True)
    ax.hist(p[y == 1], bins=bins, color=WARN, alpha=0.7, label="fraud", log=True)
    ax.axvline(threshold, color=INK, linestyle=":", linewidth=1.2, label=f"t = {threshold:.3f}")
    ax.set_xlabel("Model score")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importance: dict[str, float], path: Path, title: str, n: int = 15) -> None:
    items = list(importance.items())[:n][::-1]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.barh([k for k, _ in items], [abs(v) for _, v in items], color=ACCENT)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[list[float], list[float]]:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            xs.append(float(p[m].mean()))
            ys.append(float(y[m].mean()))
    return xs, ys


def plot_calibration(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1)
    for label, p, colour in (("raw", raw, WARN), ("calibrated", calibrated, ACCENT)):
        xs, ys = _reliability(np.asarray(y), np.asarray(p))
        ax.plot(xs, ys, marker="o", color=colour, label=label)
    ax.set_xlabel("Predicted probability (bin mean)")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_threshold_bootstrap(uncertainty: dict, chosen: float, path: Path, title: str) -> None:
    t, c = uncertainty["threshold"], uncertainty["eval_cost_of_bootstrap_thresholds"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, (label, iv, mark) in zip(axes, (("threshold", t, chosen), ("test cost", c, None)), strict=True):
        ax.axvspan(iv["p2_5"], iv["p97_5"], color=ACCENT, alpha=0.15, label="95% interval")
        ax.axvline(iv["p50"], color=ACCENT, linewidth=1.2, label="median")
        if mark is not None:
            ax.axvline(mark, color=WARN, linestyle=":", linewidth=1.2, label="chosen")
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

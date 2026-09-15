"""Train one model on the time-ordered split and evaluate it once on the test block.

    fraud-train --config configs/lightgbm.yaml

Steps: validate the raw data → time split → fit features on train only →
fit model on train → choose the alert threshold on the validation block by
expected cost → score the test block once → write ``runs/<name>/``:

    config.yaml            what ran
    metrics.json           validation report, split summary, test metrics, cost analysis
    features.joblib        fitted feature pipeline
    model/                 saved model
    test_scores.parquet    id, time, amount, label, score for the test block
    feature_importance.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from fraud_platform import models
from fraud_platform.config import FIGURES_DIR, RUNS_DIR, RunConfig
from fraud_platform.data.schema import TransactionSchema, validate_or_raise
from fraud_platform.data.source import load_transactions
from fraud_platform.data.split import time_split
from fraud_platform.evaluation.calibration import Calibrator
from fraud_platform.evaluation.cost import choose_threshold_by_cost, do_nothing_cost, expected_cost
from fraud_platform.evaluation.metrics import compute_metrics
from fraud_platform.evaluation.report import (
    plot_calibration,
    plot_cost_curve,
    plot_feature_importance,
    plot_score_distribution,
    plot_threshold_bootstrap,
)
from fraud_platform.evaluation.uncertainty import threshold_uncertainty
from fraud_platform.features.pipeline import FeaturePipeline

logger = logging.getLogger("fraud_platform.train")


def run(cfg: RunConfig, df: pd.DataFrame | None = None, *, write_figures: bool = True, tracking: bool | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(cfg.seed)
    schema = TransactionSchema()
    run_dir = RUNS_DIR / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    df = load_transactions() if df is None else df
    validation = validate_or_raise(df, schema)
    for w in validation.warnings:
        logger.warning("data: %s", w)

    split = time_split(df, cfg.split, schema)
    summary = split.summary()
    for name, s in summary.items():
        logger.info("%-5s rows=%7d frauds=%4d rate=%.4f", name, s["rows"], s["frauds"], s["fraud_rate"])

    features = FeaturePipeline(schema).fit(split.train)
    parts = (split.train, split.valid, split.test)
    x_train, x_valid, x_test = (features.transform(part) for part in parts)
    y_train, y_valid, y_test = (part[schema.target_column].to_numpy() for part in parts)
    amt_valid, amt_test = (part[schema.amount_column].to_numpy() for part in (split.valid, split.test))

    t0 = time.perf_counter()
    model = models.build(cfg.model, **cfg.params).fit(x_train, y_train)
    fit_seconds = time.perf_counter() - t0

    # The validation block is split in two by time: the first half fits the
    # calibrator, the second half chooses the threshold. Neither touches test.
    raw_valid = model.predict_proba(x_valid)
    half = len(split.valid) // 2
    calibrator = Calibrator(cfg.calibration).fit(raw_valid[:half], y_valid[:half])
    p_valid = calibrator.transform(raw_valid)
    y_sel, p_sel, amt_sel = y_valid[half:], p_valid[half:], amt_valid[half:]
    threshold, curve = choose_threshold_by_cost(y_sel, p_sel, amt_sel, cfg.cost)
    valid_metrics = compute_metrics(y_sel, p_sel, threshold, cfg.recall_targets)
    logger.info("calibration=%s on %d rows (%d frauds); threshold from %d rows (%d frauds)",
                cfg.calibration, half, int(y_valid[:half].sum()), len(y_sel), int(y_sel.sum()))

    raw_test = model.predict_proba(x_test)
    p_test = calibrator.transform(raw_test)
    test_metrics = compute_metrics(y_test, p_test, threshold, cfg.recall_targets)
    raw_test_metrics = compute_metrics(y_test, raw_test, 0.5, cfg.recall_targets)
    uncertainty = (
        threshold_uncertainty(y_sel, p_sel, amt_sel, y_test, p_test, amt_test, threshold, cfg.cost,
                              n_boot=cfg.n_boot, seed=cfg.seed)
        if cfg.n_boot > 0 else {}
    )
    test_cost = expected_cost(y_test, p_test, amt_test, threshold, cfg.cost)
    test_do_nothing = do_nothing_cost(y_test, amt_test, cfg.cost)
    logger.info(
        "test  PR-AUC=%.4f ROC-AUC=%.4f | t=%.4f precision=%.3f recall=%.3f alerts=%d | "
        "cost %.0f vs do-nothing %.0f",
        test_metrics.pr_auc, test_metrics.roc_auc, threshold, test_metrics.precision, test_metrics.recall,
        test_metrics.alerts, test_cost, test_do_nothing,
    )

    # Artefacts
    model.save(run_dir / "model")
    joblib.dump(features, run_dir / "features.joblib")
    calibrator.save(run_dir / "calibrator.joblib")
    pd.DataFrame({
        "transaction_id": split.test[schema.id_column].to_numpy(),
        "time": split.test[schema.time_column].to_numpy(),
        "amount": amt_test, "is_fraud": y_test, "score": p_test, "raw_score": raw_test,
    }).to_parquet(run_dir / "test_scores.parquet", index=False)
    importance = model.feature_importance(features.feature_names) or {}
    (run_dir / "feature_importance.json").write_text(json.dumps(importance, indent=2))
    if write_figures:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_cost_curve(
            curve, threshold, do_nothing_cost(y_valid, amt_valid, cfg.cost),
            FIGURES_DIR / f"{cfg.name}_cost_curve.png",
            f"{cfg.name}: expected cost by threshold (validation)",
        )
        plot_score_distribution(
            y_test, p_test, threshold, FIGURES_DIR / f"{cfg.name}_scores.png",
            f"{cfg.name}: test-block score distribution",
        )
        if importance:
            plot_feature_importance(
                importance, FIGURES_DIR / f"{cfg.name}_importance.png", f"{cfg.name}: feature importance"
            )
        plot_calibration(
            y_test, raw_test, p_test, FIGURES_DIR / f"{cfg.name}_calibration.png",
            f"{cfg.name}: reliability before/after {cfg.calibration} calibration (test)",
        )
        if uncertainty:
            plot_threshold_bootstrap(
                uncertainty, threshold, FIGURES_DIR / f"{cfg.name}_threshold_bootstrap.png",
                f"{cfg.name}: threshold and test-cost spread over {uncertainty['n_boot']} resamples",
            )

    result = {
        "name": cfg.name, "model": cfg.model, "params": cfg.params, "seed": cfg.seed,
        "data_validation": validation.as_dict(), "split": summary,
        "feature_names": features.feature_names, "fit_seconds": round(fit_seconds, 2),
        "threshold": threshold,
        "calibration": cfg.calibration,
        "calibration_rows": {"calibrate": int(half), "select": int(len(y_sel))},
        "validation": valid_metrics.as_dict(),
        "test": test_metrics.as_dict(),
        "test_raw": {"ece": raw_test_metrics.ece, "brier": raw_test_metrics.brier, "pr_auc": raw_test_metrics.pr_auc},
        "uncertainty": uncertainty,
        "cost": {
            **cfg.cost.__dict__,
            "test_expected_cost": test_cost,
            "test_do_nothing_cost": test_do_nothing,
            "test_cost_saved_vs_do_nothing": test_do_nothing - test_cost,
            "test_missed_fraud_amount": float(amt_test[(y_test == 1) & (p_test < threshold)].sum()),
            "test_caught_fraud_amount": float(amt_test[(y_test == 1) & (p_test >= threshold)].sum()),
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    if cfg.tracking if tracking is None else tracking:
        from fraud_platform.tracking import log_run

        result["mlflow_run_id"] = log_run(result, run_dir)
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        logger.info("logged to MLflow run %s", result["mlflow_run_id"])
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-tracking", action="store_true", help="skip MLflow logging and registration")
    args = ap.parse_args(argv)
    run(RunConfig.from_yaml(Path(args.config)), tracking=False if args.no_tracking else None)


if __name__ == "__main__":
    main()

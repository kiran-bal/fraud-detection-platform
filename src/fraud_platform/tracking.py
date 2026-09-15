"""Experiment tracking and model registry on MLflow.

Every ``fraud-train`` run is logged to the ``fraud-detection`` experiment:
flattened parameters, validation/test/cost/uncertainty metrics, the run
directory as artifacts, and a ``pyfunc`` model that bundles feature pipeline,
model and calibrator so ``mlflow models serve`` produces the same scores the
batch job does. Each logged model is registered as a new version of
``fraud-detector``; promotion to the ``champion`` alias goes through a quality
gate (``fraud-promote``).

Store: ``MLFLOW_TRACKING_URI`` if set, otherwise a local SQLite database at
``mlflow.db`` with artifacts under ``mlruns/`` (both gitignored).
``mlflow ui --backend-store-uri sqlite:///mlflow.db`` to browse.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_platform.config import PROJECT_ROOT

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # keep CLI output clean

EXPERIMENT = "fraud-detection"
MODEL_NAME = "fraud-detector"
CHAMPION = "champion"


def tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').resolve()}"


def artifact_location() -> str:
    return os.getenv("MLFLOW_ARTIFACT_LOCATION") or (PROJECT_ROOT / "mlruns").resolve().as_uri()


def _ensure_experiment() -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=artifact_location())
    mlflow.set_experiment(EXPERIMENT)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return None


class FraudPyfunc:
    """MLflow pyfunc wrapper: raw transactions in, scores and alerts out."""

    def load_context(self, context) -> None:  # pragma: no cover - exercised through mlflow
        from fraud_platform.predict import Scorer

        self.scorer = Scorer(Path(context.artifacts["run_dir"]))

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:  # pragma: no cover
        return self.scorer.score(model_input)


def log_run(result: dict, run_dir: Path, *, register: bool = True) -> str:
    """Log a finished training run. Returns the MLflow run id."""
    import mlflow

    _ensure_experiment()
    with mlflow.start_run(run_name=result["name"]) as run:
        params = {"model": result["model"], "seed": result["seed"], "calibration": result.get("calibration", "none"),
                  **{f"param.{k}": v for k, v in result["params"].items()},
                  **{f"cost.{k}": v for k in ("review_cost", "loss_multiplier") for v in [result["cost"][k]]}}
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        metrics = {}
        for section in ("validation", "test"):
            metrics.update({f"{section}.{k}": v for k, v in _flatten(result[section]).items()
                            if isinstance(v, (int, float))})
        metrics.update({f"cost.{k}": v for k, v in result["cost"].items() if isinstance(v, (int, float))})
        if "uncertainty" in result:
            metrics.update({f"uncertainty.{k}": v for k, v in _flatten(result["uncertainty"]).items()
                            if isinstance(v, (int, float))})
        metrics["threshold"] = result["threshold"]
        metrics["fit_seconds"] = result["fit_seconds"]
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"git_sha": _git_sha() or "unknown", "data_rows": result["data_validation"]["rows"],
                         "split": json.dumps(result["split"]["test"])[:250]})
        mlflow.log_artifacts(str(run_dir), artifact_path="run")
        if register:
            from mlflow.pyfunc import PythonModel

            class _Model(PythonModel, FraudPyfunc):
                pass

            mlflow.pyfunc.log_model(
                name="model", python_model=_Model(), artifacts={"run_dir": str(run_dir)},
                registered_model_name=MODEL_NAME, pip_requirements=["fraud-platform"],
            )
        return run.info.run_id


def current_champion() -> dict | None:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    except Exception:
        return None
    run = client.get_run(mv.run_id)
    return {"version": str(mv.version), "run_id": mv.run_id, "metrics": run.data.metrics}


def passes_gate(candidate: dict[str, float], champion: dict[str, float] | None, *, tolerance: float = 0.01) -> tuple[bool, str]:
    """A candidate may replace the champion if it is not worse on ranking quality or on cost.

    Pure so it can be tested; the CLI feeds it MLflow metrics.
    """
    if champion is None:
        return True, "no champion yet"
    c_auc, k_auc = candidate.get("test.pr_auc", 0.0), champion.get("test.pr_auc", 0.0)
    c_cost, k_cost = candidate.get("cost.test_expected_cost", float("inf")), champion.get("cost.test_expected_cost", float("inf"))
    if c_auc < k_auc - tolerance:
        return False, f"test PR-AUC {c_auc:.4f} is below champion {k_auc:.4f} by more than {tolerance}"
    if c_cost > k_cost * (1 + tolerance):
        return False, f"test expected cost {c_cost:.0f} exceeds champion {k_cost:.0f} by more than {tolerance:.0%}"
    return True, f"PR-AUC {c_auc:.4f} vs {k_auc:.4f}, cost {c_cost:.0f} vs {k_cost:.0f}"


def promote(version: str, *, force: bool = False) -> tuple[bool, str]:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    mv = client.get_model_version(MODEL_NAME, version)
    candidate = client.get_run(mv.run_id).data.metrics
    champ = current_champion()
    ok, reason = passes_gate(candidate, champ["metrics"] if champ else None)
    if not ok and not force:
        return False, reason
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
    return True, f"version {version} is now {CHAMPION} ({reason}{', forced' if not ok else ''})"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Promote a registered model version to the champion alias after a quality gate.")
    ap.add_argument("--version", required=True)
    ap.add_argument("--force", action="store_true", help="promote even if the gate fails")
    args = ap.parse_args(argv)
    ok, msg = promote(args.version, force=args.force)
    print(("PROMOTED: " if ok else "REJECTED: ") + msg)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

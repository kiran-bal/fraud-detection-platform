import os

import pytest

from fraud_platform import tracking
from fraud_platform.tracking import passes_gate


def test_gate_accepts_first_model_and_rejects_regressions():
    ok, _ = passes_gate({"test.pr_auc": 0.7, "cost.test_expected_cost": 100.0}, None)
    assert ok
    champ = {"test.pr_auc": 0.78, "cost.test_expected_cost": 4000.0}
    assert passes_gate({"test.pr_auc": 0.775, "cost.test_expected_cost": 4010.0}, champ)[0]      # within tolerance
    assert not passes_gate({"test.pr_auc": 0.70, "cost.test_expected_cost": 3900.0}, champ)[0]  # worse ranking
    assert not passes_gate({"test.pr_auc": 0.80, "cost.test_expected_cost": 4500.0}, champ)[0]  # worse cost


@pytest.mark.slow
def test_log_and_promote_roundtrip(tmp_path, monkeypatch, transactions):
    """End to end through a temporary MLflow file store: train → log → register → promote."""
    pytest.importorskip("mlflow")
    from fraud_platform import train
    from fraud_platform.config import RunConfig

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_LOCATION", (tmp_path / "mlruns").as_uri())
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path / "runs")
    cfg = RunConfig(name="toy", model="logreg", n_boot=10, calibration="sigmoid")
    result = train.run(cfg, df=transactions, write_figures=False, tracking=True)
    assert result["mlflow_run_id"]
    assert tracking.current_champion() is None
    ok, msg = tracking.promote("1")
    assert ok, msg
    champ = tracking.current_champion()
    assert champ["version"] == "1" and "test.pr_auc" in champ["metrics"]
    assert os.environ["MLFLOW_TRACKING_URI"].startswith("sqlite:")

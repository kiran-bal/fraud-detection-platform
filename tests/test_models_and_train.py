import numpy as np
import pytest

from fraud_platform import models, train
from fraud_platform.config import RunConfig
from fraud_platform.features.pipeline import FeaturePipeline
from fraud_platform.predict import Scorer


@pytest.mark.parametrize("name,params", [("logreg", {}), ("lightgbm", {"n_estimators": 60})])
def test_models_fit_predict_save_load(name, params, transactions, tmp_path):
    X = FeaturePipeline().fit_transform(transactions)
    y = transactions.is_fraud.to_numpy()
    m = models.build(name, **params).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(X),) and np.all((p >= 0) & (p <= 1))
    assert p[y == 1].mean() > p[y == 0].mean()
    m.save(tmp_path / "m")
    loaded = models.load(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict_proba(X), p, atol=1e-6)
    imp = loaded.feature_importance(list(X.columns))
    assert set(imp) == set(X.columns)


def test_unknown_model():
    with pytest.raises(KeyError):
        models.build("nope")


def test_end_to_end_run_and_batch_scoring(transactions, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path / "runs")
    cfg = RunConfig(name="toy", model="lightgbm", params={"n_estimators": 80})
    result = train.run(cfg, df=transactions, write_figures=False, tracking=False)
    assert result["test"]["pr_auc"] > 0.5
    assert 0 <= result["threshold"] <= 1
    assert result["cost"]["test_expected_cost"] <= result["cost"]["test_do_nothing_cost"]
    assert result["split"]["train"]["time_to"] <= result["split"]["test"]["time_from"]

    scorer = Scorer(tmp_path / "runs" / "toy")
    scored = scorer.score(transactions.drop(columns=["is_fraud"]).head(50))
    assert list(scored.columns) == ["transaction_id", "score", "alert", "threshold"]
    assert len(scored) == 50

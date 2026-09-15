"""Gradient-boosted trees, the production candidate.

``scale_pos_weight`` counters the 1:580 imbalance during training. That
distorts the raw probabilities upward, which is why thresholds are chosen on
validation data by cost rather than fixed at 0.5, and why calibration is a
tracked metric rather than assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud_platform.models.base import FraudModel

DEFAULTS: dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "scale_pos_weight": 20.0,
    "random_state": 42,
    "verbose": -1,
}


class LightGBMModel(FraudModel):
    kind = "lightgbm"

    def __init__(self, **params: Any) -> None:
        super().__init__(**{**DEFAULTS, **params})
        self.booster: lgb.LGBMClassifier | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> LightGBMModel:
        self.booster = lgb.LGBMClassifier(**self._params)
        self.booster.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        gains = self.booster.booster_.feature_importance(importance_type="gain")
        total = float(gains.sum()) or 1.0
        pairs = zip(feature_names, (float(g) / total for g in gains), strict=True)
        return dict(sorted(pairs, key=lambda kv: -kv[1]))

    def _save_estimator(self, path: Path) -> None:
        joblib.dump(self.booster, path / "booster.joblib")

    @classmethod
    def _load_estimator(cls, path: Path, params: dict[str, Any]) -> LightGBMModel:
        obj = cls(**params)
        obj.booster = joblib.load(path / "booster.joblib")
        return obj

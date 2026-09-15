"""Regularised logistic regression on standardised features: the baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_platform.models.base import FraudModel


class LogisticModel(FraudModel):
    kind = "logreg"

    def __init__(self, C: float = 0.1, class_weight: str | None = "balanced", max_iter: int = 2000, **_: Any):  # noqa: N803
        super().__init__(C=C, class_weight=class_weight, max_iter=max_iter)
        self.pipeline: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> LogisticModel:
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                C=self._params["C"], class_weight=self._params["class_weight"],
                max_iter=self._params["max_iter"], solver="lbfgs",
            )),
        ])
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        coef = self.pipeline.named_steps["clf"].coef_.ravel()
        return dict(sorted(zip(feature_names, map(float, coef), strict=True), key=lambda kv: -abs(kv[1])))

    def _save_estimator(self, path: Path) -> None:
        joblib.dump(self.pipeline, path / "pipeline.joblib")

    @classmethod
    def _load_estimator(cls, path: Path, params: dict[str, Any]) -> LogisticModel:
        obj = cls(**params)
        obj.pipeline = joblib.load(path / "pipeline.joblib")
        return obj

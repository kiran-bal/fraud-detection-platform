"""Post-hoc probability calibration.

Class weighting during training (``class_weight='balanced'``,
``scale_pos_weight``) buys ranking quality at the price of probabilities that
no longer mean "chance this is fraud". Anything downstream that reads the
score as a probability, such as an expected-loss calculation or a
customer-facing risk band, needs the calibrated value.

Two methods, both fitted on data the model did not train on:

* ``sigmoid`` (Platt): logistic regression on the logit of the score.
  Two parameters, stable with few positives, cannot fix a non-monotone miscalibration.
* ``isotonic``: monotone step function. Flexible, needs many positives or it
  overfits; with a few dozen frauds it is the noisier choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CalibrationKind = Literal["none", "sigmoid", "isotonic"]
EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class Calibrator:
    kind: CalibrationKind = "none"
    _model: object | None = field(default=None, repr=False)
    fitted: bool = False

    def fit(self, scores: np.ndarray, y: np.ndarray) -> Calibrator:
        y = np.asarray(y).astype(int)
        if self.kind == "none":
            pass
        elif self.kind == "sigmoid":
            self._model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(scores).reshape(-1, 1), y)
        elif self.kind == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, y)
        else:
            raise ValueError(f"unknown calibration '{self.kind}'")
        self.fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self.kind == "none":
            return scores
        if not self.fitted:
            raise RuntimeError("Calibrator.fit() must run before transform()")
        if self.kind == "sigmoid":
            return self._model.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]
        return self._model.predict(scores)

    def save(self, path: Path) -> None:
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: Path) -> Calibrator:
        return joblib.load(Path(path))

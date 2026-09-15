from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class FraudModel(ABC):
    kind: str = "base"

    def __init__(self, **params: Any) -> None:
        self._params = params

    def params(self) -> dict[str, Any]:
        return dict(self._params)

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> FraudModel: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def _save_estimator(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def _load_estimator(cls, path: Path, params: dict[str, Any]) -> FraudModel: ...

    def feature_importance(self, feature_names: list[str]) -> dict[str, float] | None:
        return None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.json").write_text(json.dumps({"kind": self.kind, "params": self.params()}, indent=2))
        self._save_estimator(path)

    @classmethod
    def load(cls, path: Path) -> FraudModel:
        meta = json.loads((Path(path) / "model.json").read_text())
        return cls._load_estimator(Path(path), meta["params"])

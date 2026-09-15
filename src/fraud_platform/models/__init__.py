"""Model registry. Every model: fit(X, y), predict_proba(X) -> P(fraud), save(dir), load(dir)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fraud_platform.models.base import FraudModel
from fraud_platform.models.lightgbm_model import LightGBMModel
from fraud_platform.models.logreg import LogisticModel

_REGISTRY: dict[str, type[FraudModel]] = {
    "logreg": LogisticModel,
    "lightgbm": LightGBMModel,
}


def build(name: str, **params: Any) -> FraudModel:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)


def load(path: Path) -> FraudModel:
    meta = json.loads((Path(path) / "model.json").read_text())
    return _REGISTRY[meta["kind"]].load(Path(path))


__all__ = ["FraudModel", "LightGBMModel", "LogisticModel", "build", "load"]

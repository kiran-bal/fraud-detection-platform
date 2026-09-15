"""Paths and run configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    explicit = os.getenv("FRAUD_PROJECT_ROOT")
    if explicit:
        return Path(explicit).resolve()
    cwd = Path.cwd()
    if (cwd / "configs").is_dir() and (cwd / "pyproject.toml").exists():
        return cwd
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RUNS_DIR = PROJECT_ROOT / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIGS_DIR = PROJECT_ROOT / "configs"


@dataclass(frozen=True)
class SplitConfig:
    """Fractions of the time-ordered data. Must sum to 1."""

    train: float = 0.6
    valid: float = 0.2
    test: float = 0.2

    def __post_init__(self) -> None:
        if abs(self.train + self.valid + self.test - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")


@dataclass(frozen=True)
class CostConfig:
    """Business costs used to choose the alert threshold.

    ``review_cost`` is what one analyst review costs, charged for every alert
    (true or false positive). A missed fraud costs the transaction amount
    (``loss_multiplier`` scales it, for chargeback fees and the like).
    """

    review_cost: float = 10.0
    loss_multiplier: float = 1.0


@dataclass(frozen=True)
class RunConfig:
    name: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    split: SplitConfig = field(default_factory=SplitConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    #: Recall levels at which precision is reported.
    recall_targets: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9)
    #: none | sigmoid | isotonic. Fitted on the first (by time) half of the validation block.
    calibration: str = "none"
    #: Bootstrap resamples for the threshold / cost intervals. 0 disables.
    n_boot: int = 200
    #: Log the run to MLflow and register the model.
    tracking: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        split = SplitConfig(**raw.pop("split", {}))
        cost = CostConfig(**raw.pop("cost", {}))
        if "recall_targets" in raw:
            raw["recall_targets"] = tuple(raw["recall_targets"])
        return cls(split=split, cost=cost, **raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "params": dict(self.params),
            "seed": self.seed,
            "split": self.split.__dict__.copy(),
            "cost": self.cost.__dict__.copy(),
            "recall_targets": list(self.recall_targets),
            "calibration": self.calibration,
            "n_boot": self.n_boot,
            "tracking": self.tracking,
        }

"""Batch scoring with a saved run.

    fraud-score --run runs/lightgbm --input transactions.parquet --output scored.parquet

Validates the input against the scoring schema (no label needed), applies the
fitted feature pipeline and model, and writes id, score and alert flag.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from fraud_platform import models
from fraud_platform.data.schema import TransactionSchema, validate_or_raise
from fraud_platform.evaluation.calibration import Calibrator
from fraud_platform.features.pipeline import FeaturePipeline


class Scorer:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.features: FeaturePipeline = joblib.load(self.run_dir / "features.joblib")
        self.model = models.load(self.run_dir / "model")
        cal_path = self.run_dir / "calibrator.joblib"
        self.calibrator = Calibrator.load(cal_path) if cal_path.exists() else Calibrator("none")
        meta = json.loads((self.run_dir / "metrics.json").read_text())
        self.threshold = float(meta["threshold"])
        self.schema = TransactionSchema()

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_or_raise(df, self.schema, training=False)
        p = self.calibrator.transform(self.model.predict_proba(self.features.transform(df)))
        return pd.DataFrame({
            self.schema.id_column: df[self.schema.id_column].to_numpy(),
            "score": p,
            "alert": (p >= self.threshold).astype(int),
            "threshold": self.threshold,
        })


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--input", required=True, help="parquet or csv with the scoring schema")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    src = Path(args.input)
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    out = Scorer(args.run).score(df)
    dst = Path(args.output)
    out.to_parquet(dst, index=False) if dst.suffix == ".parquet" else out.to_csv(dst, index=False)
    print(f"scored {len(out):,} rows, {int(out['alert'].sum())} alerts, written to {dst}")


if __name__ == "__main__":
    main()

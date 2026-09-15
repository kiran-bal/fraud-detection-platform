"""Feature construction shared by training, batch scoring and real-time scoring.

One class, fit once, serialised with the model, so the three code paths cannot
drift apart. What it does with this dataset:

* ``log_amount``                log1p of the amount; the raw amount is heavy-tailed.
* ``hour_of_day`` as sin/cos    the 2-day window has a strong daily cycle and fraud
                                clusters at night; sin/cos keeps 23:00 next to 00:00.
* ``is_night``                  02:00-06:00 flag, a coarse version of the same signal.
* ``amount_z_hour``             amount standardised against the hour-of-day
                                distribution seen in training (a cheap "unusual for
                                the time of day" score).
* ``V1..V28``                   passed through.

``time`` itself is never a feature: it increases monotonically through the
split, so any model would learn "later = test set", which is leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fraud_platform.data.schema import PCA_COLUMNS, TransactionSchema

SECONDS_PER_DAY = 86_400


@dataclass
class FeaturePipeline:
    schema: TransactionSchema = field(default_factory=TransactionSchema)
    hour_amount_stats: dict[int, tuple[float, float]] = field(default_factory=dict)
    global_amount_stats: tuple[float, float] = (0.0, 1.0)
    fitted: bool = False

    @property
    def feature_names(self) -> list[str]:
        return ["log_amount", "hour_sin", "hour_cos", "is_night", "amount_z_hour", *PCA_COLUMNS]

    @staticmethod
    def hour_of_day(time_seconds: pd.Series) -> pd.Series:
        return ((time_seconds % SECONDS_PER_DAY) // 3600).astype(int)

    def fit(self, df: pd.DataFrame) -> FeaturePipeline:
        log_amount = np.log1p(df[self.schema.amount_column])
        hours = self.hour_of_day(df[self.schema.time_column])
        grouped = log_amount.groupby(hours)
        self.hour_amount_stats = {
            int(h): (float(g.mean()), float(g.std(ddof=0)) or 1.0) for h, g in grouped
        }
        self.global_amount_stats = (float(log_amount.mean()), float(log_amount.std(ddof=0)) or 1.0)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FeaturePipeline.fit() must run before transform()")
        out = pd.DataFrame(index=df.index)
        log_amount = np.log1p(df[self.schema.amount_column].astype(float))
        hours = self.hour_of_day(df[self.schema.time_column])
        out["log_amount"] = log_amount
        out["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        out["is_night"] = ((hours >= 2) & (hours < 6)).astype(int)
        means = hours.map(lambda h: self.hour_amount_stats.get(int(h), self.global_amount_stats)[0])
        stds = hours.map(lambda h: self.hour_amount_stats.get(int(h), self.global_amount_stats)[1])
        out["amount_z_hour"] = (log_amount - means.astype(float)) / stds.astype(float)
        for c in PCA_COLUMNS:
            out[c] = df[c].astype(float)
        return out[self.feature_names]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

"""Synthetic transactions with a learnable fraud signal, so tests never download data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_platform.data.schema import PCA_COLUMNS


def make_transactions(n: int = 4000, fraud_rate: float = 0.02, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    is_fraud = (rng.random(n) < fraud_rate).astype(int)
    time = np.sort(rng.uniform(0, 2 * 86_400, n))
    amount = np.where(is_fraud, rng.lognormal(4.5, 1.0, n), rng.lognormal(3.0, 1.2, n)).round(2)
    df = pd.DataFrame({"transaction_id": np.arange(n), "time": time, "amount": amount})
    for i, c in enumerate(PCA_COLUMNS):
        shift = 2.5 if i < 4 else 0.0
        df[c] = rng.normal(0, 1, n) + is_fraud * shift * (1 if i % 2 == 0 else -1)
    df["is_fraud"] = is_fraud
    return df


@pytest.fixture
def transactions() -> pd.DataFrame:
    return make_transactions()

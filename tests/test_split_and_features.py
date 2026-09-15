import numpy as np
import pytest

from fraud_platform.config import SplitConfig
from fraud_platform.data.split import time_split
from fraud_platform.features.pipeline import FeaturePipeline


def test_split_is_contiguous_in_time_and_disjoint(transactions):
    shuffled = transactions.sample(frac=1, random_state=3)
    s = time_split(shuffled, SplitConfig(0.6, 0.2, 0.2))
    assert len(s.train) + len(s.valid) + len(s.test) == len(transactions)
    assert s.train.time.max() <= s.valid.time.min() <= s.valid.time.max() <= s.test.time.min()
    ids = [set(p.transaction_id) for p in (s.train, s.valid, s.test)]
    assert ids[0].isdisjoint(ids[1]) and ids[1].isdisjoint(ids[2])
    assert s.summary()["train"]["rows"] == len(s.train)


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        SplitConfig(0.5, 0.2, 0.2)


def test_features_exclude_raw_time_and_have_expected_columns(transactions):
    fp = FeaturePipeline().fit(transactions)
    X = fp.transform(transactions)
    assert "time" not in X.columns and "amount" not in X.columns
    assert list(X.columns) == fp.feature_names
    assert X.shape == (len(transactions), 5 + 28)
    assert np.isfinite(X.to_numpy()).all()


def test_hour_features_are_cyclic():
    import pandas as pd

    fp = FeaturePipeline()
    hours = fp.hour_of_day(pd.Series([0.0, 3600.0, 23 * 3600.0, 86_400.0 + 30.0]))
    assert hours.tolist() == [0, 1, 23, 0]


def test_transform_before_fit_raises(transactions):
    with pytest.raises(RuntimeError):
        FeaturePipeline().transform(transactions)


def test_amount_z_uses_training_stats_for_unseen_hours(transactions):
    fp = FeaturePipeline().fit(transactions[transactions.time < 3600])  # only hour 0 seen
    X = fp.transform(transactions)
    assert np.isfinite(X["amount_z_hour"]).all()

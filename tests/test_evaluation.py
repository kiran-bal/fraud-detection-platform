import numpy as np

from fraud_platform.config import CostConfig
from fraud_platform.evaluation.cost import choose_threshold_by_cost, do_nothing_cost, expected_cost
from fraud_platform.evaluation.metrics import compute_metrics, precision_at_recall


def test_expected_cost_counts_reviews_and_missed_amounts():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    amounts = np.array([100.0, 50.0, 10.0, 10.0])
    cost = CostConfig(review_cost=5.0)
    # alerts on rows 0 and 2 (2 reviews = 10), miss row 1 (50)
    assert expected_cost(y, p, amounts, 0.5, cost) == 60.0
    assert do_nothing_cost(y, amounts, cost) == 150.0


def test_threshold_choice_prefers_catching_expensive_fraud():
    rng = np.random.default_rng(0)
    y = np.array([0] * 990 + [1] * 10)
    p = np.concatenate([rng.uniform(0, 0.4, 990), rng.uniform(0.3, 1.0, 10)])
    amounts = np.concatenate([np.full(990, 20.0), np.full(10, 2000.0)])
    t, curve = choose_threshold_by_cost(y, p, amounts, CostConfig(review_cost=10.0))
    caught = ((p >= t) & (y == 1)).sum()
    assert caught >= 9                      # expensive fraud is worth many reviews
    assert min(c.cost for c in curve) == expected_cost(y, p, amounts, t, CostConfig(review_cost=10.0))


def test_metrics_shape_and_precision_at_recall():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.2, 0.9, 0.5])
    m = compute_metrics(y, p, 0.45, recall_targets=(0.5, 1.0))
    assert m.tp == 2 and m.fp == 0 and m.recall == 1.0 and m.alerts == 2
    assert m.precision_at_recall["1.00"] == 1.0
    assert precision_at_recall(y, p, 1.0)[0] == 1.0

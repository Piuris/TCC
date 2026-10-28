import numpy as np
import pytest

from fraude.config import CostConfig
from fraude.evaluation import (
    best_threshold_cost,
    best_threshold_f1,
    bootstrap_ci,
    classification_metrics,
    cost_curve,
    cost_report,
    dal_pozzolo_correction,
    expected_cost,
)


def test_metrics_known_confusion_matrix():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1])
    m = classification_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)


def test_trivial_classifier_has_high_accuracy_but_zero_recall():
    y = np.r_[np.ones(2), np.zeros(998)]
    m = classification_metrics(y, np.zeros(1000), 0.5)
    assert m["accuracy"] == pytest.approx(0.998)
    assert m["recall"] == 0 and m["mcc"] == 0


def test_expected_cost():
    y = np.array([1, 1, 0, 0])
    pred = np.array([0, 1, 1, 0])
    assert expected_cost(y, pred, CostConfig(fn=100, fp=5)) == 105


def test_expected_cost_with_amount():
    y = np.array([1, 1, 0])
    pred = np.array([0, 0, 0])
    amount = np.array([500.0, 10.0, 1.0])
    assert expected_cost(y, pred, CostConfig(fn=50, fp=5, fn_use_amount=True), amount) == 550


def test_cost_curve_matches_brute_force():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.1).astype(int)
    p = np.clip(0.5 * y + rng.random(300) * 0.6, 0, 1)
    cost = CostConfig(fn=50, fp=3)
    curve = cost_curve(y, p, cost)
    for t in curve["threshold"].to_numpy()[::17]:
        expected = expected_cost(y, (p >= t).astype(int), cost)
        assert curve.loc[curve["threshold"] == t, "cost"].iloc[0] == pytest.approx(expected)


def test_best_threshold_cost_reacts_to_cost_ratio():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.05).astype(int)
    p = np.clip(0.4 * y + rng.random(2000) * 0.7, 0, 1)
    t_cheap_fn, _ = best_threshold_cost(y, p, CostConfig(fn=2, fp=1))
    t_costly_fn, _ = best_threshold_cost(y, p, CostConfig(fn=200, fp=1))
    assert t_costly_fn < t_cheap_fn


def test_cost_report_savings():
    y = np.array([1, 0, 0, 0])
    r = cost_report(y, np.array([0.9, 0.1, 0.1, 0.1]), 0.5, CostConfig(fn=100, fp=5))
    assert r["cost"] == 0 and r["cost_approve_all"] == 100 and r["savings_vs_approve_all_pct"] == 100
    assert r["savings_vs_best_trivial_pct"] == 100 and r["blocked_pct"] == 25


def test_savings_vs_best_trivial_catches_block_almost_all():
    r = {"cost": 10.0, "cost_approve_all": 1000.0, "cost_block_all": 15.0}
    from fraude.evaluation import savings_vs_best_trivial
    assert round(savings_vs_best_trivial(r), 2) == 33.33


def test_best_threshold_f1_perfect_separation():
    y = np.array([0, 0, 1, 1])
    t = best_threshold_f1(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.2 < t <= 0.8


def test_dal_pozzolo_correction():
    p = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(dal_pozzolo_correction(p, 1.0), p)
    corrected = dal_pozzolo_correction(p, 0.01)
    assert corrected[0] == 0 and corrected[2] == pytest.approx(1)
    assert corrected[1] == pytest.approx(0.01 * 0.5 / (0.01 * 0.5 - 0.5 + 1))
    assert corrected[1] < 0.5


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(2)
    y = (rng.random(500) < 0.1).astype(int)
    p = np.clip(0.5 * y + rng.random(500) * 0.6, 0, 1)
    from sklearn.metrics import average_precision_score

    lo, hi = bootstrap_ci(y, p, n_boot=100)
    assert lo <= average_precision_score(y, p) <= hi

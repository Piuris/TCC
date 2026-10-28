"""Testes das melhorias: duplicatas, calibração, estatística, ensembles e custo por valor."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss

from fraude.calibration import (
    ProbabilityCalibrator,
    choose_method,
    expected_calibration_error,
)
from fraude.config import COST_SCENARIOS, CostConfig
from fraude.data import Dataset
from fraude.ensemble import best_members, build_ensemble, build_from_spec, ensemble_jobs
from fraude.evaluation import cost_report
from fraude.experiments import run_experiments, split_train_test
from fraude.stats import (
    compare_to_best,
    corrected_ttest,
    friedman_nemenyi,
    holm,
    repeated_cv_scores,
    select_parsimonious,
)


def _with_duplicates(ds: Dataset, n: int = 30) -> Dataset:
    X = pd.concat([ds.X, ds.X.iloc[:n].assign(Time=ds.X["Time"].iloc[:n] + 100)], ignore_index=True)
    y = pd.concat([ds.y, ds.y.iloc[:n]], ignore_index=True)
    return Dataset(name="dup", X=X, y=y, time=X["Time"], amount=X["Amount"], kind=ds.kind)


def test_drop_duplicates_ignores_time(small_ds):
    dup = _with_duplicates(small_ds)
    clean = dup.drop_duplicates()
    assert len(clean.y) == len(small_ds.y)
    assert clean.extra["duplicates_removed"] == 30
    assert clean.summary()["duplicates_removed"] == 30
    assert (clean.X["Amount"].to_numpy() == clean.amount.to_numpy()).all()


def test_drop_duplicates_keeps_conflicting_labels(small_ds):
    ds = small_ds.subsample(0.1)
    X = pd.concat([ds.X, ds.X.iloc[:1]], ignore_index=True)
    y = pd.concat([ds.y, pd.Series([1 - ds.y.iloc[0]])], ignore_index=True)
    out = Dataset(name="c", X=X, y=y, time=X["Time"], amount=X["Amount"], kind="creditcard").drop_duplicates()
    assert out.extra["duplicates_removed"] == 0


def test_no_duplicate_crosses_train_test_after_dedupe(small_ds, quick_cfg):
    clean = _with_duplicates(small_ds).drop_duplicates()
    tr, te = split_train_test(clean, quick_cfg)
    cols = [c for c in clean.X.columns if c != "Time"]
    keys_tr = set(map(tuple, clean.X.iloc[tr][cols].round(10).to_numpy()))
    assert not any(tuple(r) in keys_tr for r in clean.X.iloc[te][cols].round(10).to_numpy())


def _overconfident(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.02).astype(int)
    logit = np.where(y == 1, 2.0, -2.0) + rng.normal(0, 1.5, n)
    scores = 1 / (1 + np.exp(-3 * logit))
    return scores, y


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_improves_brier_and_keeps_order(method):
    s, y = _overconfident()
    tr, te = slice(0, 10_000), slice(10_000, None)
    cal = ProbabilityCalibrator(method).fit(s[tr], y[tr])
    p = cal.transform(s[te])
    assert brier_score_loss(y[te], p) < brier_score_loss(y[te], s[te])
    assert expected_calibration_error(y[te], p) < expected_calibration_error(y[te], s[te])
    tol = 1e-9 if method == "sigmoid" else 0.05
    assert average_precision_score(y[te], p) == pytest.approx(average_precision_score(y[te], s[te]), abs=tol)


def test_choose_method_prefers_calibration_for_overconfident_scores():
    s, y = _overconfident()
    method, briers = choose_method(s, y)
    assert method in {"sigmoid", "isotonic"}
    assert briers[method] < briers["none"]


def test_ece_perfect_calibration_is_zero():
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(np.array([0, 0, 1, 1]), p) == 0


def test_corrected_ttest():
    a = np.array([0.80, 0.82, 0.81, 0.79, 0.83])
    t, p = corrected_ttest(a, a, 0.25)
    assert p == 1.0
    t, p = corrected_ttest(a + 0.1, a + np.array([0, 0.001, -0.001, 0.002, 0]), 0.25)
    assert p < 0.05 and t > 0
    from scipy.stats import ttest_rel
    b = a - np.array([0.01, -0.005, 0.02, 0.0, 0.01])
    assert corrected_ttest(a, b, 0.25)[1] > ttest_rel(a, b).pvalue


def test_holm():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)


def _scores():
    rng = np.random.default_rng(0)
    base = rng.normal(0.80, 0.02, 15)
    return pd.DataFrame({
        "melhor": base + 0.010,
        "empatado_rapido": base + 0.008 + rng.normal(0, 0.01, 15),
        "pior": base - 0.15,
    })


def test_friedman_and_compare_to_best():
    scores = _scores()
    fr = friedman_nemenyi(scores)
    assert fr["p_value"] < 0.05 and fr["critical_difference"] > 0
    table = compare_to_best(scores)
    assert table.attrs["best"] == "melhor"
    assert table.loc["melhor", "equivalent_to_best"]
    assert table.loc["empatado_rapido", "equivalent_to_best"]
    assert not table.loc["pior", "equivalent_to_best"]


def test_select_parsimonious_prefers_fastest_equivalent():
    table = compare_to_best(_scores())
    chosen = select_parsimonious(table, {"melhor": 15.0, "empatado_rapido": 2.0, "pior": 0.5})
    assert chosen == "empatado_rapido"


def test_select_parsimonious_practical_margin():
    table = pd.DataFrame({"auc_pr_mean": [0.857, 0.856, 0.824],
                          "equivalent_to_best": [True, True, True]},
                         index=["ensemble", "xgb", "fraco"])
    lat = {"ensemble": 2.9, "xgb": 2.05, "fraco": 1.54}
    assert select_parsimonious(table, lat) == "xgb"
    assert select_parsimonious(table, {"ensemble": 2.0, "xgb": 1.9, "fraco": 1.54}) == "ensemble"


def test_repeated_cv_scores_resumable(small_ds, tmp_path):
    X, y = small_ds.X.iloc[:1500], small_ds.y.iloc[:1500]
    cands = {"lr": {"model": "logreg", "balancing": "none", "params": {}},
             "lgbm": {"model": "lightgbm", "balancing": "none", "params": {"clf__n_estimators": 30}}}
    build = lambda spec: build_from_spec(spec, "creditcard")  # noqa: E731
    out = tmp_path / "s.csv"
    s1 = repeated_cv_scores(build, cands, X, y, n_splits=3, n_repeats=2, out_csv=out)
    assert s1.shape == (6, 2)
    s2 = repeated_cv_scores(lambda spec: 1 / 0, cands, X, y, n_splits=3, n_repeats=2, out_csv=out)
    pd.testing.assert_frame_equal(s1, s2)


def _records():
    return [
        {"model": "xgboost", "balancing": "smote", "best_params": {"clf__n_estimators": 40}, "cv_auc_pr_mean": 0.8},
        {"model": "xgboost", "balancing": "none", "best_params": {"clf__n_estimators": 40}, "cv_auc_pr_mean": 0.7},
        {"model": "lightgbm", "balancing": "class_weight", "best_params": {"clf__n_estimators": 40},
         "cv_auc_pr_mean": 0.75},
        {"model": "random_forest", "balancing": "none", "best_params": {"clf__n_estimators": 20},
         "cv_auc_pr_mean": 0.7},
    ]


def test_best_members():
    members = best_members(_records(), ["xgboost", "lightgbm"])
    assert [(m["model"], m["balancing"]) for m in members] == [("xgboost", "smote"), ("lightgbm", "class_weight")]


def test_ensemble_is_average_of_members(small_ds):
    X, y = small_ds.X, small_ds.y
    members = best_members(_records(), ["xgboost", "lightgbm"])
    ens = build_ensemble(members, "creditcard", pos_weight=24).fit(X, y)
    singles = [build_from_spec({**m, "params": m["params"]}, "creditcard", 24).fit(X, y) for m in members]
    expected = np.mean([s.predict_proba(X.head(50))[:, 1] for s in singles], axis=0)
    np.testing.assert_allclose(ens.predict_proba(X.head(50))[:, 1], expected, rtol=1e-6, atol=1e-9)


def test_run_ensemble_experiment(small_ds, quick_cfg, tmp_path):
    jobs = ensemble_jobs(_records(), "creditcard", pos_weight=24)
    table = run_experiments(small_ds, quick_cfg, tmp_path, jobs=jobs)
    assert set(table["model"]) == {"ensemble_gbdt", "ensemble_arvores"}
    from fraude.reporting import build_report
    build_report(tmp_path)
    assert (tmp_path / "summary.md").exists()


def test_value_scenario_uses_amount():
    assert COST_SCENARIOS["valor_transacao"].fn_use_amount
    y = np.array([1, 1, 0])
    amount = np.array([1000.0, 3.0, 50.0])
    r = cost_report(y, np.array([0.1, 0.9, 0.1]), 0.5, CostConfig(fn=25, fp=5, fn_use_amount=True), amount)
    assert r["cost"] == 1000.0
    assert r["cost_approve_all"] == 1025.0

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from fraude.experiments import run_experiments, select_best, split_train_test
from fraude.models import (
    BALANCING,
    MODELS,
    build_pipeline,
    cost_proportionate_resample,
    is_applicable,
    param_space,
    undersampling_beta,
)

CONFIGS = [(m, b) for m in MODELS for b in BALANCING if is_applicable(m, b)]


@pytest.mark.parametrize("model,balancing", CONFIGS)
def test_every_configuration_trains_and_predicts(small_ds, model, balancing):
    X, y = small_ds.X, small_ds.y
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    pipe = build_pipeline(model, balancing, "creditcard", pos_weight=pos_weight)
    if model == "mlp":
        pipe.set_params(clf__max_iter=30)
    pipe.fit(X.iloc[:2000], y.iloc[:2000])
    proba = pipe.predict_proba(X.iloc[2000:])[:, 1]
    assert proba.shape == (1000,) and np.all((proba >= 0) & (proba <= 1))
    assert average_precision_score(y.iloc[2000:], proba) > y.mean()


def test_sampler_is_not_applied_at_prediction(small_ds):
    """SMOTE/subamostragem só no fit: a predição deve devolver uma linha por entrada."""
    pipe = build_pipeline("random_forest", "undersample", "creditcard").fit(small_ds.X, small_ds.y)
    assert len(pipe.predict_proba(small_ds.X)) == len(small_ds.X)


def test_isolation_forest_rejects_balancing():
    with pytest.raises(ValueError):
        build_pipeline("isolation_forest", "smote", "creditcard")


def test_param_spaces_match_pipeline(small_ds):
    for model, balancing in CONFIGS:
        pipe = build_pipeline(model, balancing, "creditcard")
        valid = pipe.get_params()
        assert set(param_space(model, balancing)) <= set(valid), (model, balancing)


def test_cost_proportionate_resample_balances_classes():
    X = np.arange(1000).reshape(-1, 1)
    y = np.r_[np.ones(20), np.zeros(980)].astype(int)
    Xr, yr = cost_proportionate_resample(X, y, seed=0)
    assert len(yr) == len(y)
    assert 0.4 < yr.mean() < 0.6


def test_undersampling_beta():
    y = np.r_[np.ones(10), np.zeros(1000)]
    assert undersampling_beta(y, 0.1) == pytest.approx(100 / 1000)
    assert undersampling_beta(y, 0.001) == 1.0


def test_paysim_pipeline(paysim_ds):
    pipe = build_pipeline("lightgbm", "class_weight", "paysim", pos_weight=19.0)
    pipe.fit(paysim_ds.X, paysim_ds.y)
    assert average_precision_score(paysim_ds.y, pipe.predict_proba(paysim_ds.X)[:, 1]) > 0.9


def test_split_is_stratified_and_disjoint(small_ds, quick_cfg):
    tr, te = split_train_test(small_ds, quick_cfg)
    assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(small_ds.y)
    assert abs(small_ds.y.iloc[tr].mean() - small_ds.y.iloc[te].mean()) < 0.01


def test_run_experiments_end_to_end(small_ds, quick_cfg, tmp_path):
    quick_cfg.models = ["logreg", "lightgbm", "isolation_forest"]
    quick_cfg.balancing = ["none", "undersample"]
    table = run_experiments(small_ds, quick_cfg, tmp_path)
    assert len(table) == 5
    assert {"auc_pr", "mcc", "latency_p95_ms", "cost"} <= set(table.columns)
    assert (tmp_path / "results.json").exists() and (tmp_path / "test_probas.npz").exists()

    import json
    records = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    under = [r for r in records if r["balancing"] == "undersample"]
    assert all("undersampling_calibration" in r for r in under)
    assert select_best(records)["cv_auc_pr_mean"] == max(r["cv_auc_pr_mean"] for r in records)

    again = run_experiments(small_ds, quick_cfg, tmp_path, resume=True)
    assert len(again) == 5

    from fraude.reporting import build_report
    build_report(tmp_path)
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "figures" / "pr_curves.png").exists()

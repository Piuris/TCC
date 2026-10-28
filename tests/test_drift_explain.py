import numpy as np

from fraude.drift import psi, run_drift, temporal_blocks
from fraude.experiments import split_train_test
from fraude.explain import LocalExplainer, run_explainability
from fraude.models import build_pipeline


def test_temporal_blocks_train_precedes_test():
    train, windows = temporal_blocks(1000, 0.4, 3)
    assert train.max() < min(w.min() for w in windows)
    assert sum(len(w) for w in windows) + len(train) == 1000


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, rng.normal(size=5000)) < 0.05
    assert psi(a, rng.normal(2, 1, size=5000)) > 0.25


def test_run_drift(small_ds, tmp_path):
    df = run_drift(small_ds, "lightgbm", "class_weight", {"clf__n_estimators": 50},
                   out_dir=tmp_path, n_windows=3, label_delay=1)
    assert set(df["strategy"]) == {"estatico", "janela_deslizante", "acumulado"}
    assert len(df) == 9
    w2 = df[df["window"] == 2].set_index("strategy")["auc_pr"]
    assert w2["estatico"] == w2["acumulado"]
    assert (tmp_path / "drift.png").exists()


def test_local_explainer_tree_and_kernel(small_ds):
    X, y = small_ds.X, small_ds.y
    for model in ("lightgbm", "logreg"):
        pipe = build_pipeline(model, "class_weight", "creditcard", pos_weight=24).fit(X, y)
        exp = LocalExplainer(pipe, X.sample(50, random_state=0), nsamples=50)
        factors = exp.top_factors(X.iloc[:2], k=3)
        assert len(factors) == 2 and len(factors[0]) == 3
        assert {"feature", "value", "contribution", "direction"} <= set(factors[0][0])


def test_shap_is_additive_for_tree_model(small_ds):
    """Propriedade de aditividade: base + Σ SHAP = margem do modelo (log-odds)."""
    pipe = build_pipeline("xgboost", "none", "creditcard").set_params(clf__n_estimators=50)
    pipe.fit(small_ds.X, small_ds.y)
    exp = LocalExplainer(pipe, small_ds.X.head(50))
    Xt, vals = exp.shap_values(small_ds.X.head(20))
    margin = pipe.named_steps["clf"].predict(Xt, output_margin=True)
    np.testing.assert_allclose(exp._explainer.expected_value + vals.sum(axis=1), margin, atol=1e-3)


def test_run_explainability(small_ds, quick_cfg, tmp_path):
    tr, te = split_train_test(small_ds, quick_cfg)
    pipe = build_pipeline("random_forest", "class_weight", "creditcard").set_params(clf__n_estimators=50)
    pipe.fit(small_ds.X.iloc[tr], small_ds.y.iloc[tr])
    summary = run_explainability(pipe, small_ds.X.iloc[tr], small_ds.X.iloc[te], small_ds.y.iloc[te],
                                 0.5, tmp_path, n_global=300)
    assert summary["method"] == "shap_tree"
    assert "verdadeiro_positivo" in summary["cases"]
    assert (tmp_path / "shap_beeswarm.png").exists()
    assert (tmp_path / "lime_verdadeiro_positivo.png").exists()

"""Explicabilidade dos modelos (SHAP e LIME) — seção 4.9 da proposta.

As explicações são calculadas no espaço de atributos *após* o pré-processamento
(o que o classificador efetivamente vê). Na base ULB esses atributos são
componentes PCA sem significado semântico; no PaySim, são interpretáveis —
por isso as duas bases são analisadas.
"""

from __future__ import annotations

import json
import logging
import threading
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import SEED  # noqa: E402
from .features import feature_names  # noqa: E402
from .models import IsolationForestDetector  # noqa: E402

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")

TREE_CLASSES = ("RandomForestClassifier", "XGBClassifier", "LGBMClassifier")


def _is_tree(clf) -> bool:
    return type(clf).__name__ in TREE_CLASSES or isinstance(clf, IsolationForestDetector)


def _positive_class(values) -> np.ndarray:
    """Normaliza as diferentes formas de saída do SHAP para (n, f) da classe fraude."""
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


class LocalExplainer:
    """Explicador usado pela API: principais fatores de uma predição individual.

    - Modelos de árvore: ``shap.TreeExplainer`` (exato e rápido, poucos ms).
    - Demais (MLP, regressão logística): ``shap.KernelExplainer`` sobre um
      fundo resumido por k-means — mais lento, por isso com ``nsamples`` baixo.
    """

    def __init__(self, pipeline, background_raw: pd.DataFrame, n_background: int = 20,
                 nsamples: int = 150):
        import shap

        self.pre = pipeline.named_steps["preprocess"]
        self.clf = pipeline.named_steps["clf"]
        self.names = feature_names(self.pre)
        self.nsamples = nsamples
        self._lock = threading.Lock()
        bg = self.pre.transform(background_raw)
        if _is_tree(self.clf):
            model = self.clf.model_ if isinstance(self.clf, IsolationForestDetector) else self.clf
            self.method = "shap_tree"
            self._explainer = shap.TreeExplainer(model)
        else:
            self.method = "shap_kernel"
            summary = shap.kmeans(bg, min(n_background, len(bg)))
            self._explainer = shap.KernelExplainer(lambda x: self.clf.predict_proba(x)[:, 1], summary)

    def shap_values(self, X_raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xt = self.pre.transform(X_raw)
        with self._lock:
            if self.method == "shap_tree":
                vals = self._explainer.shap_values(Xt)
            else:
                vals = self._explainer.shap_values(Xt, nsamples=self.nsamples, silent=True)
        vals = _positive_class(vals)
        if isinstance(self.clf, IsolationForestDetector):
            vals = -vals
        return Xt, vals

    def top_factors(self, X_raw: pd.DataFrame, k: int = 5) -> list[list[dict]]:
        Xt, vals = self.shap_values(X_raw)
        out = []
        for row_x, row_v in zip(Xt, vals):
            order = np.argsort(-np.abs(row_v))[:k]
            out.append([
                {"feature": self.names[i], "value": float(row_x[i]),
                 "contribution": float(row_v[i]),
                 "direction": "aumenta_risco" if row_v[i] > 0 else "reduz_risco"}
                for i in order
            ])
        return out


def _pick_cases(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, int]:
    """Casos representativos para a discussão: VP, FP e FN de maior relevância."""
    pred = proba >= threshold
    cases = {}
    for label, mask, sign in [("verdadeiro_positivo", (y == 1) & pred, -1),
                              ("falso_positivo", (y == 0) & pred, -1),
                              ("falso_negativo", (y == 1) & ~pred, -1)]:
        idx = np.flatnonzero(mask)
        if len(idx):
            cases[label] = int(idx[np.argsort(sign * proba[idx])[0]])
    return cases


def run_explainability(pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test,
                       threshold: float, out_dir: Path, n_global: int = 2000,
                       seed: int = SEED, calibrator=None) -> dict:
    """Gera a análise global (SHAP) e local (SHAP + LIME) com figuras.

    ``threshold`` está na escala do modelo servido; se houver ``calibrator``,
    ele é aplicado antes de comparar com o limiar. A calibração é monotônica,
    então não altera a ordem das transações nem as atribuições do SHAP.
    """
    import shap
    from lime.lime_tabular import LimeTabularExplainer

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    y_test = np.asarray(y_test).astype(int)

    background = X_train.iloc[rng.choice(len(X_train), size=min(500, len(X_train)), replace=False)]
    explainer = LocalExplainer(pipeline, background)
    names = explainer.names

    fraud_idx = np.flatnonzero(y_test == 1)
    legit_idx = np.flatnonzero(y_test == 0)
    n_global = n_global if explainer.method == "shap_tree" else min(n_global, 300)
    n_legit = max(0, min(len(legit_idx), n_global - len(fraud_idx)))
    sel = np.concatenate([fraud_idx[:n_global], rng.choice(legit_idx, n_legit, replace=False)])
    Xt, vals = explainer.shap_values(X_test.iloc[sel])

    importance = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(vals).mean(axis=0)}) \
        .sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(out_dir / "shap_global_importance.csv", index=False)

    plt.figure()
    shap.summary_plot(vals, Xt, feature_names=names, show=False, max_display=15)
    plt.title("SHAP — impacto dos atributos na predição de fraude")
    plt.savefig(out_dir / "shap_beeswarm.png", bbox_inches="tight", dpi=150)
    plt.close("all")
    plt.figure()
    shap.summary_plot(vals, Xt, feature_names=names, plot_type="bar", show=False,
                      max_display=15, color="#2a78d6")
    plt.title("SHAP — importância global (média de |SHAP|)")
    plt.savefig(out_dir / "shap_bar.png", bbox_inches="tight", dpi=150)
    plt.close("all")

    proba = pipeline.predict_proba(X_test)[:, 1]
    if calibrator is not None:
        proba = calibrator.transform(proba)
    cases = _pick_cases(y_test, proba, threshold)
    bg_t = explainer.pre.transform(background)
    lime_exp = LimeTabularExplainer(bg_t, feature_names=names, class_names=["legítima", "fraude"],
                                    discretize_continuous=True, random_state=seed, mode="classification")
    clf = pipeline.named_steps["clf"]

    local = {}
    for label, i in cases.items():
        row = X_test.iloc[[i]]
        row_t, row_v = explainer.shap_values(row)
        expected = explainer._explainer.expected_value
        expected = float(np.ravel(expected)[-1])
        if isinstance(clf, IsolationForestDetector):
            expected = -expected
        exp = shap.Explanation(values=row_v[0], base_values=expected, data=row_t[0], feature_names=names)
        plt.figure()
        shap.plots.waterfall(exp, max_display=12, show=False)
        plt.title(f"SHAP — {label.replace('_', ' ')} (p={proba[i]:.3f})")
        plt.savefig(out_dir / f"shap_waterfall_{label}.png", bbox_inches="tight", dpi=150)
        plt.close("all")

        le = lime_exp.explain_instance(row_t[0], clf.predict_proba, num_features=10)
        fig = le.as_pyplot_figure()
        fig.suptitle(f"LIME — {label.replace('_', ' ')} (p={proba[i]:.3f})")
        fig.savefig(out_dir / f"lime_{label}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

        shap_top = [names[j] for j in np.argsort(-np.abs(row_v[0]))[:5]]
        lime_top = [names[j] for j, _ in sorted(le.as_map()[1], key=lambda t: -abs(t[1]))[:5]]
        local[label] = {
            "test_index": i,
            "probability": float(proba[i]),
            "true_label": int(y_test[i]),
            "shap_top5": shap_top,
            "lime_top5": lime_top,
            "lime_rules": le.as_list(),
            "top5_overlap": len(set(shap_top) & set(lime_top)),
        }

    summary = {
        "method": explainer.method,
        "global_top10": importance.head(10).to_dict(orient="records"),
        "cases": local,
    }
    (out_dir / "explainability.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    return summary

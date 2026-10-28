"""Fábrica de modelos e estratégias de tratamento do desbalanceamento.

Cada configuração experimental é o par (algoritmo, balanceamento). O resultado
de :func:`build_pipeline` é um ``imblearn.Pipeline``::

    preprocess  ->  sampler (só no fit)  ->  clf

O ``imblearn.Pipeline`` aplica a reamostragem apenas durante o ``fit``; na
predição o *sampler* é ignorado. Assim, dentro da validação cruzada, SMOTE e
subamostragem tocam somente a partição de treino (seção 4.4 da proposta).
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn import FunctionSampler
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from .config import SEED
from .features import build_preprocessor

SUPERVISED = ["logreg", "random_forest", "xgboost", "lightgbm", "mlp"]
UNSUPERVISED = ["isolation_forest"]
MODELS = SUPERVISED + UNSUPERVISED
BALANCING = ["none", "smote", "undersample", "class_weight"]
TREE_MODELS = {"random_forest", "xgboost", "lightgbm", "isolation_forest"}

MODEL_LABELS = {
    "logreg": "Regressão Logística (referência)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "mlp": "MLP (rede neural)",
    "isolation_forest": "Isolation Forest (não supervisionado)",
}
BALANCING_LABELS = {
    "none": "Sem tratamento",
    "smote": "SMOTE",
    "undersample": "Subamostragem aleatória",
    "class_weight": "Sensível a custo (pesos)",
}


class IsolationForestDetector(BaseEstimator, ClassifierMixin):
    """Isolation Forest (Liu; Ting; Zhou, 2008) exposto como classificador.

    O treino ignora os rótulos. O escore de anomalia (``-score_samples``) é
    normalizado para [0, 1] com mínimo e máximo observados no treino, de modo
    que possa ser tratado como "probabilidade" nas mesmas métricas e na mesma
    escolha de limiar dos modelos supervisionados. Os rótulos só são usados,
    fora do ``fit``, para avaliar e escolher hiperparâmetros.
    """

    def __init__(self, n_estimators=200, max_samples="auto", max_features=1.0,
                 random_state=SEED, n_jobs=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators, max_samples=self.max_samples,
            max_features=self.max_features, random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(X)
        raw = -self.model_.score_samples(X)
        self.score_min_, self.score_max_ = float(raw.min()), float(raw.max())
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        return -self.model_.score_samples(X)

    def predict_proba(self, X):
        span = max(self.score_max_ - self.score_min_, 1e-12)
        p = np.clip((self.decision_function(X) - self.score_min_) / span, 0.0, 1.0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def cost_proportionate_resample(X, y, seed: int = SEED):
    """Reamostragem proporcional ao custo (Zadrozny; Langford; Abe, 2003).

    O ``MLPClassifier`` do scikit-learn não aceita ``sample_weight``. Sortear o
    treino com reposição, com probabilidade proporcional ao peso de classe
    "balanced", é equivalente *em esperança* a minimizar a perda ponderada —
    que é o que ``class_weight`` faz nos demais algoritmos. O tamanho total
    do treino é mantido.
    """
    y = np.asarray(y)
    n, classes = len(y), np.unique(y)
    weights = np.empty(n, dtype=float)
    for c in classes:
        mask = y == c
        weights[mask] = n / (len(classes) * mask.sum())
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n, replace=True, p=weights / weights.sum())
    return X[idx], y[idx]


def is_applicable(model: str, balancing: str) -> bool:
    """Isolation Forest não usa rótulos no treino: balanceamento não se aplica."""
    if model in UNSUPERVISED:
        return balancing == "none"
    return True


def _make_classifier(model: str, balancing: str, pos_weight: float, seed: int):
    weighted = balancing == "class_weight"
    if model == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced" if weighted else None,
                                  random_state=seed)
    if model == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed,
            class_weight="balanced_subsample" if weighted else None,
        )
    if model == "xgboost":
        return XGBClassifier(
            n_estimators=400, tree_method="hist", eval_metric="aucpr", n_jobs=-1,
            random_state=seed, scale_pos_weight=pos_weight if weighted else 1.0,
        )
    if model == "lightgbm":
        return LGBMClassifier(
            n_estimators=400, n_jobs=-1, random_state=seed, verbose=-1,
            subsample_freq=1, scale_pos_weight=pos_weight if weighted else 1.0,
        )
    if model == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200,
            batch_size=512, random_state=seed,
        )
    if model == "isolation_forest":
        return IsolationForestDetector(random_state=seed, n_jobs=-1)
    raise ValueError(f"Modelo desconhecido: {model}. Opções: {MODELS}")


def _make_sampler(model: str, balancing: str, seed: int):
    if balancing == "smote":
        return SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=seed)
    if balancing == "undersample":
        return RandomUnderSampler(sampling_strategy=0.1, random_state=seed)
    if balancing == "class_weight" and model == "mlp":
        return FunctionSampler(func=cost_proportionate_resample, kw_args={"seed": seed},
                               validate=False)
    return "passthrough"


def build_pipeline(model: str, balancing: str, kind: str, pos_weight: float = 1.0,
                   seed: int = SEED) -> ImbPipeline:
    """Monta o pipeline completo de uma configuração experimental.

    ``pos_weight`` = negativos / positivos no treino; usado como
    ``scale_pos_weight`` no XGBoost e no LightGBM quando ``balancing='class_weight'``.
    """
    if not is_applicable(model, balancing):
        raise ValueError(f"Balanceamento '{balancing}' não se aplica a '{model}'.")
    if balancing not in BALANCING:
        raise ValueError(f"Balanceamento desconhecido: {balancing}. Opções: {BALANCING}")
    return ImbPipeline([
        ("preprocess", build_preprocessor(kind)),
        ("sampler", _make_sampler(model, balancing, seed)),
        ("clf", _make_classifier(model, balancing, pos_weight, seed)),
    ])


def param_space(model: str, balancing: str, minority_ratio: float = 0.0) -> dict:
    """Espaço da busca aleatória, com faixas usuais da literatura.

    ``minority_ratio`` (positivos / negativos no treino) descarta proporções de
    reamostragem que não fariam sentido — menores que a proporção original.
    """
    spaces = {
        "logreg": {"clf__C": loguniform(1e-3, 1e2)},
        "random_forest": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 8, 16, 24],
            "clf__min_samples_leaf": [1, 2, 5, 10],
            "clf__max_features": ["sqrt", 0.3, 0.5],
        },
        "xgboost": {
            "clf__n_estimators": randint(200, 700),
            "clf__max_depth": randint(3, 9),
            "clf__learning_rate": loguniform(0.02, 0.3),
            "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.6, 0.4),
            "clf__min_child_weight": [1, 3, 5, 10],
        },
        "lightgbm": {
            "clf__n_estimators": randint(200, 700),
            "clf__num_leaves": randint(15, 128),
            "clf__learning_rate": loguniform(0.02, 0.3),
            "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.6, 0.4),
            "clf__min_child_samples": randint(10, 100),
        },
        "mlp": {
            "clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
            "clf__alpha": loguniform(1e-5, 1e-2),
            "clf__learning_rate_init": loguniform(1e-4, 1e-2),
        },
        "isolation_forest": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_samples": [256, 1024, 4096],
            "clf__max_features": [0.5, 0.75, 1.0],
        },
    }
    space = dict(spaces[model])
    ratios = {"smote": [0.05, 0.1, 0.25, 0.5], "undersample": [0.01, 0.05, 0.1, 0.5]}
    if balancing in ratios:
        valid = [r for r in ratios[balancing] if r > minority_ratio] or [1.0]
        space["sampler__sampling_strategy"] = valid
    return space


def search_n_jobs(model: str) -> int:
    """Evita paralelismo aninhado: bibliotecas de árvores já usam todos os núcleos."""
    return 1 if model in {"random_forest", "xgboost", "lightgbm", "isolation_forest"} else -1


def undersampling_beta(y, sampling_strategy: float) -> float:
    """β de Dal Pozzolo et al. (2015): probabilidade de um negativo ser mantido."""
    y = np.asarray(y)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    kept_neg = min(n_neg, n_pos / sampling_strategy)
    return kept_neg / n_neg

"""Calibração de probabilidades.

Modelos de boosting treinados sobre bases muito desbalanceadas (e ainda mais
após SMOTE) produzem escores concentrados perto de 0 e 1: ordenam bem as
transações (AUC-PR alta), mas o número devolvido não é uma probabilidade
confiável. Calibrar torna a "probabilidade de fraude" da API interpretável e o
limiar de decisão mais estável.

O calibrador é ajustado sobre as predições *out-of-fold* do treino (as mesmas
usadas para escolher o limiar), e o método — Platt (sigmoide) ou regressão
isotônica — é escolhido por validação cruzada interna sobre essas predições,
pelo menor Brier. O teste não participa de nenhuma escolha.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from .config import SEED  # noqa: E402

METHODS = ("none", "sigmoid", "isotonic")
EPS = 1e-7


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


class ProbabilityCalibrator:
    """Mapeamento monotônico escore → probabilidade calibrada."""

    def __init__(self, method: str = "sigmoid"):
        if method not in METHODS:
            raise ValueError(f"Método desconhecido: {method}. Opções: {METHODS}")
        self.method = method

    def fit(self, scores, y) -> "ProbabilityCalibrator":
        y = np.asarray(y).astype(int)
        if self.method == "sigmoid":
            self.model_ = LogisticRegression(C=1e4, max_iter=1000).fit(_logit(scores), y)
        elif self.method == "isotonic":
            self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
                np.asarray(scores, dtype=float), y)
        return self

    def transform(self, scores) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self.method == "sigmoid":
            return self.model_.predict_proba(_logit(scores))[:, 1]
        if self.method == "isotonic":
            return self.model_.predict(scores)
        return scores


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """ECE com faixas de igual largura, ponderado pelo nº de exemplos na faixa."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_metrics(y, p) -> dict:
    """Brier, log loss e ECE — e a AUC-PR, pois a isotônica (em degraus) cria
    empates nos escores e pode reduzir um pouco a capacidade de ordenação."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p)),
            "ece": expected_calibration_error(y, p),
            "auc_pr": float(average_precision_score(y, p))}


def choose_method(scores, y, n_splits: int = 5, seed: int = SEED) -> tuple[str, dict]:
    """Escolhe o método pelo menor Brier em CV interna sobre as predições OOF."""
    scores, y = np.asarray(scores, dtype=float), np.asarray(y).astype(int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    briers = {}
    for method in METHODS:
        vals = []
        for tr, va in cv.split(scores.reshape(-1, 1), y):
            cal = ProbabilityCalibrator(method).fit(scores[tr], y[tr])
            vals.append(brier_score_loss(y[va], np.clip(cal.transform(scores[va]), 0, 1)))
        briers[method] = float(np.mean(vals))
    return min(briers, key=briers.get), briers


def adaptive_bins(y, p, n_bins: int = 10, min_pos: int = 5) -> list[tuple[float, float, int]]:
    """Faixas com o mesmo número de fraudes, do maior escore para o menor.

    Com fraudes raras, faixas de largura fixa ficam quase todas sem nenhuma
    fraude (taxa observada 0, impossível de mostrar em escala log). Dividir os
    positivos igualmente entre ~``n_bins`` faixas (no mínimo ``min_pos`` por
    faixa) dá taxas observadas estáveis em toda a escala. Devolve
    (probabilidade média, taxa observada, nº de transações) por faixa.
    """
    order = np.argsort(-np.asarray(p, dtype=float), kind="stable")
    p, y = np.asarray(p, dtype=float)[order], np.asarray(y).astype(int)[order]
    per_bin = max(min_pos, int(y.sum()) // n_bins)
    bins, start, pos = [], 0, 0
    for i in range(len(p)):
        pos += y[i]
        if pos >= per_bin:
            bins.append((start, i + 1))
            start, pos = i + 1, 0
    if start < len(p):
        if bins:
            bins[-1] = (bins[-1][0], len(p))
        else:
            bins.append((start, len(p)))
    return [(float(p[a:b].mean()), float(y[a:b].mean()), b - a) for a, b in bins]


def reliability_plot(y, curves: dict[str, np.ndarray], path: Path, title: str = "") -> None:
    """Diagrama de confiabilidade em escala log (as fraudes vivem perto de 0)."""
    y = np.asarray(y).astype(int)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([1e-5, 1], [1e-5, 1], color="#52514e", linestyle="--", linewidth=1,
            label="Calibração perfeita")
    for (name, p), color in zip(curves.items(), colors):
        p = np.asarray(p, dtype=float)
        pts = adaptive_bins(y, p)
        xs = [max(x, 1e-5) for x, _, _ in pts]
        ys = [r for _, r, _ in pts]
        ece = expected_calibration_error(y, p)
        ax.plot(xs, ys, marker="o", markersize=7, color=color, markeredgecolor="white",
                markeredgewidth=2, label=f"{name} (ECE={ece:.5f})")
    ax.set(xscale="log", yscale="log", xlim=(1e-5, 1.05), ylim=(1e-5, 1.05),
           xlabel="Probabilidade prevista (média da faixa)",
           ylabel="Taxa de fraude observada (faixas com nº igual de fraudes)",
           title=title or "Diagrama de confiabilidade (teste)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)

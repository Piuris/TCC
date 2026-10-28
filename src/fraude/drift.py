"""Análise exploratória de *concept drift* (seção 4.10 da proposta).

Validação temporal: o modelo é treinado sobre um período inicial e avaliado em
janelas subsequentes, sempre com o treino precedendo o teste. Três estratégias
são comparadas:

- ``estatico``         : treina uma vez no período inicial e nunca mais.
- ``janela_deslizante``: antes de cada janela, retreina com o bloco mais recente
                         de mesmo tamanho do treino inicial.
- ``acumulado``        : antes de cada janela, retreina com todo o histórico.

``label_delay`` simula o atraso na confirmação das fraudes (Dal Pozzolo et al.,
2018): com atraso 1, a janela imediatamente anterior ainda não tem rótulos e
não entra no retreinamento.

Além das métricas, o PSI (Population Stability Index) do escore e do valor
mede a mudança na distribuição de entrada (*virtual drift*).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from .config import SEED  # noqa: E402
from .data import Dataset  # noqa: E402
from .evaluation import bootstrap_ci  # noqa: E402
from .ensemble import build_from_spec  # noqa: E402

log = logging.getLogger(__name__)

STRATEGY_COLORS = {"estatico": "#2a78d6", "janela_deslizante": "#eb6834", "acumulado": "#1baf7a"}
STRATEGY_LABELS = {"estatico": "Estático", "janela_deslizante": "Janela deslizante",
                   "acumulado": "Acumulado"}


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index. < 0,1 estável; 0,1–0,25 moderado; > 0,25 relevante."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-6, None), np.clip(cur, 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def temporal_blocks(n: int, train_frac: float, n_windows: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Índices do treino inicial e das janelas seguintes (dados já ordenados no tempo)."""
    n_train = int(n * train_frac)
    train = np.arange(n_train)
    windows = np.array_split(np.arange(n_train, n), n_windows)
    return train, windows


def run_drift(ds: Dataset, model: str = "lightgbm", balancing: str = "class_weight",
              params: dict | None = None, out_dir: Path | None = None, train_frac: float = 0.4,
              n_windows: int = 6, label_delay: int = 0, seed: int = SEED,
              members: list[dict] | None = None) -> pd.DataFrame:
    params = params or {}
    spec = {"model": model, "balancing": balancing, "params": params, "members": members}
    X, y, amount = ds.X, ds.y.to_numpy(), ds.amount.to_numpy()
    train_idx, windows = temporal_blocks(len(y), train_frac, n_windows)

    def fit(idx):
        y_tr = y[idx]
        pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        pipe = build_from_spec(spec, ds.kind, pos_weight=pos_weight, seed=seed)
        return pipe.fit(X.iloc[idx], y_tr)

    static = fit(train_idx)
    ref_scores = static.predict_proba(X.iloc[train_idx])[:, 1]
    rows = []
    for w, w_idx in enumerate(windows):
        known_end = (train_idx[-1] + 1) if w - label_delay <= 0 else windows[w - label_delay - 1][-1] + 1
        history = np.arange(known_end)
        has_new_labels = known_end > len(train_idx)
        models = {
            "estatico": static,
            "janela_deslizante": fit(history[-len(train_idx):]) if has_new_labels else static,
            "acumulado": fit(history) if has_new_labels else static,
        }
        y_w = y[w_idx]
        t = ds.time.iloc[w_idx]
        for name, mdl in models.items():
            p = mdl.predict_proba(X.iloc[w_idx])[:, 1]
            has_both = 0 < y_w.sum() < len(y_w)
            lo, hi = bootstrap_ci(y_w, p, n_boot=200, seed=seed) if has_both else (np.nan, np.nan)
            rows.append({
                "window": w + 1,
                "time_start": float(t.iloc[0]),
                "time_end": float(t.iloc[-1]),
                "strategy": name,
                "n": int(len(w_idx)),
                "n_fraud": int(y_w.sum()),
                "fraud_rate_pct": 100 * float(y_w.mean()),
                "auc_pr": average_precision_score(y_w, p) if has_both else np.nan,
                "auc_pr_ci_low": lo,
                "auc_pr_ci_high": hi,
                "auc_roc": roc_auc_score(y_w, p) if has_both else np.nan,
                "psi_score": psi(ref_scores, static.predict_proba(X.iloc[w_idx])[:, 1]),
                "psi_amount": psi(amount[train_idx], amount[w_idx]),
                "train_size": int(len(train_idx) if mdl is static else
                                  len(train_idx) if name == "janela_deslizante" else len(history)),
            })
        log.info("janela %d/%d: fraudes=%d  AUC-PR estático=%.4f", w + 1, n_windows, int(y_w.sum()),
                 rows[-3]["auc_pr"])

    df = pd.DataFrame(rows)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "drift.csv", index=False)
        (out_dir / "drift_config.json").write_text(json.dumps({
            "model": model, "balancing": balancing, "params": params, "members": members,
            "train_frac": train_frac,
            "n_windows": n_windows, "label_delay": label_delay, "seed": seed,
        }, indent=2, default=str), encoding="utf-8")
        plot_drift(df, out_dir / "drift.png", ds.name)
    return df


def plot_drift(df: pd.DataFrame, path: Path, title: str) -> None:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.2]})
    for s in STRATEGY_LABELS:
        d = df[df["strategy"] == s]
        ax.plot(d["window"], d["auc_pr"], marker="o", markersize=7, color=STRATEGY_COLORS[s],
                markeredgecolor="white", markeredgewidth=2, label=STRATEGY_LABELS[s])
        if s == "estatico":
            ax.fill_between(d["window"], d["auc_pr_ci_low"], d["auc_pr_ci_high"],
                            color=STRATEGY_COLORS[s], alpha=0.15, linewidth=0,
                            label="IC 95% (estático, bootstrap)")
    ax.set(ylabel="AUC-PR", ylim=(0, 1.02),
           title=f"Concept drift — desempenho por janela temporal ({title})")
    ax.legend(fontsize=8, loc="lower left")
    d = df[df["strategy"] == "estatico"]
    ax2.bar(d["window"], d["n_fraud"], color="#86b6ef", width=0.6)
    ax2.set(xlabel="Janela temporal (após o período de treino)", ylabel="nº fraudes")
    ax2.set_xticks(d["window"])
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)

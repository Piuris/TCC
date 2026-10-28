"""Comparação estatística entre configurações.

Com 98 fraudes no teste, diferenças de 0,01 em AUC-PR estão dentro do ruído.
Em vez de comparar médias, as configurações candidatas são reavaliadas com
validação cruzada repetida (5 partições × 3 repetições, mesmas partições para
todas, só no treino) e comparadas com dois procedimentos:

- **Friedman + Nemenyi** (Demšar, 2006): ranking médio de cada configuração
  nas 15 partições e a diferença crítica (CD). Muito usado, mas pressupõe
  blocos independentes, o que partições da mesma base não são; fica como
  visão geral.
- **Teste t corrigido para CV repetida** (Nadeau; Bengio, 2003; Bouckaert;
  Frank, 2004): compara cada configuração com a melhor, corrigindo a variância
  pela sobreposição entre os treinos das partições. Valores-p ajustados por
  Holm. É o teste usado para a decisão.

Regra de seleção (parcimônia): entre as configurações *não* significativamente
piores que a melhor, escolhe-se a de menor latência de inferência — um ganho
estatisticamente indistinguível não justifica um modelo mais caro.
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
from scipy import stats as st  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score  # noqa: E402

from .config import SEED  # noqa: E402

log = logging.getLogger(__name__)
ALPHA = 0.05


def repeated_cv_scores(build, candidates: dict[str, dict], X: pd.DataFrame, y: pd.Series,
                       n_splits: int = 5, n_repeats: int = 3, seed: int = SEED,
                       out_csv: Path | None = None) -> pd.DataFrame:
    """AUC-PR de cada candidato em cada partição (linhas) — retomável via ``out_csv``.

    ``build(spec)`` devolve o pipeline não treinado de um candidato.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = pd.read_csv(out_csv) if out_csv is not None and out_csv.exists() else pd.DataFrame()
    for i, (key, spec) in enumerate(candidates.items(), 1):
        if key in scores.columns:
            log.info("[%d/%d] %s já avaliado — pulando", i, len(candidates), key)
            continue
        log.info("[%d/%d] CV repetida: %s ...", i, len(candidates), key)
        vals = cross_val_score(build(spec), X, y, cv=cv, scoring="average_precision", n_jobs=1)
        scores[key] = vals
        log.info("    AUC-PR = %.4f ± %.4f", vals.mean(), vals.std())
        if out_csv is not None:
            scores.to_csv(out_csv, index=False)
    return scores[list(candidates)]


def friedman_nemenyi(scores: pd.DataFrame, alpha: float = ALPHA) -> dict:
    k, n = scores.shape[1], scores.shape[0]
    stat, p = st.friedmanchisquare(*[scores[c].to_numpy() for c in scores.columns])
    ranks = scores.rank(axis=1, ascending=False).mean().sort_values()
    q = st.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = float(q * np.sqrt(k * (k + 1) / (6 * n)))
    return {"statistic": float(stat), "p_value": float(p), "critical_difference": cd,
            "average_ranks": ranks.round(4).to_dict()}


def corrected_ttest(a, b, test_train_ratio: float) -> tuple[float, float]:
    """Teste t corrigido de Nadeau e Bengio para CV repetida (bicaudal)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    k = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return (0.0, 1.0) if d.mean() == 0 else (np.inf * np.sign(d.mean()), 0.0)
    t = d.mean() / np.sqrt((1 / k + test_train_ratio) * var)
    return float(t), float(2 * st.t.sf(abs(t), df=k - 1))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Ajuste de Holm-Bonferroni (controla o erro por família)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[key] = running
    return out


def compare_to_best(scores: pd.DataFrame, n_splits: int = 5, alpha: float = ALPHA) -> pd.DataFrame:
    means = scores.mean()
    best = means.idxmax()
    ratio = 1 / (n_splits - 1)
    raw = {}
    tstats = {}
    for c in scores.columns:
        if c != best:
            tstats[c], raw[c] = corrected_ttest(scores[best], scores[c], ratio)
    adjusted = holm(raw)
    ranks = scores.rank(axis=1, ascending=False).mean()
    table = pd.DataFrame({
        "auc_pr_mean": means,
        "auc_pr_std": scores.std(),
        "avg_rank": ranks,
        "t_vs_best": pd.Series(tstats),
        "p_vs_best": pd.Series(raw),
        "p_holm": pd.Series(adjusted),
    })
    table.loc[best, ["t_vs_best", "p_vs_best", "p_holm"]] = [0.0, 1.0, 1.0]
    table["equivalent_to_best"] = table["p_holm"] >= alpha
    table.attrs["best"] = best
    return table.sort_values("auc_pr_mean", ascending=False)


PRACTICAL_MARGIN = 0.01
LATENCY_TOL_MS = 0.5


def select_parsimonious(table: pd.DataFrame, latency_ms: dict[str, float],
                        margin: float = PRACTICAL_MARGIN,
                        latency_tol_ms: float = LATENCY_TOL_MS) -> str:
    """Seleção por parcimônia com margem prática.

    Só o teste estatístico não basta: com 15 partições ele tem pouco poder e
    declara "equivalentes" configurações vários pontos de AUC-PR abaixo da
    melhor. Por isso a candidata precisa ser (i) não significativamente pior que
    a melhor *e* (ii) estar a no máximo ``margin`` da média da melhor. Entre
    essas, as de latência até ``latency_tol_ms`` acima da mais rápida contam
    como igualmente rápidas, e vence a de maior AUC-PR média.
    """
    best_mean = float(table["auc_pr_mean"].max())
    eq = table[table["equivalent_to_best"] & (table["auc_pr_mean"] >= best_mean - margin)].copy()
    eq["latency_ms"] = [latency_ms.get(k, np.inf) for k in eq.index]
    fast = eq[eq["latency_ms"] <= eq["latency_ms"].min() + latency_tol_ms]
    return fast.sort_values(["auc_pr_mean", "latency_ms"], ascending=[False, True]).index[0]


def plot_ranks(friedman: dict, table: pd.DataFrame, labels: dict[str, str], path: Path) -> None:
    """Ranking médio com a diferença crítica de Nemenyi e o resultado do teste t corrigido."""
    ranks = pd.Series(friedman["average_ranks"]).sort_values()
    cd = friedman["critical_difference"]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(ranks) + 1.6))
    ys = np.arange(len(ranks))[::-1]
    for y_, (key, r) in zip(ys, ranks.items()):
        eq = bool(table.loc[key, "equivalent_to_best"])
        color = "#2a78d6" if eq else "#b5b4ae"
        ax.plot([r - cd / 2, r + cd / 2], [y_, y_], color=color, linewidth=2, alpha=0.6)
        ax.plot([r], [y_], "o", markersize=7, color=color, markeredgecolor="white", markeredgewidth=2)
    ax.set_yticks(ys, [labels.get(k, k) for k in ranks.index], fontsize=8)
    ax.axvline(ranks.iloc[0] + cd / 2, color="#52514e", linestyle=":", linewidth=1)
    ax.set_xlim(0.5, len(ranks) + 0.5)
    ax.set_xlabel(f"Ranking médio na CV repetida (menor = melhor) — barras = ±CD/2, CD={cd:.2f}")
    ax.set_title("Comparação estatística das configurações\n"
                 "azul = não significativamente pior que a melhor (t corrigido + Holm, α=0,05)")
    ax.grid(axis="y", visible=False)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

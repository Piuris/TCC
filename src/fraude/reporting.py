"""Tabelas e figuras do capítulo de resultados."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from .config import COST_SCENARIOS  # noqa: E402
from .ensemble import ENSEMBLES  # noqa: E402
from .evaluation import cost_curve, expected_cost, savings_vs_best_trivial  # noqa: E402
from .models import BALANCING, BALANCING_LABELS, MODEL_LABELS, MODELS  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ALL_MODELS = MODELS + list(ENSEMBLES)
MODEL_COLORS = dict(zip(ALL_MODELS, SERIES))
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_2, "ytick.color": INK_2, "lines.linewidth": 2,
    "legend.frameon": False, "font.size": 10,
})


def _key(r: dict) -> str:
    return f"{r['model']}__{r['balancing']}"


def _best_per_model(records: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for r in records:
        if r["model"] not in best or r["cv_auc_pr_mean"] > best[r["model"]]["cv_auc_pr_mean"]:
            best[r["model"]] = r
    return [best[m] for m in ALL_MODELS if m in best]


def plot_auc_pr_heatmap(records: list[dict], path: Path) -> None:
    records = [r for r in records if r["balancing"] in BALANCING]
    if not records:
        return
    models = [m for m in MODELS if any(r["model"] == m for r in records)]
    bals = list(BALANCING_LABELS)
    grid = np.full((len(models), len(bals)), np.nan)
    for r in records:
        grid[models.index(r["model"]), bals.index(r["balancing"])] = \
            r["test_metrics"]["default"]["auc_pr"]
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(models) + 1.5))
    im = ax.imshow(grid, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(bals)), [BALANCING_LABELS[b] for b in bals], rotation=15)
    ax.set_yticks(range(len(models)), [MODEL_LABELS[m] for m in models])
    for i in range(len(models)):
        for j in range(len(bals)):
            if not np.isnan(grid[i, j]):
                v = grid[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v > 0.55 else INK, fontsize=9)
            else:
                ax.text(j, i, "n/a", ha="center", va="center", color=INK_2, fontsize=8)
    fig.colorbar(im, ax=ax, label="AUC-PR (teste)")
    ax.set_title("AUC-PR por algoritmo e técnica de balanceamento")
    fig.savefig(path)
    plt.close(fig)


def plot_curves(records: list[dict], probas: dict, y_test: np.ndarray, out_dir: Path) -> None:
    best = _best_per_model(records)
    prevalence = y_test.mean()

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for r in best:
        p, rc, _ = precision_recall_curve(y_test, probas[_key(r)])
        ax.plot(rc, p, color=MODEL_COLORS[r["model"]],
                label=f"{MODEL_LABELS[r['model']]} — {BALANCING_LABELS[r['balancing']]} "
                      f"(AP={r['test_metrics']['default']['auc_pr']:.3f})")
    ax.axhline(prevalence, color=INK_2, linestyle="--", linewidth=1,
               label=f"Classificador aleatório (AP={prevalence:.4f})")
    ax.set(xlabel="Revocação", ylabel="Precisão", xlim=(0, 1), ylim=(0, 1.02),
           title="Curvas precisão-revocação (melhor configuração por algoritmo)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.13))
    fig.savefig(out_dir / "pr_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for r in best:
        fpr, tpr, _ = roc_curve(y_test, probas[_key(r)])
        ax.plot(fpr, tpr, color=MODEL_COLORS[r["model"]],
                label=f"{MODEL_LABELS[r['model']]} (AUC={r['test_metrics']['default']['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], color=INK_2, linestyle="--", linewidth=1)
    ax.set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos",
           title="Curvas ROC (melhor configuração por algoritmo)")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(out_dir / "roc_curves.png")
    plt.close(fig)


def plot_cost_curves(records: list[dict], probas: dict, y_test, amount_test, out_dir: Path,
                     scenario: str = "moderado") -> None:
    cost = COST_SCENARIOS[scenario]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for r in _best_per_model(records):
        curve = cost_curve(y_test, probas[_key(r)], cost, amount_test)
        curve = curve[np.isfinite(curve["threshold"])]
        grid = np.linspace(0, 1, 401)
        idx = np.clip(np.searchsorted(curve["threshold"].to_numpy(), grid), 0, len(curve) - 1)
        ax.plot(grid, curve["cost"].to_numpy()[idx], color=MODEL_COLORS[r["model"]],
                label=MODEL_LABELS[r["model"]])
        thr = r["thresholds"][f"cost_{scenario}"]
        ax.plot([thr], [r["test_cost"][scenario]["cost"]], "o", markersize=7,
                color=MODEL_COLORS[r["model"]], markeredgecolor="white", markeredgewidth=2)
    approve_all = expected_cost(y_test, np.zeros(len(y_test), dtype=int), cost, amount_test)
    ax.axhline(approve_all, color=INK_2, linestyle="--", linewidth=1, label="Aprovar tudo")
    ax.set(xlabel="Limiar de decisão", ylabel="Custo total no teste",
           title=f"Custo × limiar — cenário '{scenario}' (FN={cost.fn:g}, FP={cost.fp:g})\n"
                 "marcador = limiar escolhido nas predições out-of-fold do treino")
    ax.set_ylim(0, approve_all * 1.6)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / f"cost_curves_{scenario}.png")
    plt.close(fig)


def plot_tradeoff(records: list[dict], out_dir: Path) -> None:
    """AUC-PR × latência: o compromisso desempenho/custo computacional."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m in ALL_MODELS:
        rs = [r for r in records if r["model"] == m]
        if not rs:
            continue
        ax.scatter([r["latency_mean_ms"] for r in rs],
                   [r["test_metrics"]["default"]["auc_pr"] for r in rs],
                   s=60, color=MODEL_COLORS[m], edgecolor="white", linewidth=2,
                   label=MODEL_LABELS[m])
    ax.set_xscale("log")
    ax.set(xlabel="Latência média por transação (ms, escala log)", ylabel="AUC-PR (teste)",
           title="Desempenho × custo de inferência (todas as configurações)")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "tradeoff_latency_aucpr.png")
    plt.close(fig)


def plot_confusion(records: list[dict], out_dir: Path, scenario: str = "moderado") -> None:
    best = _best_per_model(records)
    fig, axes = plt.subplots(1, len(best), figsize=(3.1 * len(best), 3.2))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, best):
        m = r["test_metrics"][f"cost_{scenario}"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(np.log1p(cm), cmap=BLUES)
        ax.grid(False)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}".replace(",", "."), ha="center", va="center",
                    color="white" if np.log1p(v) > np.log1p(cm).max() * 0.6 else INK)
        ax.set_xticks([0, 1], ["Legítima", "Fraude"])
        ax.set_yticks([0, 1], ["Legítima", "Fraude"])
        ax.set_xlabel("Predito")
        ax.set_title(f"{MODEL_LABELS[r['model']].split(' (')[0]}\nlimiar={m['threshold']:.3f}",
                     fontsize=9)
    axes[0].set_ylabel("Real")
    fig.suptitle(f"Matrizes de confusão no teste — limiar de custo '{scenario}'")
    fig.savefig(out_dir / f"confusion_{scenario}.png")
    plt.close(fig)


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)


def markdown_summary(table: pd.DataFrame, records: list[dict], dataset_info: dict) -> str:
    lines = [
        f"# Resultados — {dataset_info['name']}",
        "",
        f"- Transações: {dataset_info['n_rows']:,} | fraudes: {dataset_info['n_fraud']:,} "
        f"({dataset_info['fraud_rate_pct']}%)".replace(",", "."),
        f"- Protocolo: CV estratificada k={dataset_info['config']['cv_folds']}, busca aleatória "
        f"n_iter={dataset_info['config']['n_iter_search']}, seleção por AUC-PR, "
        f"semente={dataset_info['config']['seed']}",
        "- Limiares escolhidos com predições *out-of-fold* do treino; teste usado uma única vez.",
        "",
        "## Tabela comparativa (limiar de custo, cenário moderado FN:FP = 20:1)",
        "",
    ]
    cols = ["model", "balancing", "cv_auc_pr", "auc_pr", "auc_pr_ci95", "auc_roc", "precision",
            "recall", "f1", "mcc", "fp", "fn", "savings_pct", "train_seconds", "latency_mean_ms",
            "latency_p95_ms"]
    t = table[cols].copy()
    t["model"] = t["model"].map(MODEL_LABELS)
    t["balancing"] = t["balancing"].map(BALANCING_LABELS)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, row in t.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], 2 if c.endswith(("seconds", "_ms", "pct")) else 4)
                                       for c in cols) + " |")

    under = [r for r in records if "undersampling_calibration" in r]
    if under:
        lines += ["", "## Calibração após subamostragem (Dal Pozzolo et al., 2015)", "",
                  "| modelo | β | Brier bruto | Brier corrigido |", "|---|---|---|---|"]
        for r in under:
            c = r["undersampling_calibration"]
            lines.append(f"| {MODEL_LABELS[r['model']]} | {c['beta']:.4f} | "
                         f"{c['brier_raw']:.5f} | {c['brier_corrected']:.5f} |")

    if any("valor_transacao" in r["test_cost"] for r in records):
        lines += ["", "## Custo dependente do valor (FN = valor da fraude, piso 25; FP = 5)", "",
                  "Economia contra a melhor política trivial (a mais barata entre aprovar e "
                  "bloquear tudo): quando o FN é muito mais caro que o FP, o limiar ótimo "
                  "cai para perto de zero e \"economia vs. aprovar tudo\" engana.", "",
                  "| modelo | balanceamento | limiar | prejuízo | aprovar tudo | "
                  "bloquear tudo | economia vs. aprovar tudo % | economia vs. melhor trivial % |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in sorted(records, key=lambda r: r["test_cost"]["valor_transacao"]["cost"]):
            c = r["test_cost"]["valor_transacao"]
            lines.append(f"| {MODEL_LABELS[r['model']]} | {BALANCING_LABELS[r['balancing']]} | "
                         f"{c['threshold']:.6f} | "
                         f"{c['cost']:,.2f} | {c['cost_approve_all']:,.2f} | "
                         f"{c['cost_block_all']:,.2f} | {c['savings_vs_approve_all_pct']:.2f} | "
                         f"{savings_vs_best_trivial(c):.2f} |")

    lines += ["", "## Acurácia é enganosa", "",
              f"Um classificador que aprova tudo teria acurácia de "
              f"{100 - dataset_info['fraud_rate_pct']:.3f}% e revocação 0.", ""]
    return "\n".join(lines)


def build_report(out_dir: Path) -> None:
    import json

    records = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    probas = dict(np.load(out_dir / "test_probas.npz"))
    split = np.load(out_dir / "split.npz")
    info = json.loads((out_dir / "dataset.json").read_text(encoding="utf-8"))
    table = pd.read_csv(out_dir / "results.csv")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_auc_pr_heatmap(records, fig_dir / "auc_pr_heatmap.png")
    plot_curves(records, probas, split["y_test"], fig_dir)
    for scenario in COST_SCENARIOS:
        plot_cost_curves(records, probas, split["y_test"], split["amount_test"], fig_dir, scenario)
    plot_confusion(records, fig_dir)
    plot_tradeoff(records, fig_dir)
    (out_dir / "summary.md").write_text(markdown_summary(table, records, info), encoding="utf-8")

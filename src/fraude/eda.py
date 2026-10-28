"""Análise exploratória de dados (estatísticas e figuras)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import Dataset  # noqa: E402

LEGIT, FRAUD = "#2a78d6", "#eb6834"


def run_eda(ds: Dataset, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = ds.X.copy()
    df["is_fraud"] = ds.y.to_numpy()
    numeric = df.select_dtypes("number").drop(columns="is_fraud")

    stats = {
        **ds.summary(),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v},
        "duplicates": int(df.duplicated().sum()),
        "amount_by_class": ds.amount.groupby(ds.y.to_numpy()).describe().round(2).to_dict(orient="index"),
        "trivial_accuracy_pct": round(100 * (1 - ds.fraud_rate), 4),
        "top_correlations_with_fraud": numeric.corrwith(df["is_fraud"]).abs()
        .sort_values(ascending=False).head(10).round(4).to_dict(),
        "outlier_pct_iqr": {
            c: round(100 * float(((numeric[c] < q1 - 1.5 * iqr) | (numeric[c] > q3 + 1.5 * iqr)).mean()), 2)
            for c in numeric.columns
            for q1, q3 in [numeric[c].quantile([0.25, 0.75])]
            for iqr in [q3 - q1]
        },
    }
    numeric.describe().T.round(4).to_csv(out_dir / "descriptive_stats.csv")
    (out_dir / "eda.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str),
                                      encoding="utf-8")

    counts = ds.y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(["Legítima", "Fraude"], counts.values, color=[LEGIT, FRAUD], width=0.55)
    ax.set_yscale("log")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}".replace(",", ".") + f"\n({100 * v / counts.sum():.3f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="Transações (escala log)", title="Distribuição das classes")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_dir / "class_balance.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.logspace(0, np.log10(max(ds.amount.max(), 10) + 1), 50)
    for cls, color, label in [(0, LEGIT, "Legítima"), (1, FRAUD, "Fraude")]:
        ax.hist(ds.amount[ds.y == cls] + 1, bins=bins, density=True, alpha=0.6, color=color, label=label)
    ax.set_xscale("log")
    ax.set(xlabel="Valor + 1 (escala log)", ylabel="Densidade", title="Valor da transação por classe")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_dir / "amount_by_class.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    unit = 3600 if ds.kind == "creditcard" else 1
    hours = (ds.time / unit).astype(int)
    per_hour = pd.DataFrame({"h": hours, "y": ds.y}).groupby("h")["y"].agg(["size", "sum"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
    a1.plot(per_hour.index, per_hour["size"], color=LEGIT)
    a1.set(ylabel="Transações/h", title="Distribuição temporal")
    a2.bar(per_hour.index, per_hour["sum"], color=FRAUD, width=0.9)
    a2.set(xlabel="Hora desde o início da base", ylabel="Fraudes/h")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_dir / "temporal_distribution.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    corr = numeric.corrwith(df["is_fraud"]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.22 * len(corr) + 1))
    ax.barh(corr.index, corr.values, color=[FRAUD if v > 0 else LEGIT for v in corr.values])
    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.set(xlabel="Correlação de Pearson com a classe fraude", title="Correlação dos atributos com a fraude")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_dir / "correlation_with_fraud.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    top = corr.abs().sort_values(ascending=False).index[:6]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    for ax, col in zip(axes.ravel(), top):
        for cls, color in [(0, LEGIT), (1, FRAUD)]:
            ax.hist(numeric.loc[ds.y.to_numpy() == cls, col], bins=50, density=True,
                    alpha=0.55, color=color, label="Fraude" if cls else "Legítima")
        ax.set_title(col, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Atributos mais correlacionados com a fraude")
    fig.tight_layout()
    fig.savefig(out_dir / "top_features_by_class.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return stats

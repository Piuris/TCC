"""Métricas, função de custo, escolha de limiar e custo computacional.

A acurácia é reportada apenas para ilustrar por que é enganosa; a seleção de
modelos é guiada pela AUC-PR e a escolha do limiar pela função de custo.
"""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import SEED, CostConfig


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else 0.0,
        "auc_roc": roc_auc_score(y_true, proba),
        "auc_pr": average_precision_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "brier": brier_score_loss(y_true, np.clip(proba, 0, 1)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def bootstrap_ci(y_true, proba, metric=average_precision_score, n_boot: int = 500,
                 alpha: float = 0.05, seed: int = SEED) -> tuple[float, float]:
    """Intervalo de confiança por bootstrap estratificado (útil com poucas fraudes)."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    if len(pos) == 0 or len(neg) == 0:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        stats.append(metric(y_true[idx], proba[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _fn_costs(y_true, cost: CostConfig, amount=None) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    if cost.fn_use_amount and amount is not None:
        per_fraud = np.maximum(np.asarray(amount, dtype=float), cost.fn)
    else:
        per_fraud = np.full(len(y_true), cost.fn)
    return per_fraud * y_true


def expected_cost(y_true, pred, cost: CostConfig, amount=None) -> float:
    """Prejuízo total = Σ custo(FN) + n(FP) × custo(FP)."""
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)
    fn_cost = _fn_costs(y_true, cost, amount)[(pred == 0) & (y_true == 1)].sum()
    fp_cost = cost.fp * int(((pred == 1) & (y_true == 0)).sum())
    return float(fn_cost + fp_cost)


def cost_curve(y_true, proba, cost: CostConfig, amount=None) -> pd.DataFrame:
    """Custo total para cada limiar candidato (todos os escores distintos).

    Implementação O(n log n): ordena os escores e usa somas acumuladas, o que
    permite varrer centenas de milhares de limiares.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(proba, kind="stable")
    p_sorted = proba[order]
    fraud_cost = _fn_costs(y_true, cost, amount)[order]
    legit = (1 - y_true)[order]

    cum_fraud = np.concatenate([[0.0], np.cumsum(fraud_cost)])
    cum_legit = np.concatenate([[0], np.cumsum(legit)])
    total_legit = cum_legit[-1]

    candidates = np.concatenate([np.unique(p_sorted), [np.inf]])
    k = np.searchsorted(p_sorted, candidates, side="left")
    fn_cost = cum_fraud[k]
    fp = total_legit - cum_legit[k]
    total = fn_cost + cost.fp * fp
    return pd.DataFrame({"threshold": candidates, "fn_cost": fn_cost, "fp": fp, "cost": total})


def best_threshold_cost(y_true, proba, cost: CostConfig, amount=None) -> tuple[float, float]:
    curve = cost_curve(y_true, proba, cost, amount)
    curve = curve[np.isfinite(curve["threshold"])]
    best = curve.loc[curve["cost"].idxmin()]
    return float(best["threshold"]), float(best["cost"])


def best_threshold_f1(y_true, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def cost_report(y_true, proba, threshold: float, cost: CostConfig, amount=None) -> dict:
    """Custo do modelo no limiar escolhido vs. políticas triviais de referência."""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    model_cost = expected_cost(y_true, pred, cost, amount)
    approve_all = expected_cost(y_true, np.zeros_like(y_true), cost, amount)
    block_all = expected_cost(y_true, np.ones_like(y_true), cost, amount)
    return {
        "cost_config": asdict(cost),
        "threshold": float(threshold),
        "cost": model_cost,
        "cost_approve_all": approve_all,
        "cost_block_all": block_all,
        "blocked_pct": float(100 * pred.mean()),
        "savings_vs_approve_all_pct": 100 * (1 - model_cost / approve_all) if approve_all else 0.0,
        "savings_vs_best_trivial_pct": savings_vs_best_trivial(
            {"cost": model_cost, "cost_approve_all": approve_all, "cost_block_all": block_all}),
    }


def savings_vs_best_trivial(report: dict) -> float:
    """Economia contra a mais barata entre "aprovar tudo" e "bloquear tudo".

    Comparar só com "aprovar tudo" engana quando o custo de deixar passar uma
    fraude é muito maior que o de bloquear uma legítima (ex.: custo pelo valor no
    PaySim, com fraudes de centenas de milhares): o limiar ótimo cai para perto
    de zero, o modelo bloqueia quase tudo e a "economia" fica perto de 100%,
    embora mal supere bloquear todas as transações.
    """
    trivial = min(report["cost_approve_all"], report["cost_block_all"])
    return 100 * (1 - report["cost"] / trivial) if trivial else 0.0


def dal_pozzolo_correction(proba, beta: float) -> np.ndarray:
    """Corrige probabilidades estimadas por um modelo treinado com subamostragem.

    p = β·pₛ / (β·pₛ − pₛ + 1), em que β é a probabilidade de um exemplo
    negativo ter sido mantido na subamostragem.
    """
    ps = np.asarray(proba, dtype=float)
    return beta * ps / (beta * ps - ps + 1.0)


def measure_latency(model, X: pd.DataFrame, n_single: int = 300, batch_size: int = 10_000,
                    seed: int = SEED) -> dict:
    """Latência de inferência por transação (uma por vez) e vazão em lote.

    A medida unitária reflete a decisão de autorização em tempo real, que chega
    uma transação por vez; a de lote reflete reprocessamento offline.
    """
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(X), size=min(n_single, len(X)), replace=False)
    model.predict_proba(X.iloc[[rows[0]]])
    times = []
    for i in rows:
        row = X.iloc[[i]]
        t0 = time.perf_counter()
        model.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.asarray(times)

    batch = X.iloc[: min(batch_size, len(X))]
    t0 = time.perf_counter()
    model.predict_proba(batch)
    batch_s = time.perf_counter() - t0
    return {
        "latency_mean_ms": float(times.mean()),
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "latency_p99_ms": float(np.percentile(times, 99)),
        "batch_throughput_tps": float(len(batch) / batch_s) if batch_s > 0 else float("inf"),
    }

"""Protocolo experimental (capítulo 2 da proposta).

Para cada par (algoritmo, balanceamento):

1. Busca aleatória de hiperparâmetros com validação cruzada estratificada
   (k = 5), selecionando pela AUC-PR. Toda transformação estimada a partir dos
   dados — padronização, winsorização, SMOTE, subamostragem — está dentro do
   pipeline e é reajustada somente no treino de cada partição.
2. Predições *out-of-fold* no conjunto de treino com a melhor configuração,
   usadas para escolher os limiares (F1 máximo e custo mínimo em cada cenário
   de política de risco). O conjunto de teste nunca participa dessa escolha.
3. Avaliação única no conjunto de teste isolado, com os limiares já fixados.
4. Medição do tempo de treinamento e da latência de inferência.

Os resultados são gravados de forma incremental, permitindo retomar uma
execução interrompida (``resume=True``).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from .config import COST_SCENARIOS, ExperimentConfig, set_global_seed
from .data import Dataset
from .evaluation import (
    best_threshold_cost,
    best_threshold_f1,
    bootstrap_ci,
    classification_metrics,
    cost_report,
    dal_pozzolo_correction,
    savings_vs_best_trivial,
    measure_latency,
)
from .models import build_pipeline, is_applicable, param_space, search_n_jobs, undersampling_beta

log = logging.getLogger(__name__)

HEADLINE_SCENARIO = "moderado"


def split_train_test(ds: Dataset, cfg: ExperimentConfig):
    idx = np.arange(len(ds.y))
    tr, te = train_test_split(idx, test_size=cfg.test_size, stratify=ds.y, random_state=cfg.seed)
    return np.sort(tr), np.sort(te)


def _config_key(model: str, balancing: str) -> str:
    return f"{model}__{balancing}"


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def oof_predict(pipe, X: pd.DataFrame, y: pd.Series, cv) -> tuple[np.ndarray, list[float]]:
    """Predições *out-of-fold* e AUC-PR de cada partição, num único passe."""
    oof = np.zeros(len(y))
    scores = []
    for tr_i, va_i in cv.split(X, y):
        model = clone(pipe).fit(X.iloc[tr_i], y.iloc[tr_i])
        p = model.predict_proba(X.iloc[va_i])[:, 1]
        oof[va_i] = p
        scores.append(float(average_precision_score(y.iloc[va_i], p)))
    return oof, scores


def run_single(ds: Dataset, model: str, balancing: str, cfg: ExperimentConfig,
               tr: np.ndarray, te: np.ndarray, pipe=None, params: dict | None = None,
               extra: dict | None = None) -> tuple[dict, np.ndarray]:
    """Executa uma configuração e devolve (registro de resultados, probas no teste).

    Sem ``pipe``: busca aleatória de hiperparâmetros para (``model``, ``balancing``).
    Com ``pipe``: avalia um pipeline já configurado (ex.: ensemble), sem busca.
    """
    X_tr, y_tr = ds.X.iloc[tr], ds.y.iloc[tr]
    X_te, y_te = ds.X.iloc[te], ds.y.iloc[te]
    amt_tr, amt_te = ds.amount.iloc[tr].to_numpy(), ds.amount.iloc[te].to_numpy()
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)

    t0 = time.perf_counter()
    if pipe is None:
        pipe = build_pipeline(model, balancing, ds.kind, pos_weight=pos_weight, seed=cfg.seed)
        search = RandomizedSearchCV(
            pipe, param_space(model, balancing, minority_ratio=1 / pos_weight),
            n_iter=cfg.n_iter_search, scoring=cfg.scoring, cv=skf, refit=True,
            random_state=cfg.seed, n_jobs=search_n_jobs(model) if cfg.n_jobs == -1 else cfg.n_jobs,
            error_score="raise",
        )
        search.fit(X_tr, y_tr)
        best, best_params = search.best_estimator_, search.best_params_
        bi = search.best_index_
        fold_scores = [float(search.cv_results_[f"split{k}_test_score"][bi]) for k in range(cfg.cv_folds)]
        train_seconds = float(search.refit_time_)
        oof, _ = oof_predict(best, X_tr, y_tr, skf)
    else:
        best_params = params or {}
        oof, fold_scores = oof_predict(pipe, X_tr, y_tr, skf)
        t1 = time.perf_counter()
        best = clone(pipe).fit(X_tr, y_tr)
        train_seconds = time.perf_counter() - t1
    search_seconds = time.perf_counter() - t0

    thr_f1 = best_threshold_f1(y_tr, oof)
    thr_cost = {name: best_threshold_cost(y_tr, oof, c, amt_tr)[0]
                for name, c in COST_SCENARIOS.items()}

    proba = best.predict_proba(X_te)[:, 1]

    record = {
        "dataset": ds.name,
        "model": model,
        "balancing": balancing,
        "best_params": _to_jsonable(best_params),
        **(extra or {}),
        "cv_auc_pr_mean": float(np.mean(fold_scores)),
        "cv_auc_pr_std": float(np.std(fold_scores)),
        "cv_fold_scores": [float(s) for s in fold_scores],
        "search_seconds": search_seconds,
        "train_seconds": train_seconds,
        "test_auc_pr_ci95": bootstrap_ci(y_te, proba, seed=cfg.seed),
        "thresholds": {"default": 0.5, "f1_oof": thr_f1, **{f"cost_{k}": v for k, v in thr_cost.items()}},
        "test_metrics": {
            "default": classification_metrics(y_te, proba, 0.5),
            "f1_oof": classification_metrics(y_te, proba, thr_f1),
            **{f"cost_{k}": classification_metrics(y_te, proba, v) for k, v in thr_cost.items()},
        },
        "test_cost": {k: cost_report(y_te, proba, thr_cost[k], c, amt_te)
                      for k, c in COST_SCENARIOS.items()},
    }

    if balancing == "undersample":
        beta = undersampling_beta(y_tr, best_params["sampler__sampling_strategy"])
        corrected = dal_pozzolo_correction(proba, beta)
        record["undersampling_calibration"] = {
            "beta": beta,
            "brier_raw": classification_metrics(y_te, proba)["brier"],
            "brier_corrected": classification_metrics(y_te, corrected)["brier"],
        }

    record.update(measure_latency(best, X_te, seed=cfg.seed))
    return _to_jsonable(record), proba


def flatten(record: dict, scenario: str = HEADLINE_SCENARIO) -> dict:
    """Linha da tabela comparativa: métricas no limiar de custo do cenário principal."""
    m = record["test_metrics"][f"cost_{scenario}"]
    return {
        "dataset": record["dataset"],
        "model": record["model"],
        "balancing": record["balancing"],
        "cv_auc_pr": f"{record['cv_auc_pr_mean']:.4f} ± {record['cv_auc_pr_std']:.4f}",
        "auc_pr": m["auc_pr"],
        "auc_pr_ci95": "[{:.3f}, {:.3f}]".format(*record["test_auc_pr_ci95"]),
        "auc_roc": m["auc_roc"],
        "threshold": m["threshold"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "mcc": m["mcc"],
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
        "cost": record["test_cost"][scenario]["cost"],
        "savings_pct": record["test_cost"][scenario]["savings_vs_approve_all_pct"],
        "savings_vs_trivial_pct": savings_vs_best_trivial(record["test_cost"][scenario]),
        "train_seconds": record["train_seconds"],
        "latency_mean_ms": record["latency_mean_ms"],
        "latency_p95_ms": record["latency_p95_ms"],
    }


def run_experiments(ds: Dataset, cfg: ExperimentConfig, out_dir: Path,
                    resume: bool = True, jobs: list[tuple[str, str, dict]] | None = None) -> pd.DataFrame:
    """Executa as configurações e grava resultados de forma incremental.

    ``jobs`` permite avaliar configurações prontas (ex.: ensembles) como
    tuplas (modelo, balanceamento, kwargs de :func:`run_single`); por padrão,
    todas as combinações de ``cfg.models`` × ``cfg.balancing``.
    """
    set_global_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "results.json"
    probas_path = out_dir / "test_probas.npz"

    records: dict[str, dict] = {}
    probas: dict[str, np.ndarray] = {}
    if resume and records_path.exists():
        records = {_config_key(r["model"], r["balancing"]): r
                   for r in json.loads(records_path.read_text(encoding="utf-8"))}
        if probas_path.exists():
            probas = dict(np.load(probas_path))

    tr, te = split_train_test(ds, cfg)
    np.savez_compressed(out_dir / "split.npz", train=tr, test=te,
                        y_test=ds.y.iloc[te].to_numpy(), amount_test=ds.amount.iloc[te].to_numpy())
    (out_dir / "dataset.json").write_text(json.dumps({**ds.summary(), "config": _to_jsonable(cfg.__dict__)},
                                                     indent=2, ensure_ascii=False), encoding="utf-8")

    if jobs is None:
        jobs = [(m, b, {}) for m in cfg.models for b in cfg.balancing if is_applicable(m, b)]
    for i, (model, balancing, kwargs) in enumerate(jobs, 1):
        key = _config_key(model, balancing)
        if key in records and key in probas:
            log.info("[%d/%d] %s já concluído — pulando", i, len(jobs), key)
            continue
        log.info("[%d/%d] %s ...", i, len(jobs), key)
        record, proba = run_single(ds, model, balancing, cfg, tr, te, **kwargs)
        records[key], probas[key] = record, proba
        log.info("    AUC-PR teste=%.4f  CV=%.4f±%.4f  treino=%.1fs  latência=%.2fms",
                 record["test_metrics"]["default"]["auc_pr"], record["cv_auc_pr_mean"],
                 record["cv_auc_pr_std"], record["train_seconds"], record["latency_mean_ms"])
        records_path.write_text(json.dumps(list(records.values()), indent=2, ensure_ascii=False),
                                encoding="utf-8")
        np.savez_compressed(probas_path, **probas)

    table = pd.DataFrame([flatten(r) for r in records.values()])
    table = table.sort_values("auc_pr", ascending=False).reset_index(drop=True)
    table.to_csv(out_dir / "results.csv", index=False)
    return table


def load_records(out_dir: Path) -> list[dict]:
    return json.loads((out_dir / "results.json").read_text(encoding="utf-8"))


def select_best(records: list[dict], scenario: str = HEADLINE_SCENARIO,
                criterion: str = "cv_auc_pr_mean") -> dict:
    """Seleciona a configuração final pelo desempenho na validação cruzada.

    O critério padrão usa a CV (e não o teste) para que a escolha do modelo
    não contamine a estimativa final de desempenho.
    """
    return max(records, key=lambda r: r[criterion])

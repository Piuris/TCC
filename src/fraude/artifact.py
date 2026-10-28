"""Treinamento, serialização e carga do modelo servido pela API."""

from __future__ import annotations

import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import ARTIFACTS_DIR, COST_SCENARIOS, ExperimentConfig
from .data import Dataset
from .evaluation import (
    best_threshold_cost,
    classification_metrics,
    cost_report,
    dal_pozzolo_correction,
    measure_latency,
)
from .calibration import ProbabilityCalibrator, calibration_metrics, choose_method, reliability_plot
from .ensemble import build_from_spec
from .experiments import oof_predict, split_train_test
from .models import undersampling_beta

log = logging.getLogger(__name__)


def library_versions() -> dict:
    import imblearn
    import lightgbm
    import shap
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "imbalanced-learn": imblearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "shap": shap.__version__,
    }


def artifact_path(kind: str, directory: Path | None = None) -> Path:
    return Path(directory or ARTIFACTS_DIR) / f"model_{kind}.joblib"


def train_final(ds: Dataset, model: str, balancing: str, params: dict | None = None,
                scenario: str = "moderado", cfg: ExperimentConfig | None = None,
                out_dir: Path | None = None, members: list[dict] | None = None,
                calibration: str = "auto", report_dir: Path | None = None) -> Path:
    """Treina a configuração escolhida e grava o artefato para a API.

    O ajuste usa só a partição de treino (a mesma dos experimentos). Com as
    predições out-of-fold do treino escolhem-se o método de calibração
    (``calibration="auto"``: Platt, isotônica ou nenhuma, pelo Brier em CV
    interna) e, já sobre as probabilidades calibradas, o limiar de custo. As
    métricas do artefato são as do teste isolado.
    """
    cfg = cfg or ExperimentConfig()
    params = params or {}
    tr, te = split_train_test(ds, cfg)
    X_tr, y_tr, X_te, y_te = ds.X.iloc[tr], ds.y.iloc[tr], ds.X.iloc[te], ds.y.iloc[te]
    amt_tr, amt_te = ds.amount.iloc[tr].to_numpy(), ds.amount.iloc[te].to_numpy()
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    spec = {"model": model, "balancing": balancing, "params": params, "members": members}
    pipe = build_from_spec(spec, ds.kind, pos_weight=pos_weight, seed=cfg.seed)

    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    oof_raw, _ = oof_predict(pipe, X_tr, y_tr, skf)

    method, inner_briers = (choose_method(oof_raw, y_tr, seed=cfg.seed) if calibration == "auto"
                            else (calibration, {}))
    calibrator = ProbabilityCalibrator(method).fit(oof_raw, y_tr) if method != "none" else None
    beta = None
    if calibrator is not None:
        oof = calibrator.transform(oof_raw)
    elif balancing == "undersample":
        beta = undersampling_beta(y_tr, pipe.get_params()["sampler__sampling_strategy"])
        oof = dal_pozzolo_correction(oof_raw, beta)
    else:
        oof = oof_raw
    log.info("Calibração: %s (Brier CV interna: %s)", method,
             {k: round(v, 6) for k, v in inner_briers.items()})
    cost = COST_SCENARIOS[scenario]
    threshold, _ = best_threshold_cost(y_tr, oof, cost, amt_tr)

    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    train_seconds = time.perf_counter() - t0
    raw = pipe.predict_proba(X_te)[:, 1]
    if calibrator is not None:
        proba = calibrator.transform(raw)
    elif beta is not None:
        proba = dal_pozzolo_correction(raw, beta)
    else:
        proba = raw

    calib_report = {
        "method": method,
        "inner_cv_brier": inner_briers,
        "test_raw": calibration_metrics(y_te, raw),
        "test_calibrated": calibration_metrics(y_te, proba),
    }
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        curves = {"Sem calibração": raw}
        for m in ("sigmoid", "isotonic"):
            curves[{"sigmoid": "Platt (sigmoide)", "isotonic": "Isotônica"}[m]] =                 ProbabilityCalibrator(m).fit(oof_raw, y_tr).transform(raw)
        reliability_plot(y_te, curves, report_dir / "reliability.png",
                         title=f"Confiabilidade no teste — {model} ({balancing})")
        (report_dir / "calibration.json").write_text(
            json.dumps(calib_report, indent=2, ensure_ascii=False), encoding="utf-8")

    rng = np.random.default_rng(cfg.seed)
    background = X_tr.iloc[rng.choice(len(X_tr), size=min(200, len(X_tr)), replace=False)]

    bundle = {
        "pipeline": pipe,
        "calibrator": calibrator,
        "kind": ds.kind,
        "dataset": ds.name,
        "model": model,
        "balancing": balancing,
        "params": params,
        "members": members,
        "threshold": float(threshold),
        "threshold_policy": {"scenario": scenario, **cost.__dict__},
        "undersampling_beta": beta,
        "calibration": calib_report,
        "input_columns": list(ds.X.columns),
        "background": background,
        "dataset_info": ds.summary(),
        "train_seconds": train_seconds,
        "test_metrics": classification_metrics(y_te, proba, threshold),
        "test_cost": cost_report(y_te, proba, threshold, cost, amt_te),
        "test_cost_all_scenarios": {
            k: cost_report(y_te, proba, best_threshold_cost(y_tr, oof, c, amt_tr)[0], c, amt_te)
            for k, c in COST_SCENARIOS.items()},
        "latency": measure_latency(_Served(pipe, calibrator), X_te, seed=cfg.seed),
        "versions": library_versions(),
        "seed": cfg.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = artifact_path(ds.kind, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    meta = {k: v for k, v in bundle.items() if k not in ("pipeline", "background", "calibrator")}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
    log.info("Artefato salvo em %s (limiar=%.4f)", path, threshold)
    return path


class _Served:
    """Pipeline + calibrador, para medir a latência do que a API executa."""

    def __init__(self, pipe, calibrator):
        self.pipe, self.calibrator = pipe, calibrator

    def predict_proba(self, X):
        p = self.pipe.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.transform(p)
        return np.column_stack([1 - p, p])


class Predictor:
    """Encapsula o artefato para servir predições."""

    def __init__(self, bundle: dict, threshold: float | None = None):
        self.bundle = bundle
        self.pipeline = bundle["pipeline"]
        self.threshold = float(threshold if threshold is not None else bundle["threshold"])
        self.columns = bundle["input_columns"]
        self._explainer = None

    @classmethod
    def load(cls, path: Path, threshold: float | None = None) -> "Predictor":
        return cls(joblib.load(path), threshold=threshold)

    @property
    def version(self) -> str:
        b = self.bundle
        return f"{b['kind']}-{b['model']}-{b['balancing']}-{b['created_at'][:19]}"

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.pipeline.predict_proba(X[self.columns])[:, 1]
        calibrator = self.bundle.get("calibrator")
        if calibrator is not None:
            return calibrator.transform(p)
        beta = self.bundle.get("undersampling_beta")
        return dal_pozzolo_correction(p, beta) if beta else p

    def explainer(self):
        if self._explainer is None:
            from .explain import LocalExplainer

            self._explainer = LocalExplainer(self.pipeline, self.bundle["background"])
        return self._explainer

    def metadata(self) -> dict:
        b = self.bundle
        return {
            "version": self.version,
            "dataset": b["dataset"],
            "model": b["model"],
            "balancing": b["balancing"],
            "params": b["params"],
            "members": b.get("members"),
            "calibration": b.get("calibration"),
            "threshold": self.threshold,
            "threshold_policy": b["threshold_policy"],
            "test_metrics": b["test_metrics"],
            "test_cost": b["test_cost"],
            "latency": b["latency"],
            "versions": b["versions"],
            "created_at": b["created_at"],
        }

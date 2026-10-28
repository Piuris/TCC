"""Ensembles por votação suave e fábrica única de pipelines a partir de registros.

Os ensembles combinam, pela média das probabilidades, as melhores
configurações de cada algoritmo de árvore (escolhidas pela AUC-PR média na
CV, nunca pelo teste). Cada membro mantém sua própria técnica de
balanceamento; o pré-processamento — que não tem hiperparâmetros — é comum.

Estrutura do pipeline::

    preprocess -> passthrough -> VotingClassifier([sampler_i -> clf_i, ...])
"""

from __future__ import annotations

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import VotingClassifier

from .config import SEED
from .features import build_preprocessor
from .models import BALANCING_LABELS, MODEL_LABELS, build_pipeline

ENSEMBLES: dict[str, list[str]] = {
    "ensemble_gbdt": ["xgboost", "lightgbm"],
    "ensemble_arvores": ["xgboost", "lightgbm", "random_forest"],
}
ENSEMBLE_BALANCING = "misto"

MODEL_LABELS.update({
    "ensemble_gbdt": "Ensemble XGBoost + LightGBM",
    "ensemble_arvores": "Ensemble XGBoost + LightGBM + RF",
})
BALANCING_LABELS.setdefault(ENSEMBLE_BALANCING, "Melhor de cada membro")


def is_ensemble(model: str) -> bool:
    return model in ENSEMBLES


def _member_steps(spec: dict, kind: str, pos_weight: float, seed: int) -> ImbPipeline:
    full = build_pipeline(spec["model"], spec["balancing"], kind, pos_weight=pos_weight, seed=seed)
    full.set_params(**spec.get("params", {}))
    return ImbPipeline([("sampler", full.named_steps["sampler"]), ("clf", full.named_steps["clf"])])


def build_ensemble(members: list[dict], kind: str, pos_weight: float = 1.0,
                   seed: int = SEED) -> ImbPipeline:
    """``members``: [{"model", "balancing", "params"}, ...]."""
    voting = VotingClassifier(
        [(m["model"], _member_steps(m, kind, pos_weight, seed)) for m in members],
        voting="soft",
    )
    return ImbPipeline([("preprocess", build_preprocessor(kind)), ("sampler", "passthrough"),
                        ("clf", voting)])


def build_from_spec(spec: dict, kind: str, pos_weight: float = 1.0, seed: int = SEED):
    """Monta o pipeline de qualquer configuração: simples ou ensemble.

    ``spec`` tem ``model``, ``balancing``, ``params`` e, para ensembles, ``members``.
    """
    if is_ensemble(spec["model"]):
        return build_ensemble(spec["members"], kind, pos_weight, seed)
    pipe = build_pipeline(spec["model"], spec["balancing"], kind, pos_weight=pos_weight, seed=seed)
    return pipe.set_params(**(spec.get("params") or {}))


def best_members(records: list[dict], models: list[str]) -> list[dict]:
    """Melhor configuração (maior AUC-PR média na CV) de cada algoritmo."""
    out = []
    for m in models:
        candidates = [r for r in records if r["model"] == m]
        if not candidates:
            raise ValueError(f"Sem resultados para '{m}' — rode os experimentos antes.")
        best = max(candidates, key=lambda r: r["cv_auc_pr_mean"])
        out.append({"model": m, "balancing": best["balancing"], "params": best["best_params"],
                    "cv_auc_pr_mean": best["cv_auc_pr_mean"], "source": best.get("source", "")})
    return out


def ensemble_jobs(records: list[dict], kind: str, pos_weight: float, seed: int = SEED):
    """Tarefas para :func:`fraude.experiments.run_experiments`."""
    jobs = []
    for name, models in ENSEMBLES.items():
        members = best_members(records, models)
        jobs.append((name, ENSEMBLE_BALANCING, {
            "pipe": build_ensemble(members, kind, pos_weight, seed),
            "params": {},
            "extra": {"members": members},
        }))
    return jobs

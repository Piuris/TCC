"""Parâmetros globais do experimento.

Todos os valores que influenciam resultados ficam aqui, para que cada execução
seja reproduzível e o relatório possa citar exatamente o que foi usado.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FRAUDE_DATA_DIR", ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("FRAUDE_ARTIFACTS_DIR", ROOT / "artifacts"))
REPORTS_DIR = Path(os.environ.get("FRAUDE_REPORTS_DIR", ROOT / "reports"))

SEED = 42


@dataclass(frozen=True)
class CostConfig:
    """Custos usados na função de custo econômico (seção 4.7 da proposta).

    ``fn`` é o custo de deixar passar uma fraude; ``fp`` é o custo de bloquear
    uma transação legítima (atrito, análise manual, abandono de compra).
    Quando ``fn_use_amount`` é verdadeiro, o custo de um FN passa a ser o valor
    da própria transação (mais realista), com ``fn`` como piso.
    """

    fn: float = 100.0
    fp: float = 5.0
    fn_use_amount: bool = False


COST_SCENARIOS: dict[str, CostConfig] = {
    "conservador": CostConfig(fn=100.0, fp=10.0),
    "moderado": CostConfig(fn=100.0, fp=5.0),
    "agressivo": CostConfig(fn=100.0, fp=1.0),
    "valor_transacao": CostConfig(fn=25.0, fp=5.0, fn_use_amount=True),
}


@dataclass
class ExperimentConfig:
    seed: int = SEED
    test_size: float = 0.2
    cv_folds: int = 5
    n_iter_search: int = 10
    scoring: str = "average_precision"
    n_jobs: int = -1
    models: list[str] = field(
        default_factory=lambda: ["logreg", "random_forest", "xgboost", "lightgbm", "mlp",
                                 "isolation_forest"]
    )
    balancing: list[str] = field(
        default_factory=lambda: ["none", "smote", "undersample", "class_weight"]
    )
    sample_frac: float | None = None


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs() -> None:
    for d in (DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

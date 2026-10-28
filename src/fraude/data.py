"""Carga das bases de dados utilizadas no trabalho.

- ULB Credit Card Fraud Detection (base principal, anonimizada por PCA).
- PaySim (base sintética complementar, com atributos interpretáveis).
- Base sintética pequena no formato da ULB, usada em testes e demonstrações.

Cada carregador devolve um :class:`Dataset` com os atributos *brutos* (como
chegariam à API) — toda transformação estimada a partir dos dados fica dentro
do pipeline de modelagem, para evitar vazamento de dados.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, SEED

log = logging.getLogger(__name__)

ULB_FILE = "creditcard.csv"
PAYSIM_FILE = "paysim.csv"
SYNTHETIC_FILE = "synthetic_creditcard.csv"

TIME_COLUMNS = ("Time", "step")
ULB_V_COLS = [f"V{i}" for i in range(1, 29)]
ULB_COLS = ["Time", *ULB_V_COLS, "Amount"]

PAYSIM_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
PAYSIM_COLS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "nameDest",
]
PAYSIM_ORIGIN_COLS = ["oldbalanceOrg", "newbalanceOrig"]


@dataclass
class Dataset:
    """Base pronta para modelagem.

    ``time`` guarda a ordenação temporal (segundos na ULB, horas no PaySim) e é
    usada na análise de *concept drift*; ``amount`` alimenta a função de custo
    baseada em valor.
    """

    name: str
    X: pd.DataFrame
    y: pd.Series
    time: pd.Series
    amount: pd.Series
    kind: str
    description: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def fraud_rate(self) -> float:
        return float(self.y.mean())

    def summary(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "n_rows": int(len(self.y)),
            "n_features": int(self.X.shape[1]),
            "n_fraud": int(self.y.sum()),
            "fraud_rate_pct": round(100 * self.fraud_rate, 4),
            **self.extra,
        }

    def _select(self, mask: np.ndarray, name: str | None = None, **extra) -> "Dataset":
        return Dataset(
            name=name or self.name,
            X=self.X.loc[mask].reset_index(drop=True),
            y=self.y.loc[mask].reset_index(drop=True),
            time=self.time.loc[mask].reset_index(drop=True),
            amount=self.amount.loc[mask].reset_index(drop=True),
            kind=self.kind,
            description=self.description,
            extra={**self.extra, **extra},
        )

    def drop_duplicates(self) -> "Dataset":
        """Remove transações repetidas, mantendo a primeira ocorrência.

        Duas linhas são duplicatas quando todos os atributos e o rótulo são
        iguais, ignorando apenas o instante (``Time``/``step``). Sem esta etapa,
        cópias idênticas podem cair uma no treino e outra no teste (ou em
        partições diferentes da CV), e o modelo é avaliado sobre exemplos que
        já "viu" — superestimando o desempenho.
        """
        cols = [c for c in self.X.columns if c not in TIME_COLUMNS]
        key = self.X[cols].assign(_y=self.y.to_numpy())
        dup = key.duplicated(keep="first").to_numpy()
        return self._select(~dup, duplicates_removed=int(dup.sum()),
                            duplicate_frauds_removed=int(self.y.to_numpy()[dup].sum()))

    def subsample(self, frac: float | None, seed: int = SEED) -> "Dataset":
        """Amostra estratificada (preserva a taxa de fraude e a ordem temporal)."""
        if not frac or frac >= 1:
            return self
        rng = np.random.default_rng(seed)
        idx = []
        for cls in (0, 1):
            cls_idx = np.flatnonzero(self.y.to_numpy() == cls)
            k = max(1, int(round(len(cls_idx) * frac)))
            idx.append(rng.choice(cls_idx, size=k, replace=False))
        mask = np.zeros(len(self.y), dtype=bool)
        mask[np.concatenate(idx)] = True
        return self._select(mask, name=f"{self.name}[{frac:.0%}]")


def load_creditcard(path: Path | None = None) -> Dataset:
    path = Path(path or DATA_DIR / ULB_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"Base ULB não encontrada em {path}. Execute `python -m fraude download --dataset creditcard`."
        )
    df = pd.read_csv(path)
    df["Class"] = df["Class"].astype(str).str.strip("'\"").astype(int)
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    return _creditcard_from_frame(df, name="ulb_creditcard",
                                  description="ULB/Worldline — cartões europeus, set/2013 (PCA)")


def _creditcard_from_frame(df: pd.DataFrame, name: str, description: str) -> Dataset:
    missing = set(ULB_COLS + ["Class"]) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes na base: {sorted(missing)}")
    X = df[ULB_COLS].astype(float)
    return Dataset(
        name=name,
        X=X,
        y=df["Class"].astype(int).rename("is_fraud"),
        time=df["Time"].astype(float),
        amount=df["Amount"].astype(float),
        kind="creditcard",
        description=description,
    )


def load_paysim(path: Path | None = None, only_fraud_prone_types: bool = True,
                without_origin_balance: bool = False) -> Dataset:
    """Carrega o PaySim (Lopez-Rojas; Elmir; Axelsson, 2016).

    Por padrão restringe às operações TRANSFER e CASH_OUT, únicas em que o
    simulador gera fraudes — as demais seriam trivialmente legítimas e só
    inflariam as métricas. ``isFlaggedFraud`` é descartado por ser a saída de
    uma regra do próprio simulador (seria vazamento de rótulo). ``nameOrig`` e
    ``nameDest`` são identificadores; deles só se aproveita o prefixo do
    destino (``M`` = estabelecimento), na engenharia de atributos.

    ``without_origin_balance=True`` remove os saldos da origem (ver
    ``PAYSIM_ORIGIN_COLS``): cenário em que o atalho do simulador não está
    disponível e a comparação entre algoritmos volta a ser informativa.
    """
    path = Path(path or DATA_DIR / PAYSIM_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"Base PaySim não encontrada em {path}. Execute `python -m fraude download --dataset paysim`."
        )
    usecols = [*PAYSIM_COLS, "isFraud"]
    df = pd.read_csv(path, usecols=usecols)
    if only_fraud_prone_types:
        df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    if without_origin_balance:
        cols = [c for c in PAYSIM_COLS if c not in PAYSIM_ORIGIN_COLS]
        name, kind = "paysim_sem_saldo", "paysim_restrito"
        description = "PaySim sem os saldos da conta de origem (sem o atalho do simulador)"
    else:
        cols, name, kind = PAYSIM_COLS, "paysim", "paysim"
        description = "PaySim — simulador de mobile money (sintético, interpretável)"
    return Dataset(
        name=name,
        X=df[cols].copy(),
        y=df["isFraud"].astype(int).rename("is_fraud"),
        time=df["step"].astype(float),
        amount=df["amount"].astype(float),
        kind=kind,
        description=description,
    )


def make_synthetic_creditcard(n_samples: int = 20_000, fraud_rate: float = 0.01,
                              seed: int = SEED) -> pd.DataFrame:
    """Gera uma base com o mesmo esquema da ULB, para testes e demonstrações.

    Não substitui a base real: serve para validar o pipeline e a API sem
    depender de download. Inclui deriva leve ao longo do tempo para que a
    análise de *concept drift* tenha algo a observar.
    """
    from sklearn.datasets import make_classification

    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n_samples, n_features=28, n_informative=10, n_redundant=6,
        n_clusters_per_class=2, weights=[1 - fraud_rate], class_sep=1.2,
        flip_y=0.0, random_state=seed,
    )
    df = pd.DataFrame(X, columns=ULB_V_COLS)
    df["Time"] = np.sort(rng.uniform(0, 172_800, n_samples))
    shift = (df["Time"] / 172_800).to_numpy()
    df.loc[y == 1, "V3"] += 1.5 * shift[y == 1]
    amount = rng.lognormal(mean=3.5, sigma=1.2, size=n_samples)
    amount[y == 1] = rng.lognormal(mean=4.2, sigma=1.5, size=int(y.sum()))
    df["Amount"] = np.round(amount, 2)
    df["Class"] = y
    return df[[*ULB_COLS, "Class"]]


def load_synthetic(path: Path | None = None, **kwargs) -> Dataset:
    path = Path(path or DATA_DIR / SYNTHETIC_FILE)
    df = pd.read_csv(path) if path.exists() else make_synthetic_creditcard(**kwargs)
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    return _creditcard_from_frame(df, name="synthetic_creditcard",
                                  description="Base sintética no esquema da ULB (apenas testes)")


LOADERS = {
    "creditcard": load_creditcard,
    "paysim": load_paysim,
    "paysim_sem_saldo": lambda: load_paysim(without_origin_balance=True),
    "synthetic": load_synthetic,
}


def load_dataset(name: str, sample_frac: float | None = None, seed: int = SEED,
                 dedupe: bool = True) -> Dataset:
    """Carrega a base; por padrão remove duplicatas *antes* de qualquer separação."""
    if name not in LOADERS:
        raise ValueError(f"Base desconhecida: {name}. Opções: {list(LOADERS)}")
    ds = LOADERS[name]()
    if dedupe:
        ds = ds.drop_duplicates()
        log.info("Duplicatas removidas: %d (fraudes: %d)", ds.extra["duplicates_removed"],
                 ds.extra["duplicate_frauds_removed"])
    return ds.subsample(sample_frac, seed=seed)

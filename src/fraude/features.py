"""Pré-processamento e engenharia de atributos.

Divisão em duas camadas:

1. *Engenharia de atributos sem estado* (``CreditCardFeatures`` / ``PaySimFeatures``):
   funções determinísticas de uma única transação (hora do dia, log do valor,
   inconsistências de saldo). Não aprendem nada dos dados, então não vazam.
2. *Transformações com estado* (``build_preprocessor``): winsorização, imputação
   e padronização. São estimadas no ``fit`` e, por estarem dentro do pipeline,
   são reajustadas apenas no treino de cada partição da validação cruzada.

O pipeline recebe os atributos brutos — os mesmos que a API recebe — e por isso
o artefato serializado é autossuficiente em produção.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import PAYSIM_TYPES, ULB_V_COLS

SECONDS_PER_DAY = 86_400


def signed_log1p(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def _cyclic_hour(hour: pd.Series) -> tuple[pd.Series, pd.Series]:
    angle = 2 * np.pi * hour / 24.0
    return np.sin(angle), np.cos(angle)


class CreditCardFeatures(BaseEstimator, TransformerMixin):
    """Atributos derivados para a base ULB.

    ``Time`` é o número de segundos desde a primeira transação da base. Usado
    cru, ele só codifica *a posição da transação na base* e não generaliza para
    dados futuros; por isso é convertido em hora do dia (codificação cíclica)
    e descartado.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        hour = (out["Time"] % SECONDS_PER_DAY) / 3600.0
        out["hour_sin"], out["hour_cos"] = _cyclic_hour(hour)
        out["amount_log"] = signed_log1p(out["Amount"])
        return out.drop(columns=["Time", "Amount"])

    def get_feature_names_out(self, input_features=None):
        return np.array([*ULB_V_COLS, "hour_sin", "hour_cos", "amount_log"])


class PaySimFeatures(BaseEstimator, TransformerMixin):
    """Atributos derivados para o PaySim.

    Os erros de saldo medem a inconsistência entre o valor movimentado e a
    variação dos saldos de origem e destino. São fortemente preditivos no
    PaySim — em parte por um artefato do simulador, o que é discutido como
    limitação no texto.

    Com ``restricted=True`` só usa valor, hora, tipo e a conta de destino: nada
    derivado dos saldos da origem.
    """

    def __init__(self, restricted: bool = False):
        self.restricted = restricted

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        amount = X["amount"].astype(float)
        old_d, new_d = X["oldbalanceDest"].astype(float), X["newbalanceDest"].astype(float)

        hour = X["step"].astype(float) % 24
        out["hour_sin"], out["hour_cos"] = _cyclic_hour(hour)
        out["amount_log"] = signed_log1p(amount)
        out["oldbalanceDest_log"] = signed_log1p(old_d)
        out["newbalanceDest_log"] = signed_log1p(new_d)
        out["error_balance_dest_log"] = signed_log1p(old_d + amount - new_d)
        out["dest_zero_before_after"] = ((old_d == 0) & (new_d == 0) & (amount > 0)).astype(float)
        name_dest = X["nameDest"] if "nameDest" in X else pd.Series("C", index=X.index)
        out["dest_is_merchant"] = name_dest.astype(str).str.startswith("M").astype(float)
        if not self.restricted:
            old_o, new_o = X["oldbalanceOrg"].astype(float), X["newbalanceOrig"].astype(float)
            out["oldbalanceOrg_log"] = signed_log1p(old_o)
            out["newbalanceOrig_log"] = signed_log1p(new_o)
            out["error_balance_orig_log"] = signed_log1p(new_o + amount - old_o)
            out["amount_to_balance_ratio"] = amount / (old_o + 1.0)
            out["orig_emptied"] = ((old_o > 0) & (new_o == 0)).astype(float)
        out = out[self._numeric_names()]
        out["type"] = X["type"].astype(str)
        return out

    def _numeric_names(self) -> list[str]:
        return PAYSIM_RESTRICTED_NUMERIC if self.restricted else PAYSIM_NUMERIC

    def get_feature_names_out(self, input_features=None):
        return np.array(self._numeric_names() + ["type"])


PAYSIM_NUMERIC = [
    "hour_sin", "hour_cos", "amount_log", "oldbalanceOrg_log", "newbalanceOrig_log",
    "oldbalanceDest_log", "newbalanceDest_log", "error_balance_orig_log",
    "error_balance_dest_log", "amount_to_balance_ratio", "orig_emptied",
    "dest_zero_before_after", "dest_is_merchant",
]
PAYSIM_ORIGIN_FEATURES = [
    "oldbalanceOrg_log", "newbalanceOrig_log", "error_balance_orig_log",
    "amount_to_balance_ratio", "orig_emptied",
]
PAYSIM_RESTRICTED_NUMERIC = [c for c in PAYSIM_NUMERIC if c not in PAYSIM_ORIGIN_FEATURES]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Limita valores aos quantis estimados no treino (tratamento de atípicos).

    Aplicado apenas a atributos monetários: nas componentes PCA, valores
    extremos são frequentemente o próprio sinal de fraude (seção 4.6), e
    cortá-los degradaria a detecção.
    """

    def __init__(self, lower: float = 0.0, upper: float = 0.999):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(arr, self.lower, axis=0)
        self.high_ = np.nanquantile(arr, self.upper, axis=0)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _numeric(winsorize: bool) -> Pipeline:
    steps = [("winsor", Winsorizer())] if winsorize else []
    steps += [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    return Pipeline(steps)


def _build_steps(kind: str) -> tuple[BaseEstimator, ColumnTransformer]:
    if kind == "creditcard":
        ct = ColumnTransformer(
            [
                ("pca", _numeric(winsorize=False), [*ULB_V_COLS, "hour_sin", "hour_cos"]),
                ("money", _numeric(winsorize=True), ["amount_log"]),
            ],
            verbose_feature_names_out=False,
        )
        return CreditCardFeatures(), ct

    if kind in ("paysim", "paysim_restrito"):
        restricted = kind == "paysim_restrito"
        numeric = PAYSIM_RESTRICTED_NUMERIC if restricted else PAYSIM_NUMERIC
        money = [c for c in numeric if c.endswith("_log") or c.endswith("_ratio")]
        flags = [c for c in numeric if c not in money]
        ct = ColumnTransformer(
            [
                ("money", _numeric(winsorize=True), money),
                ("flags", _numeric(winsorize=False), flags),
                ("type", OneHotEncoder(categories=[PAYSIM_TYPES], handle_unknown="ignore",
                                       sparse_output=False), ["type"]),
            ],
            verbose_feature_names_out=False,
        )
        return PaySimFeatures(restricted=restricted), ct

    raise ValueError(f"Tipo de base desconhecido: {kind}")


class Preprocessor(BaseEstimator, TransformerMixin):
    """Engenharia de atributos + transformações por coluna, como um único passo.

    (O ``imblearn.Pipeline`` não aceita um ``Pipeline`` aninhado como etapa.)
    """

    def __init__(self, kind: str = "creditcard"):
        self.kind = kind

    def fit(self, X, y=None):
        self.engineer_, self.columns_ = _build_steps(self.kind)
        self.columns_.fit(self.engineer_.fit_transform(X), y)
        return self

    def transform(self, X):
        return self.columns_.transform(self.engineer_.transform(X))

    def get_feature_names_out(self, input_features=None):
        return self.columns_.get_feature_names_out()


def build_preprocessor(kind: str) -> Preprocessor:
    """Pré-processamento completo para o tipo de base."""
    _build_steps(kind)
    return Preprocessor(kind)


def feature_names(preprocessor: Preprocessor) -> list[str]:
    """Nomes dos atributos após o pré-processamento (para SHAP/LIME)."""
    return [str(n) for n in preprocessor.get_feature_names_out()]

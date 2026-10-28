"""Contratos de entrada e saída da API.

A validação é feita pelo Pydantic a partir das anotações de tipo: campos
ausentes, tipos errados, valores negativos onde não cabem e NaN/infinito são
recusados com HTTP 422 antes de chegar ao modelo.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

STRICT = ConfigDict(allow_inf_nan=False, extra="forbid")


class _Base(BaseModel):
    model_config = STRICT
    transaction_id: str | None = Field(None, max_length=64, description="Identificador opcional, devolvido na resposta.")


_v_fields = {
    f"V{i}": (float, Field(..., description=f"Componente principal {i} (anonimizada por PCA)."))
    for i in range(1, 29)
}

CreditCardTransaction = create_model(
    "CreditCardTransaction",
    __base__=_Base,
    Time=(float, Field(..., ge=0, description="Segundos desde a primeira transação da base.")),
    Amount=(float, Field(..., ge=0, description="Valor da transação.")),
    **_v_fields,
)
CreditCardTransaction.model_config["json_schema_extra"] = {
    "example": {"transaction_id": "tx-001", "Time": 406.0, "Amount": 0.0,
                **{f"V{i}": 0.0 for i in range(1, 29)}}
}


class PaySimTransaction(_Base):
    step: int = Field(..., ge=0, description="Hora da simulação (1 passo = 1 hora).")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(..., ge=0, description="Saldo da origem antes da operação.")
    newbalanceOrig: float = Field(..., ge=0, description="Saldo da origem depois da operação.")
    oldbalanceDest: float = Field(..., ge=0, description="Saldo do destino antes da operação.")
    newbalanceDest: float = Field(..., ge=0, description="Saldo do destino depois da operação.")
    nameDest: str = Field("C0", pattern=r"^[CM]\w*$", description="Destino; prefixo M = estabelecimento.")

    model_config = ConfigDict(**STRICT, json_schema_extra={"example": {
        "transaction_id": "tx-002", "step": 1, "type": "TRANSFER", "amount": 181.0,
        "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0, "nameDest": "C553264065"}})


class Factor(BaseModel):
    feature: str
    value: float = Field(description="Valor do atributo após o pré-processamento.")
    contribution: float = Field(description="Valor SHAP (positivo aumenta o risco).")
    direction: Literal["aumenta_risco", "reduz_risco"]


class Prediction(BaseModel):
    transaction_id: str | None
    fraud_probability: float = Field(ge=0, le=1)
    threshold: float
    decision: Literal["bloquear", "aprovar"]
    is_fraud: bool
    top_factors: list[Factor] | None = Field(None, description="Principais fatores (SHAP).")
    explanation_method: str | None = None
    model_version: str
    latency_ms: float = Field(description="Tempo de processamento no servidor.")


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    latency_ms: float


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    kind: str
    loaded: bool
    metadata: dict | None = None

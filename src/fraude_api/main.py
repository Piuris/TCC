"""API REST de predição de fraude.

Executar::

    uvicorn fraude_api.main:app --port 8000

Documentação interativa em http://localhost:8000/docs.

Variáveis de ambiente:

- ``FRAUDE_ARTIFACTS_DIR``          : diretório dos artefatos (padrão ``artifacts/``).
- ``FRAUDE_THRESHOLD_CREDITCARD``   : sobrescreve o limiar do modelo ULB.
- ``FRAUDE_THRESHOLD_PAYSIM``       : sobrescreve o limiar do modelo PaySim.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from fraude.artifact import Predictor, artifact_path
from fraude.config import ARTIFACTS_DIR

from .schemas import BatchResponse, CreditCardTransaction, ModelInfo, PaySimTransaction, Prediction

log = logging.getLogger("fraude_api")
KINDS = ("creditcard", "paysim")
MAX_BATCH = 1000


def _load_predictors(directory: Path) -> dict[str, Predictor]:
    predictors = {}
    for kind in KINDS:
        path = artifact_path(kind, directory)
        if not path.exists():
            log.warning("Modelo '%s' não encontrado em %s", kind, path)
            continue
        override = os.environ.get(f"FRAUDE_THRESHOLD_{kind.upper()}")
        predictors[kind] = Predictor.load(path, threshold=float(override) if override else None)
        sample = predictors[kind].bundle["background"].head(1)
        predictors[kind].predict_proba(sample)
        predictors[kind].explainer().top_factors(sample)
        log.info("Modelo '%s' carregado: %s (limiar=%.4f)", kind, predictors[kind].version,
                 predictors[kind].threshold)
    return predictors


def create_app(artifacts_dir: Path | None = None) -> FastAPI:
    directory = Path(artifacts_dir or os.environ.get("FRAUDE_ARTIFACTS_DIR", ARTIFACTS_DIR))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictors = _load_predictors(directory)
        yield

    app = FastAPI(
        title="API de Detecção de Fraudes",
        version="1.0.0",
        description="Protótipo do TCC — classifica transações como legítimas ou fraudulentas, "
                    "devolvendo probabilidade, decisão no limiar configurado e os principais "
                    "fatores (SHAP) que levaram à decisão.",
        lifespan=lifespan,
    )

    def get_predictor(request: Request, kind: str) -> Predictor:
        pred = request.app.state.predictors.get(kind)
        if pred is None:
            raise HTTPException(503, f"Modelo '{kind}' indisponível. Execute "
                                     f"`python -m fraude train-final --dataset {kind}`.")
        return pred

    def predict(pred: Predictor, items: list[BaseModel], explain: bool) -> list[Prediction]:
        t0 = time.perf_counter()
        rows = [it.model_dump(exclude={"transaction_id"}) for it in items]
        X = pd.DataFrame(rows)
        proba = pred.predict_proba(X)
        factors = None
        method = None
        if explain:
            explainer = pred.explainer()
            factors = explainer.top_factors(X[pred.columns], k=5)
            method = explainer.method
        elapsed = (time.perf_counter() - t0) * 1000
        return [
            Prediction(
                transaction_id=it.transaction_id,
                fraud_probability=float(p),
                threshold=pred.threshold,
                decision="bloquear" if p >= pred.threshold else "aprovar",
                is_fraud=bool(p >= pred.threshold),
                top_factors=factors[i] if factors else None,
                explanation_method=method,
                model_version=pred.version,
                latency_ms=elapsed / len(items),
            )
            for i, (it, p) in enumerate(zip(items, proba))
        ]

    explain_q = Query(True, description="Calcula os principais fatores da decisão (SHAP).")

    @app.get("/health", tags=["operação"])
    def health(request: Request):
        return {"status": "ok", "models_loaded": sorted(request.app.state.predictors)}

    @app.get("/v1/models", response_model=list[ModelInfo], tags=["operação"])
    def models(request: Request):
        preds = request.app.state.predictors
        return [ModelInfo(kind=k, loaded=k in preds,
                          metadata=preds[k].metadata() if k in preds else None) for k in KINDS]

    @app.post("/v1/creditcard/predict", response_model=Prediction, tags=["cartão de crédito (ULB)"])
    def predict_creditcard(tx: CreditCardTransaction, request: Request, explain: bool = explain_q):
        return predict(get_predictor(request, "creditcard"), [tx], explain)[0]

    @app.post("/v1/creditcard/predict/batch", response_model=BatchResponse,
              tags=["cartão de crédito (ULB)"])
    def predict_creditcard_batch(txs: list[CreditCardTransaction], request: Request,
                                 explain: bool = Query(False)):
        return _batch(get_predictor(request, "creditcard"), txs, explain)

    @app.post("/v1/paysim/predict", response_model=Prediction, tags=["mobile money (PaySim)"])
    def predict_paysim(tx: PaySimTransaction, request: Request, explain: bool = explain_q):
        return predict(get_predictor(request, "paysim"), [tx], explain)[0]

    @app.post("/v1/paysim/predict/batch", response_model=BatchResponse, tags=["mobile money (PaySim)"])
    def predict_paysim_batch(txs: list[PaySimTransaction], request: Request,
                             explain: bool = Query(False)):
        return _batch(get_predictor(request, "paysim"), txs, explain)

    def _batch(pred: Predictor, txs: list, explain: bool) -> BatchResponse:
        if not txs:
            raise HTTPException(422, "Lote vazio.")
        if len(txs) > MAX_BATCH:
            raise HTTPException(413, f"Lote acima do limite de {MAX_BATCH} transações.")
        t0 = time.perf_counter()
        out = predict(pred, txs, explain)
        return BatchResponse(predictions=out, latency_ms=(time.perf_counter() - t0) * 1000)

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()

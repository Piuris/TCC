"""Testes de contrato da API."""

import pytest
from fastapi.testclient import TestClient

from fraude.artifact import Predictor, artifact_path, train_final
from fraude_api.main import create_app

from .conftest import make_paysim_frame


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, small_ds, quick_cfg):
    d = tmp_path_factory.mktemp("artifacts")
    train_final(small_ds, "lightgbm", "class_weight", {"clf__n_estimators": 60}, cfg=quick_cfg, out_dir=d)
    train_final(make_paysim_frame(), "random_forest", "undersample",
                {"clf__n_estimators": 30, "sampler__sampling_strategy": 0.5}, cfg=quick_cfg, out_dir=d)
    return d


@pytest.fixture(scope="module")
def client(artifacts):
    with TestClient(create_app(artifacts)) as c:
        yield c


def cc_tx(small_ds, i=0, **overrides):
    row = small_ds.X.iloc[i].to_dict()
    return {"transaction_id": f"tx-{i}", **row, **overrides}


PAYSIM_FRAUD = {"transaction_id": "p1", "step": 5, "type": "TRANSFER", "amount": 9000.0,
                "oldbalanceOrg": 9000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0, "nameDest": "C123"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["models_loaded"] == ["creditcard", "paysim"]


def test_models_metadata(client):
    info = {m["kind"]: m for m in client.get("/v1/models").json()}
    assert info["creditcard"]["loaded"]
    meta = info["creditcard"]["metadata"]
    assert 0 < meta["threshold"] < 1
    assert {"auc_pr", "recall", "precision"} <= set(meta["test_metrics"])


def test_predict_creditcard_contract(client, small_ds):
    r = client.post("/v1/creditcard/predict", json=cc_tx(small_ds))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transaction_id"] == "tx-0"
    assert 0 <= body["fraud_probability"] <= 1
    assert body["decision"] in {"bloquear", "aprovar"}
    assert body["is_fraud"] == (body["fraud_probability"] >= body["threshold"])
    assert len(body["top_factors"]) == 5
    assert body["explanation_method"] == "shap_tree"


def test_predict_matches_offline_model(client, small_ds, artifacts):
    pred = Predictor.load(artifact_path("creditcard", artifacts))
    expected = pred.predict_proba(small_ds.X.iloc[[3]])[0]
    r = client.post("/v1/creditcard/predict?explain=false", json=cc_tx(small_ds, 3))
    assert r.json()["fraud_probability"] == pytest.approx(expected)
    assert r.json()["top_factors"] is None


def test_fraud_is_blocked(client, small_ds):
    i = int(small_ds.y[small_ds.y == 1].index[0])
    r = client.post("/v1/creditcard/predict", json=cc_tx(small_ds, i)).json()
    assert r["fraud_probability"] > 0.5


@pytest.mark.parametrize("bad", [
    {"Amount": -1.0},
    {"V1": "abc"},
    {"extra_field": 1},
])
def test_invalid_creditcard_payload_is_rejected(client, small_ds, bad):
    assert client.post("/v1/creditcard/predict", json=cc_tx(small_ds, **bad)).status_code == 422


def test_missing_field_is_rejected(client, small_ds):
    tx = cc_tx(small_ds)
    del tx["V17"]
    assert client.post("/v1/creditcard/predict", json=tx).status_code == 422


def test_batch(client, small_ds):
    r = client.post("/v1/creditcard/predict/batch", json=[cc_tx(small_ds, i) for i in range(10)])
    assert r.status_code == 200
    assert [p["transaction_id"] for p in r.json()["predictions"]] == [f"tx-{i}" for i in range(10)]
    assert client.post("/v1/creditcard/predict/batch", json=[]).status_code == 422


def test_paysim_predict_with_calibrated_undersampling(client):
    r = client.post("/v1/paysim/predict", json=PAYSIM_FRAUD)
    assert r.status_code == 200, r.text
    assert r.json()["decision"] == "bloquear"


def test_paysim_invalid_type(client):
    assert client.post("/v1/paysim/predict", json={**PAYSIM_FRAUD, "type": "PIX"}).status_code == 422


def test_missing_model_returns_503(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        r = c.post("/v1/paysim/predict", json=PAYSIM_FRAUD)
        assert r.status_code == 503


def test_threshold_override(artifacts, monkeypatch, small_ds):
    monkeypatch.setenv("FRAUDE_THRESHOLD_CREDITCARD", "0.0")
    with TestClient(create_app(artifacts)) as c:
        r = c.post("/v1/creditcard/predict?explain=false", json=cc_tx(small_ds)).json()
        assert r["threshold"] == 0.0 and r["decision"] == "bloquear"

import numpy as np
import pandas as pd

from fraude.features import (
    CreditCardFeatures,
    PaySimFeatures,
    Winsorizer,
    build_preprocessor,
    feature_names,
    signed_log1p,
)


def test_creditcard_features_drop_raw_time_and_amount(small_ds):
    out = CreditCardFeatures().transform(small_ds.X.head(10))
    assert "Time" not in out and "Amount" not in out
    assert {"hour_sin", "hour_cos", "amount_log"} <= set(out.columns)
    np.testing.assert_allclose(out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1.0)


def test_hour_is_cyclic():
    X = pd.DataFrame({"Time": [0.0, 86_400.0], "Amount": [1.0, 1.0],
                      **{f"V{i}": [0.0, 0.0] for i in range(1, 29)}})
    out = CreditCardFeatures().transform(X)
    np.testing.assert_allclose(out.iloc[0][["hour_sin", "hour_cos"]],
                               out.iloc[1][["hour_sin", "hour_cos"]], atol=1e-12)


def test_signed_log_handles_negative():
    np.testing.assert_allclose(signed_log1p([-np.e + 1, 0, np.e - 1]), [-1, 0, 1])


def test_paysim_balance_errors(paysim_ds):
    out = PaySimFeatures().transform(paysim_ds.X)
    assert out.isna().sum().sum() == 0
    assert set(out["type"]) <= {"TRANSFER", "CASH_OUT"}
    assert out.loc[paysim_ds.y == 1, "orig_emptied"].mean() > 0.9


def test_paysim_restricted_uses_no_origin_balance(paysim_ds):
    from fraude.data import PAYSIM_ORIGIN_COLS
    from fraude.features import PAYSIM_ORIGIN_FEATURES, build_preprocessor, feature_names

    X = paysim_ds.X.drop(columns=PAYSIM_ORIGIN_COLS)
    out = PaySimFeatures(restricted=True).transform(X)
    assert not set(PAYSIM_ORIGIN_FEATURES) & set(out.columns)
    pre = build_preprocessor("paysim_restrito").fit(X, paysim_ds.y)
    names = feature_names(pre)
    assert len(names) == pre.transform(X).shape[1]
    assert not any("Org" in n or "Orig" in n or n.startswith("orig_") for n in names)


def test_winsorizer_uses_only_fit_statistics():
    w = Winsorizer(lower=0.0, upper=0.9).fit(np.arange(100).reshape(-1, 1))
    assert w.transform([[1000.0]])[0, 0] == w.high_[0] < 100


def test_preprocessor_is_fit_only_on_training_data(small_ds):
    """Os parâmetros de padronização devem vir do treino, não da base inteira."""
    X = small_ds.X
    train, test = X.iloc[:1000], X.iloc[1000:].copy()
    test["Amount"] = test["Amount"] * 1000
    pre = build_preprocessor("creditcard").fit(train)
    scaler = pre.columns_.named_transformers_["money"].named_steps["scale"]
    expected = signed_log1p(train["Amount"]).clip(max=np.quantile(signed_log1p(train["Amount"]), 0.999))
    np.testing.assert_allclose(scaler.mean_[0], expected.mean(), rtol=1e-6)
    pre.transform(test)
    np.testing.assert_allclose(scaler.mean_[0], expected.mean(), rtol=1e-6)


def test_preprocessor_handles_missing_values(small_ds):
    X = small_ds.X.head(200).copy()
    X.loc[X.index[:5], "V1"] = np.nan
    out = build_preprocessor("creditcard").fit_transform(X)
    assert not np.isnan(out).any()


def test_feature_names(small_ds, paysim_ds):
    pre = build_preprocessor("creditcard").fit(small_ds.X.head(100))
    assert len(feature_names(pre)) == 31
    pre = build_preprocessor("paysim").fit(paysim_ds.X)
    names = feature_names(pre)
    assert "type_TRANSFER" in names and "error_balance_orig_log" in names

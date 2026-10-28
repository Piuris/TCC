import numpy as np
import pandas as pd
import pytest

from fraude.config import ExperimentConfig
from fraude.data import Dataset, PAYSIM_COLS, load_synthetic


@pytest.fixture(scope="session")
def small_ds(tmp_path_factory):
    missing = tmp_path_factory.mktemp("synth") / "nao_existe.csv"
    return load_synthetic(missing, n_samples=3000, fraud_rate=0.04, seed=7)


@pytest.fixture(scope="session")
def quick_cfg():
    return ExperimentConfig(cv_folds=3, n_iter_search=2, n_jobs=1)


def make_paysim_frame(n=1500, seed=3) -> Dataset:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    amount = rng.lognormal(8, 1.5, n)
    old_o = np.where(y == 1, amount, amount * rng.uniform(1, 5, n))
    new_o = np.where(y == 1, 0.0, old_o - amount)
    old_d = np.where(y == 1, 0.0, rng.lognormal(9, 1, n))
    new_d = np.where(y == 1, 0.0, old_d + amount)
    X = pd.DataFrame({
        "step": np.sort(rng.integers(1, 200, n)),
        "type": rng.choice(["TRANSFER", "CASH_OUT"], n),
        "amount": amount, "oldbalanceOrg": old_o, "newbalanceOrig": new_o,
        "oldbalanceDest": old_d, "newbalanceDest": new_d,
        "nameDest": [f"C{i}" for i in range(n)],
    })[PAYSIM_COLS]
    return Dataset(name="paysim_test", X=X, y=pd.Series(y, name="is_fraud"),
                   time=X["step"].astype(float), amount=X["amount"], kind="paysim")


@pytest.fixture(scope="session")
def paysim_ds():
    return make_paysim_frame()

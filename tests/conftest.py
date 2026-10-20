"""Shared fixtures. Tests are hermetic: no network, no dependency on the
real trained artifact (a tiny model is trained on synthetic valid data)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heartml import config
from heartml.features import build_pipeline, to_model_frame

VALID_PATIENT = {
    "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0,
    "restecg": 2, "thalach": 130, "exang": 1, "oldpeak": 1.8, "slope": 2,
    "ca": 1, "thal": 7,
}


def make_synthetic(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 78, n), "trestbps": rng.integers(95, 200, n),
        "chol": rng.integers(130, 560, n), "thalach": rng.integers(70, 200, n),
        "oldpeak": rng.uniform(0, 6, n).round(1), "ca": rng.integers(0, 4, n),
        "sex": rng.integers(0, 2, n), "fbs": rng.integers(0, 2, n),
        "exang": rng.integers(0, 2, n), "cp": rng.integers(1, 5, n),
        "restecg": rng.integers(0, 3, n), "slope": rng.integers(1, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    }).astype(float)
    df.loc[[0, 5], "ca"] = np.nan
    df.loc[[3], "thal"] = np.nan
    df[config.TARGET] = ((df["exang"] + (df["cp"] == 4) + df["oldpeak"] / 3)
                         > 1.2).astype(int)
    return df


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture(scope="session")
def fitted_pipeline(synthetic_df):
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(to_model_frame(synthetic_df), synthetic_df[config.TARGET])
    return pipe


@pytest.fixture(scope="session")
def model_file(tmp_path_factory, fitted_pipeline):
    import json

    import joblib
    d = tmp_path_factory.mktemp("model")
    path = d / "heart_model.joblib"
    joblib.dump(fitted_pipeline, path)
    (d / "model_metadata.json").write_text(json.dumps(
        {"model_name": "test_lr", "model_version": "test"}))
    return path

"""Feature schema, preprocessing and model behaviour."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from heartml import config
from heartml.features import build_preprocessor, to_model_frame
from heartml.predict import predict_records
from tests.conftest import VALID_PATIENT


def test_feature_groups_are_disjoint_and_complete():
    groups = [config.NUMERIC_FEATURES, config.BINARY_FEATURES,
              config.NOMINAL_FEATURES]
    flat = [c for g in groups for c in g]
    assert len(flat) == len(set(flat)) == 13
    assert set(flat) == set(config.RAW_COLUMNS) - {"num"}


def test_to_model_frame_orders_and_rejects_missing_columns(synthetic_df):
    shuffled = synthetic_df[list(reversed(config.FEATURES))]
    assert list(to_model_frame(shuffled).columns) == config.FEATURES
    with pytest.raises(ValueError, match="missing feature columns"):
        to_model_frame(synthetic_df.drop(columns=["chol"]))


def test_preprocessor_output_dimension_and_no_nans(synthetic_df):
    pre = build_preprocessor().fit(to_model_frame(synthetic_df))
    out = pre.transform(to_model_frame(synthetic_df))
    # 6 numeric + 3 binary + one-hot(cp 4 + restecg 3 + slope 3 + thal 3)
    assert out.shape == (len(synthetic_df), 6 + 3 + 13)
    assert not np.isnan(out).any()


def test_unseen_category_does_not_crash(fitted_pipeline, synthetic_df):
    row = to_model_frame(synthetic_df.head(1)).copy()
    row["restecg"] = 5.0  # never seen during fit -> ignored by one-hot
    assert fitted_pipeline.predict_proba(row).shape == (1, 2)


def test_pipeline_predicts_valid_probabilities(fitted_pipeline, synthetic_df):
    proba = fitted_pipeline.predict_proba(to_model_frame(synthetic_df))
    assert proba.shape == (len(synthetic_df), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(fitted_pipeline.predict(to_model_frame(synthetic_df))) <= {0, 1}


def test_pipeline_learns_signal(fitted_pipeline, synthetic_df):
    acc = (fitted_pipeline.predict(to_model_frame(synthetic_df))
           == synthetic_df[config.TARGET]).mean()
    assert acc > 0.75


def test_predict_records_output_contract(fitted_pipeline):
    out = predict_records(fitted_pipeline, [VALID_PATIENT])[0]
    assert out["prediction"] in (0, 1)
    assert 0.0 <= out["probability_disease"] <= 1.0
    assert out["confidence"] >= 0.5
    assert out["label"] in ("no heart disease", "heart disease present")


def test_predict_records_accepts_missing_ca_thal(fitted_pipeline):
    rec = {**VALID_PATIENT, "ca": None, "thal": None}
    assert predict_records(fitted_pipeline, [rec])[0]["prediction"] in (0, 1)


@pytest.mark.parametrize("bad", [
    {"age": -3}, {"sex": 2}, {"cp": 0}, {"chol": 5000}, {"thal": 4},
    {"oldpeak": "high"},
])
def test_predict_records_rejects_invalid_values(fitted_pipeline, bad):
    with pytest.raises(ValidationError):
        predict_records(fitted_pipeline, [{**VALID_PATIENT, **bad}])


def test_predict_records_rejects_missing_required_field(fitted_pipeline):
    rec = {k: v for k, v in VALID_PATIENT.items() if k != "age"}
    with pytest.raises(ValidationError):
        predict_records(fitted_pipeline, [rec])


def test_frame_dtype_is_float(synthetic_df):
    assert all(t == np.float64 for t in to_model_frame(synthetic_df).dtypes)
    assert isinstance(to_model_frame(pd.DataFrame([VALID_PATIENT])), pd.DataFrame)

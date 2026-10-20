"""Feature engineering: one ColumnTransformer shared by every model.

Design decisions (explain these in your own words in the report):
* numeric  -> median imputation (robust to skew/outliers) + standard scaling
* binary   -> most-frequent imputation, passed through as 0/1 (no scaling)
* nominal  -> most-frequent imputation + one-hot encoding
  (cp, restecg, slope, thal are unordered codes; treating them as numbers
  would impose a false ordering on Logistic Regression)
All statistics are learned in .fit(), i.e. on training folds only.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartml import config


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_FEATURES),
            ("bin", binary, config.BINARY_FEATURES),
            ("nom", nominal, config.NOMINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator) -> Pipeline:
    """Full model = preprocessing + estimator, saved and served as ONE object."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select the 13 features in the canonical order and cast to float.

    Used identically at training and inference time, so the served model
    always sees the same columns/dtypes it was trained on.
    """
    missing = [c for c in config.FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return df[config.FEATURES].astype(float)

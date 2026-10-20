"""Data loading, validation and cleaning."""

from pathlib import Path

import pandas as pd

from heartml import config
from heartml.data import clean, load_raw, validate

FIXTURE = Path(__file__).parent / "fixtures" / "sample_raw.data"


def test_load_raw_has_expected_columns_and_parses_missing():
    df = load_raw(FIXTURE)
    assert list(df.columns) == config.RAW_COLUMNS
    assert df.shape == (8, 14)
    assert df["thal"].isna().sum() == 1  # '?' -> NaN
    assert df["ca"].isna().sum() == 1


def test_validate_reports_missing_and_duplicates():
    report = validate(load_raw(FIXTURE))
    assert report["missing_per_column"] == {"ca": 1, "thal": 1}
    assert report["n_duplicate_rows"] == 1  # row 3 repeated as row 8
    assert report["invalid_categories"] == {}


def test_validate_flags_invalid_category_and_range():
    df = load_raw(FIXTURE)
    df.loc[0, "cp"] = 9
    df.loc[1, "age"] = 250
    report = validate(df)
    assert report["invalid_categories"] == {"cp": [9.0]}
    assert report["out_of_range"] == {"age": 1}


def test_clean_binarises_target_and_drops_duplicates():
    cleaned = clean(load_raw(FIXTURE))
    assert config.TARGET in cleaned.columns and "num" not in cleaned.columns
    assert set(cleaned[config.TARGET].unique()) <= {0, 1}
    assert len(cleaned) == 7
    # num 2 and num 1 both map to class 1
    assert cleaned[config.TARGET].tolist()[:3] == [0, 1, 1]
    # missing values are kept for in-pipeline imputation
    assert cleaned[["ca", "thal"]].isna().sum().sum() == 2


def test_clean_column_order_matches_schema():
    cleaned = clean(load_raw(FIXTURE))
    assert list(cleaned.columns) == config.FEATURES + [config.TARGET]
    assert isinstance(cleaned, pd.DataFrame)

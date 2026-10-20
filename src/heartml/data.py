"""Data acquisition, loading, validation and cleaning."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from heartml import config


def sha256_of(path: Path) -> str:
    """Fingerprint a file so the exact dataset version can be recorded."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_raw(url: str = config.UCI_URL, dest: Path = config.RAW_FILE,
                 force: bool = False) -> Path:
    """Download the raw UCI file (skipped if it already exists)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[download] already present: {dest}")
        return dest
    print(f"[download] {url}")
    with urllib.request.urlopen(url, timeout=30) as resp:
        dest.write_bytes(resp.read())
    print(f"[download] saved {dest} ({dest.stat().st_size} bytes)")
    return dest


def load_raw(path: Path = config.RAW_FILE) -> pd.DataFrame:
    """Raw file has NO header and uses '?' for missing values."""
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS,
                       na_values="?")


def validate(df: pd.DataFrame) -> dict:
    """Return a data-quality report (does not modify the data)."""
    report = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "missing_per_column": {c: int(v) for c, v in df.isna().sum().items()
                               if v > 0},
        "n_duplicate_rows": int(df.duplicated().sum()),
        "invalid_categories": {},
        "out_of_range": {},
    }
    for col, allowed in config.ALLOWED_VALUES.items():
        bad = df[col].dropna()[~df[col].dropna().isin(allowed)]
        if len(bad):
            report["invalid_categories"][col] = sorted(bad.unique().tolist())
    for col, (lo, hi) in config.NUMERIC_RANGES.items():
        n_bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n_bad:
            report["out_of_range"][col] = n_bad
    if "num" in df.columns:
        report["num_distribution"] = {
            int(k): int(v) for k, v in df["num"].value_counts().sort_index().items()
        }
    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Binarise the target, drop duplicates, cast types.

    Missing values are deliberately KEPT (as NaN): they are imputed inside
    the scikit-learn pipeline, fitted on training folds only, which avoids
    leaking test-set statistics into preprocessing.
    """
    out = df.copy()
    out[config.TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns=["num"])
    out = out.drop_duplicates().reset_index(drop=True)
    int_like = config.BINARY_FEATURES + config.NOMINAL_FEATURES + ["ca"]
    for col in int_like:
        out[col] = out[col].astype("Int64")  # nullable integer keeps NaN
    return out[config.FEATURES + [config.TARGET]]


def load_clean(path: Path = config.CLEAN_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def write_manifest(raw_path: Path, report: dict, clean_df: pd.DataFrame,
                   path: Path = config.MANIFEST_FILE) -> dict:
    manifest = {
        "source": "UCI Machine Learning Repository - Heart Disease (id 45)",
        "subset": "Cleveland (processed.cleveland.data)",
        "url": config.UCI_URL,
        "doi": config.DATASET_DOI,
        "raw_sha256": sha256_of(raw_path),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_quality_report": report,
        "clean_rows": int(len(clean_df)),
        "clean_class_counts": {
            int(k): int(v)
            for k, v in clean_df[config.TARGET].value_counts().sort_index().items()
        },
    }
    path.write_text(json.dumps(manifest, indent=2))
    return manifest

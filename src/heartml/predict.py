"""Model loading and inference (used by the API, the CLI demo and the tests)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from heartml import config
from heartml.features import to_model_frame
from heartml.schema import PatientFeatures

DEFAULT_MODEL_PATH = config.MODELS_DIR / "heart_model.joblib"
LABELS = {0: "no heart disease", 1: "heart disease present"}


def load_model(path: Path | str = DEFAULT_MODEL_PATH):
    """Load the fitted sklearn Pipeline.

    SECURITY: joblib/pickle can execute code while loading. Only load
    artifacts produced by this project's own training run (or CI).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model artifact not found: {path}")
    return joblib.load(path)


def load_metadata(model_path: Path | str = DEFAULT_MODEL_PATH) -> dict:
    meta = Path(model_path).with_name("model_metadata.json")
    return json.loads(meta.read_text()) if meta.exists() else {}


def predict_records(model, records: Iterable[dict]) -> list[dict]:
    """Validate each record against the schema, then predict."""
    validated = [PatientFeatures(**r).model_dump() for r in records]
    frame = to_model_frame(pd.DataFrame(validated))
    proba = model.predict_proba(frame)[:, 1]
    results = []
    for p in proba:
        pred = int(p >= 0.5)
        results.append({
            "prediction": pred,
            "label": LABELS[pred],
            "probability_disease": round(float(p), 4),
            "confidence": round(float(p if pred == 1 else 1 - p), 4),
        })
    return results

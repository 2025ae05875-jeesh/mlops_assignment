"""Load the saved artifact in a FRESH process and predict sample patients.

Usage:  python scripts/predict_sample.py [path/to/heart_model.joblib]
"""

import json
import sys

from heartml.predict import (
    DEFAULT_MODEL_PATH,
    load_metadata,
    load_model,
    predict_records,
)

SAMPLES = [
    {"age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0,
     "restecg": 2, "thalach": 130, "exang": 1, "oldpeak": 1.8, "slope": 2,
     "ca": 1, "thal": 7},
    {"age": 41, "sex": 0, "cp": 2, "trestbps": 126, "chol": 220, "fbs": 0,
     "restecg": 0, "thalach": 172, "exang": 0, "oldpeak": 0.0, "slope": 1,
     "ca": 0, "thal": 3},
    {"age": 62, "sex": 1, "cp": 3, "trestbps": 150, "chol": 240, "fbs": 1,
     "restecg": 0, "thalach": 145, "exang": 0, "oldpeak": 0.6, "slope": 2,
     "ca": None, "thal": None},
]


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    model = load_model(path)
    meta = load_metadata(path)
    print(f"Loaded {meta.get('model_name')} v{meta.get('model_version')} from {path}")
    for rec, out in zip(SAMPLES, predict_records(model, SAMPLES)):
        print(json.dumps({"age": rec["age"], "cp": rec["cp"], **out}))


if __name__ == "__main__":
    main()

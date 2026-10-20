"""Single source of truth for paths, schema and reproducibility settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- Data source (UCI ML Repository, dataset id 45, Cleveland subset) ----
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
DATASET_DOI = "10.24432/C52P4X"

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_FILE = RAW_DIR / "processed.cleveland.data"
CLEAN_FILE = PROCESSED_DIR / "heart_clean.csv"
MANIFEST_FILE = PROJECT_ROOT / "data" / "data_manifest.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# ---- Schema (order = column order of the raw UCI file) ----
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"  # derived: 0 = no disease (num == 0), 1 = disease (num 1-4)

# Feature groups (my design decision -- see report section 4)
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
NOMINAL_FEATURES = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + NOMINAL_FEATURES

# Allowed values / plausible ranges used by validation and the API
ALLOWED_VALUES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}
NUMERIC_RANGES = {
    "age": (1, 120),
    "trestbps": (50, 250),
    "chol": (100, 700),
    "thalach": (50, 250),
    "oldpeak": (0.0, 10.0),
}

# ---- Reproducibility ----
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

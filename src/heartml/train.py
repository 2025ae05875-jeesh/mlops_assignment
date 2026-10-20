"""Train, tune, evaluate and track Logistic Regression and Random Forest.

Usage:
    python -m heartml.train                  # full grid search (default)
    python -m heartml.train --quick          # tiny grids (CI smoke run)

Protocol (identical for both models):
 1. Stratified 80/20 train/test split, seed 42. Test set is touched ONCE.
 2. GridSearchCV, 5-fold StratifiedKFold on the training set, refit on ROC-AUC.
 3. Every grid candidate is logged as a nested MLflow child run.
 4. Champion = highest mean CV ROC-AUC (selection never uses the test set).
 5. Champion pipeline (preprocessing + model) saved with joblib + metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)

from heartml import config  # noqa: E402
from heartml.data import load_clean, sha256_of  # noqa: E402
from heartml.features import build_pipeline, to_model_frame  # noqa: E402

EXPERIMENT = "heart-disease-uci"
SCORING = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}
REPORT_DIR = config.PROJECT_ROOT / "reports"
MIN_TEST_ROC_AUC = float(os.getenv("MIN_TEST_ROC_AUC", "0.80"))  # quality gate
N_JOBS = int(os.getenv("N_JOBS", "-1"))  # -1 = all CPUs for GridSearchCV


def model_specs(quick: bool = False) -> dict:
    seed = config.RANDOM_SEED
    if quick:
        return {
            "logistic_regression": (
                LogisticRegression(max_iter=2000, solver="liblinear",
                                   random_state=seed),
                {"model__C": [1.0]},
            ),
            "random_forest": (
                RandomForestClassifier(random_state=seed, n_jobs=1),
                {"model__n_estimators": [100], "model__max_depth": [5]},
            ),
        }
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=5000, solver="liblinear",
                               random_state=seed),
            {
                "model__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
                "model__penalty": ["l1", "l2"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 4, 8],
                "model__min_samples_leaf": [1, 3, 5],
                "model__max_features": ["sqrt", 0.5],
            },
        ),
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def test_metrics(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "test_roc_auc": roc_auc_score(y_test, proba),
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred, zero_division=0),
        "test_recall": recall_score(y_test, pred, zero_division=0),
        "test_f1": f1_score(y_test, pred, zero_division=0),
    }


def save_plots(name: str, model, X_test, y_test, out_dir) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["no disease", "disease"],
        cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"{name}: confusion matrix (test)")
    fig.tight_layout()
    p = out_dir / f"confusion_matrix_{name}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"{name}: ROC curve (test)")
    fig.tight_layout()
    p = out_dir / f"roc_curve_{name}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    est = model.named_steps["model"]
    names = model.named_steps["preprocess"].get_feature_names_out()
    if hasattr(est, "feature_importances_"):
        values, title = est.feature_importances_, "impurity importance"
    else:
        values, title = np.abs(est.coef_[0]), "|coefficient| (scaled inputs)"
    order = np.argsort(values)[-15:]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(names)[order], values[order])
    ax.set_title(f"{name}: {title}")
    fig.tight_layout()
    p = out_dir / f"feature_importance_{name}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    return paths


def log_child_runs(name: str, search: GridSearchCV) -> None:
    res = search.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"{name}-cand{i:02d}", nested=True):
            mlflow.set_tag("model_family", name)
            mlflow.set_tag("run_type", "cv_candidate")
            mlflow.log_params({k.replace("model__", ""): str(v)
                               for k, v in params.items()})
            for m in SCORING:
                mlflow.log_metric(f"cv_{m}_mean", float(res[f"mean_test_{m}"][i]))
                mlflow.log_metric(f"cv_{m}_std", float(res[f"std_test_{m}"][i]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="tiny grids for CI smoke runs")
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI",
                             (config.PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    print(f"[mlflow] tracking uri: {tracking_uri}")

    df = load_clean()
    X = to_model_frame(df)
    y = df[config.TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_SEED)
    data_hash = sha256_of(config.CLEAN_FILE)
    print(f"[data] train={X_train.shape} test={X_test.shape} sha256={data_hash[:12]}")

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_SEED)
    fig_dir = REPORT_DIR / "figures" / "models"
    summary = []
    best = None

    for name, (estimator, grid) in model_specs(args.quick).items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"model_family": name, "run_type": "tuned_model",
                             "git_commit": git_commit(),
                             "quick_mode": str(args.quick)})
            mlflow.log_params({
                "model_name": name,
                "random_seed": config.RANDOM_SEED,
                "test_size": config.TEST_SIZE,
                "cv_folds": config.CV_FOLDS,
                "search": "GridSearchCV",
                "refit_metric": "roc_auc",
                "n_candidates": int(np.prod([len(v) for v in grid.values()])),
                "n_train": len(X_train), "n_test": len(X_test),
                "data_sha256": data_hash,
                "sklearn_version": sklearn.__version__,
            })

            search = GridSearchCV(build_pipeline(estimator), grid, cv=cv,
                                  scoring=SCORING, refit="roc_auc", n_jobs=N_JOBS,
                                  return_train_score=False)
            search.fit(X_train, y_train)
            log_child_runs(name, search)

            bi = search.best_index_
            cv_res = {f"cv_{m}_mean": float(search.cv_results_[f"mean_test_{m}"][bi])
                      for m in SCORING}
            cv_res.update({f"cv_{m}_std":
                           float(search.cv_results_[f"std_test_{m}"][bi])
                           for m in SCORING})
            best_params = {k.replace("model__", "best_"): str(v)
                           for k, v in search.best_params_.items()}
            mlflow.log_params(best_params)
            model = search.best_estimator_
            tm = test_metrics(model, X_test, y_test)
            mlflow.log_metrics({**cv_res, **tm})

            for p in save_plots(name, model, X_test, y_test, fig_dir):
                mlflow.log_artifact(str(p), artifact_path="plots")
            cv_csv = REPORT_DIR / f"cv_results_{name}.csv"
            pd.DataFrame(search.cv_results_).drop(columns=["params"]).to_csv(
                cv_csv, index=False)
            mlflow.log_artifact(str(cv_csv), artifact_path="cv")

            signature = infer_signature(X_train, model.predict_proba(X_train)[:, 1])
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     signature=signature,
                                     input_example=X_train.head(3))

            row = {"model": name, "run_id": run.info.run_id, **best_params,
                   **{k: round(v, 4) for k, v in {**cv_res, **tm}.items()}}
            summary.append(row)
            print(f"[{name}] CV ROC-AUC={cv_res['cv_roc_auc_mean']:.4f} "
                  f"(+/-{cv_res['cv_roc_auc_std']:.4f}) | "
                  f"test ROC-AUC={tm['test_roc_auc']:.4f} "
                  f"acc={tm['test_accuracy']:.4f} prec={tm['test_precision']:.4f} "
                  f"rec={tm['test_recall']:.4f}")
            if best is None or cv_res["cv_roc_auc_mean"] > best[1]:
                best = (name, cv_res["cv_roc_auc_mean"], model, row)

    # ---- champion selection (CV-based) and packaging ----
    name, _, model, row = best
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "heart_model.joblib"
    joblib.dump(model, model_path)
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    metadata = {
        "model_name": name,
        "model_version": version,
        "selection_rule": "highest mean 5-fold CV ROC-AUC on training set",
        "mlflow_run_id": row["run_id"],
        "metrics": {k: v for k, v in row.items() if k.startswith(("cv_", "test_"))},
        "best_params": {k: v for k, v in row.items() if k.startswith("best_")},
        "features": config.FEATURES,
        "numeric_features": config.NUMERIC_FEATURES,
        "binary_features": config.BINARY_FEATURES,
        "nominal_features": config.NOMINAL_FEATURES,
        "target": "1 = heart disease (UCI num 1-4), 0 = no disease (num 0)",
        "decision_threshold": 0.5,
        "random_seed": config.RANDOM_SEED,
        "data_sha256": data_hash,
        "git_commit": git_commit(),
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "library_versions": {"scikit-learn": sklearn.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__,
                             "joblib": joblib.__version__,
                             "mlflow": mlflow.__version__},
    }
    (config.MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2))
    pd.DataFrame(summary).to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print(f"[champion] {name} -> {model_path}")
    print(pd.DataFrame(summary).set_index("model").T.to_string())

    if row["test_roc_auc"] < MIN_TEST_ROC_AUC:
        print(f"[gate] FAIL: test ROC-AUC {row['test_roc_auc']} < {MIN_TEST_ROC_AUC}")
        sys.exit(1)
    print(f"[gate] PASS: test ROC-AUC {row['test_roc_auc']} >= {MIN_TEST_ROC_AUC}")


if __name__ == "__main__":
    main()

"""
ml_steps.py
-----------
Core machine-learning logic for the Flood Prediction pipeline.

This module contains pure, framework-agnostic functions ported directly from the
notebook `DecisionTree_Flood_5-fold_imbalance_Tuning.ipynb`. The Airflow DAG
(`flood_ml_pipeline.py`) imports these functions and wires them into tasks.

Stages (mirrors the notebook):
    1. load_data            -> read raw CSV, drop "Unnamed" columns
    2. validate_data        -> structural / quality checks (missing, duplicate cols)
    3. preprocess_data      -> missing-value imputation, IQR outlier capping,
                               label encoding, standard scaling
    4. split_data           -> stratified 80/20 train/test split
    5. tune_model           -> DecisionTree + GridSearchCV (5-fold StratifiedKFold, scoring=f1)
    6. evaluate_model       -> metrics, confusion matrix, comparison table

All artifacts are written to a per-run directory so each Airflow run is isolated.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration (override via Airflow Variables / env if desired)
# --------------------------------------------------------------------------- #
TARGET_COL = "flooding"
PROVINCE_COL = "province"
OUTLIER_COLS = ["MinRain", "MaxRain", "AvgRain", "AvgFloodRiskArea(Square meter)"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5

# Random Forest hyperparameter grid (notebook cell 18)
PARAM_GRID = {
    "n_estimators": [50, 100, 200],          # number of trees in the forest
    "max_depth": [None, 10, 20],             # max depth of each tree
    "min_samples_split": [2, 5, 10],         # min samples required to split a node
    "min_samples_leaf": [1, 2, 4],           # min samples required at a leaf
    "max_features": ["sqrt", "log2"],        # features considered at each split
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def handle_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Cap a column to its IQR bounds (notebook cell 10)."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


# --------------------------------------------------------------------------- #
# 1. Load
# --------------------------------------------------------------------------- #
def load_data(raw_csv_path: str, run_dir: str) -> str:
    """Read the raw CSV, drop unnamed columns, persist a clean parquet."""
    _ensure_dir(run_dir)
    logger.info("Loading raw dataset from %s", raw_csv_path)
    df = pd.read_csv(raw_csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    out = os.path.join(run_dir, "01_loaded.parquet")
    df.to_parquet(out, index=False)
    logger.info("Loaded %d rows x %d cols -> %s", df.shape[0], df.shape[1], out)
    return out


# --------------------------------------------------------------------------- #
# 2. Validate
# --------------------------------------------------------------------------- #
def validate_data(loaded_path: str, run_dir: str) -> Dict:
    """Structural & quality checks (notebook cells 2-6)."""
    df = pd.read_parquet(loaded_path)

    report = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_per_column": {c: int(v) for c, v in df.isnull().sum().items()},
        "duplicate_columns": bool(df.columns.duplicated().any()),
    }

    # Hard requirements
    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found"
    assert PROVINCE_COL in df.columns, f"Column '{PROVINCE_COL}' not found"
    assert df.shape[0] > 0, "Dataset is empty"

    # Class balance of the target
    report["target_distribution"] = {
        str(k): int(v) for k, v in df[TARGET_COL].value_counts().items()
    }

    out = os.path.join(run_dir, "02_validation_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info("Validation report -> %s\n%s", out, json.dumps(report, indent=2, ensure_ascii=False))
    return report


# --------------------------------------------------------------------------- #
# 3. Preprocess
# --------------------------------------------------------------------------- #
def preprocess_data(loaded_path: str, run_dir: str) -> Dict[str, str]:
    """Impute, cap outliers, encode, scale (notebook cells 11-13)."""
    df = pd.read_parquet(loaded_path)

    # 3a. Missing-value handling: median for numeric, mode for categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 3b. Outlier capping via IQR
    for col in OUTLIER_COLS:
        if col in df.columns:
            df = handle_outliers_iqr(df, col)

    # 3c. Label-encode province
    le = LabelEncoder()
    df["province_encoded"] = le.fit_transform(df[PROVINCE_COL])
    df_final = df.drop(columns=[PROVINCE_COL])

    # 3d. Split X / y and standard-scale features
    X = df_final.drop(columns=[TARGET_COL])
    y = df_final[TARGET_COL]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Persist artifacts
    feat_path = os.path.join(run_dir, "03_X_scaled.parquet")
    target_path = os.path.join(run_dir, "03_y.parquet")
    scaler_path = os.path.join(run_dir, "03_scaler.joblib")
    le_path = os.path.join(run_dir, "03_label_encoder.joblib")
    # keep original (unscaled) frame so the evaluation step can build a readable table
    orig_path = os.path.join(run_dir, "03_df_original.parquet")

    X_scaled.to_parquet(feat_path)
    y.to_frame(name=TARGET_COL).to_parquet(target_path)
    df.to_parquet(orig_path, index=True)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, le_path)

    logger.info("Preprocessing complete. Features: %s", list(X_scaled.columns))
    return {
        "features": feat_path,
        "target": target_path,
        "scaler": scaler_path,
        "label_encoder": le_path,
        "original": orig_path,
    }


# --------------------------------------------------------------------------- #
# 4. Split
# --------------------------------------------------------------------------- #
def split_data(features_path: str, target_path: str, run_dir: str) -> Dict[str, str]:
    """Stratified 80/20 split (notebook cell 17)."""
    X = pd.read_parquet(features_path)
    y = pd.read_parquet(target_path)[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    paths = {
        "X_train": os.path.join(run_dir, "04_X_train.parquet"),
        "X_test": os.path.join(run_dir, "04_X_test.parquet"),
        "y_train": os.path.join(run_dir, "04_y_train.parquet"),
        "y_test": os.path.join(run_dir, "04_y_test.parquet"),
    }
    X_train.to_parquet(paths["X_train"])
    X_test.to_parquet(paths["X_test"])
    y_train.to_frame(name=TARGET_COL).to_parquet(paths["y_train"])
    y_test.to_frame(name=TARGET_COL).to_parquet(paths["y_test"])

    logger.info(
        "Split done. Total=%d Train=%d Test=%d", len(X), len(X_train), len(X_test)
    )
    return paths


# --------------------------------------------------------------------------- #
# 5. Tune (GridSearchCV)
# --------------------------------------------------------------------------- #
def tune_model(split_paths: Dict[str, str], run_dir: str) -> Dict:
    """RandomForest + GridSearchCV, 5-fold StratifiedKFold, scoring=f1 (cell 18)."""
    X_train = pd.read_parquet(split_paths["X_train"])
    y_train = pd.read_parquet(split_paths["y_train"])[TARGET_COL]

    rf_clf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    grid_search = GridSearchCV(
        estimator=rf_clf,
        param_grid=PARAM_GRID,
        cv=StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    logger.info("Starting GridSearchCV ...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    model_path = os.path.join(run_dir, "05_best_model.joblib")
    joblib.dump(best_model, model_path)

    result = {
        "best_params": grid_search.best_params_,
        "best_cv_f1": float(grid_search.best_score_),
        "model_path": model_path,
    }
    out = os.path.join(run_dir, "05_tuning_result.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Best params: %s | CV F1=%.4f", result["best_params"], result["best_cv_f1"])
    return result


# --------------------------------------------------------------------------- #
# 6. Evaluate
# --------------------------------------------------------------------------- #
def evaluate_model(
    model_path: str,
    split_paths: Dict[str, str],
    original_path: str,
    run_dir: str,
) -> Dict:
    """Evaluate on the test set + build comparison table (cells 19-20)."""
    model = joblib.load(model_path)
    X_test = pd.read_parquet(split_paths["X_test"])
    y_test = pd.read_parquet(split_paths["y_test"])[TARGET_COL]

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        ),
    }

    metrics_path = os.path.join(run_dir, "06_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    # Comparison table on the original (readable) frame
    df_orig = pd.read_parquet(original_path)
    comparison = df_orig.loc[y_test.index].copy()
    cols_to_show = [c for c in ["province", "month", "AvgRain", "flooding"] if c in comparison.columns]
    comparison = comparison[cols_to_show]
    comparison["Predicted"] = y_pred
    comparison["Is_Correct"] = comparison[TARGET_COL] == comparison["Predicted"]
    comparison_path = os.path.join(run_dir, "06_prediction_comparison.csv")
    comparison.to_csv(comparison_path, index=False)

    n_errors = int((~comparison["Is_Correct"]).sum())
    logger.info(
        "Test metrics: acc=%.4f prec=%.4f recall=%.4f f1=%.4f | errors=%d/%d",
        metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["f1"], n_errors, len(y_test),
    )

    metrics["n_errors"] = n_errors
    metrics["n_test"] = int(len(y_test))
    metrics["comparison_csv"] = comparison_path
    metrics["metrics_json"] = metrics_path
    return metrics

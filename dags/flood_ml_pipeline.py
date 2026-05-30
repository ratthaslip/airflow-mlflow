"""
flood_ml_pipeline.py
--------------------
Airflow DAG that trains a Random Forest flood-prediction model end-to-end,
ported from the notebook `RandomForest_Flood_5-fold_imbalance_Tuning.ipynb`.

Pipeline graph:

    load_data >> validate_data >> preprocess_data >> split_data >> tune_model
        >> evaluate_model >> check_quality >> [register_model | alert_low_quality]

Design notes
------------
* The heavy ML logic lives in `flood_ml/ml_steps.py` (importable, testable).
* Each task passes lightweight artifact *paths* through XCom, not large objects.
* Every DAG run writes artifacts to an isolated directory keyed by run_id, so
  concurrent / historical runs never clobber each other.
* `class_weight='balanced'` + `scoring='f1'` are kept to handle the imbalanced
  flooding target, exactly as in the notebook.

Configuration
-------------
Two Airflow Variables (optional) control I/O; sensible defaults are used:
    flood_raw_csv_path   -> default: /opt/airflow/data/flood_dataset.csv
    flood_artifacts_dir  -> default: /opt/airflow/data/artifacts

The F1 quality gate threshold can be set via:
    flood_min_f1         -> default: 0.50
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

from flood_ml import ml_steps

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration via Airflow Variables (with safe defaults)
# --------------------------------------------------------------------------- #
RAW_CSV_PATH = Variable.get(
    "flood_raw_csv_path", default_var="/opt/airflow/data/flood_dataset.csv"
)
ARTIFACTS_DIR = Variable.get(
    "flood_artifacts_dir", default_var="/opt/airflow/data/artifacts"
)
MIN_F1 = float(Variable.get("flood_min_f1", default_var="0.50"))


def _run_dir(run_id: str) -> str:
    """Isolated artifact directory for a single DAG run."""
    safe = run_id.replace(":", "_").replace("+", "_").replace("/", "_")
    path = os.path.join(ARTIFACTS_DIR, safe)
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Task callables (thin wrappers around ml_steps, handling XCom plumbing)
# --------------------------------------------------------------------------- #
def task_load(**ctx):
    run_dir = _run_dir(ctx["run_id"])
    return ml_steps.load_data(RAW_CSV_PATH, run_dir)


def task_validate(ti, **ctx):
    loaded = ti.xcom_pull(task_ids="load_data")
    run_dir = _run_dir(ctx["run_id"])
    return ml_steps.validate_data(loaded, run_dir)


def task_preprocess(ti, **ctx):
    loaded = ti.xcom_pull(task_ids="load_data")
    run_dir = _run_dir(ctx["run_id"])
    return ml_steps.preprocess_data(loaded, run_dir)


def task_split(ti, **ctx):
    pre = ti.xcom_pull(task_ids="preprocess_data")
    run_dir = _run_dir(ctx["run_id"])
    return ml_steps.split_data(pre["features"], pre["target"], run_dir)


def task_tune(ti, **ctx):
    split_paths = ti.xcom_pull(task_ids="split_data")
    run_dir = _run_dir(ctx["run_id"])
    return ml_steps.tune_model(split_paths, run_dir)


def task_evaluate(ti, **ctx):
    tuning = ti.xcom_pull(task_ids="tune_model")
    split_paths = ti.xcom_pull(task_ids="split_data")
    pre = ti.xcom_pull(task_ids="preprocess_data")
    run_dir = _run_dir(ctx["run_id"])
    return ml_steps.evaluate_model(
        tuning["model_path"], split_paths, pre["original"], run_dir
    )


def task_check_quality(ti, **ctx):
    """Quality gate: branch on test-set F1 score."""
    metrics = ti.xcom_pull(task_ids="evaluate_model")
    f1 = metrics["f1"]
    logger.info("Quality gate: F1=%.4f vs threshold=%.4f", f1, MIN_F1)
    return "register_model" if f1 >= MIN_F1 else "alert_low_quality"


def task_register(ti, **ctx):
    """Promote the run's model to a stable 'latest' location."""
    tuning = ti.xcom_pull(task_ids="tune_model")
    metrics = ti.xcom_pull(task_ids="evaluate_model")
    import shutil

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    latest = os.path.join(ARTIFACTS_DIR, "flood_rf_model_latest.joblib")
    shutil.copyfile(tuning["model_path"], latest)
    logger.info(
        "Registered model -> %s (test F1=%.4f, params=%s)",
        latest, metrics["f1"], tuning["best_params"],
    )
    return latest


# --------------------------------------------------------------------------- #
# DAG definition
# --------------------------------------------------------------------------- #
default_args = {
    "owner": "ratthaslip",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="flood_ml_pipeline",
    description="Random Forest flood-prediction training pipeline (GridSearchCV, 5-fold, imbalanced)",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,          # trigger manually; set e.g. "@weekly" to retrain on a cadence
    catchup=False,
    default_args=default_args,
    tags=["ml", "flood", "random-forest", "ais"],
    doc_md=__doc__,
) as dag:

    load_data = PythonOperator(task_id="load_data", python_callable=task_load)
    validate_data = PythonOperator(task_id="validate_data", python_callable=task_validate)
    preprocess_data = PythonOperator(task_id="preprocess_data", python_callable=task_preprocess)
    split_data = PythonOperator(task_id="split_data", python_callable=task_split)
    tune_model = PythonOperator(task_id="tune_model", python_callable=task_tune)
    evaluate_model = PythonOperator(task_id="evaluate_model", python_callable=task_evaluate)

    check_quality = BranchPythonOperator(
        task_id="check_quality", python_callable=task_check_quality
    )
    register_model = PythonOperator(task_id="register_model", python_callable=task_register)
    alert_low_quality = EmptyOperator(task_id="alert_low_quality")
    done = EmptyOperator(task_id="done", trigger_rule="none_failed_min_one_success")

    (
        load_data
        >> validate_data
        >> preprocess_data
        >> split_data
        >> tune_model
        >> evaluate_model
        >> check_quality
        >> [register_model, alert_low_quality]
        >> done
    )

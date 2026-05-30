# Flood Prediction ML Pipeline (Apache Airflow 2.8.3)

An end-to-end Airflow pipeline that trains a tuned **Random Forest** classifier to
predict flooding events, ported from the notebook
`RandomForest_Flood_5-fold_imbalance_Tuning.ipynb`.

It runs on top of the existing `apache/airflow:2.8.3` CeleryExecutor stack in this
repo — no changes to `docker-compose.yaml` are required.

---

## Files added

```
dags/
  flood_ml_pipeline.py        # the Airflow DAG (task wiring + quality gate)
  flood_ml/
    __init__.py
    ml_steps.py               # pure ML logic (load/validate/preprocess/split/tune/evaluate)
data/
  flood_dataset.csv           # input data (sample provided — replace with the real file)
include/
  generate_sample_data.py     # regenerate the synthetic sample dataset
  run_pipeline_local.py        # run the whole pipeline locally without Airflow (smoke test)
requirements.txt              # + scikit-learn, joblib, pyarrow
```

---

## Pipeline stages

The DAG mirrors the notebook one-to-one. Each task hands the next task **artifact
file paths** through XCom (not big objects), and all artifacts for a run live in an
isolated per-run directory under `data/artifacts/<run_id>/`.

| # | Task              | Notebook cell(s) | What it does |
|---|-------------------|------------------|--------------|
| 1 | `load_data`       | 1                | Read `flood_dataset.csv`, drop `Unnamed:` columns |
| 2 | `validate_data`   | 2–6              | Schema/dtype/missing/duplicate checks, target balance; writes JSON report |
| 3 | `preprocess_data` | 9–13             | Median/mode imputation, IQR outlier capping, label-encode `province`, `StandardScaler` |
| 4 | `split_data`      | 17               | Stratified 80/20 train/test split |
| 5 | `tune_model`      | 18               | `RandomForestClassifier(class_weight='balanced')` + `GridSearchCV`, 5-fold shuffled `StratifiedKFold`, `scoring='f1'` |
| 6 | `evaluate_model`  | 19–20            | Accuracy/precision/recall/F1, confusion matrix, prediction comparison CSV |
| 7 | `check_quality`   | —                | Branch: F1 ≥ threshold → `register_model`, else → `alert_low_quality` |
| 8 | `register_model`  | —                | Copy the best model to `data/artifacts/flood_rf_model_latest.joblib` |

```
load_data → validate_data → preprocess_data → split_data → tune_model
   → evaluate_model → check_quality ─┬─→ register_model ─┐
                                     └─→ alert_low_quality┴─→ done
```

The imbalanced-data handling from the notebook is preserved: `class_weight='balanced'`
on the estimator and `scoring='f1'` for model selection.

---

## How to run

### 1. Install the new dependencies (rebuild the image)
The repo already builds a custom image from `requirements.txt`. Rebuild so
scikit-learn / joblib / pyarrow are baked in:

```bash
docker compose build
docker compose up -d
```

> The `data/` folder is mounted into the containers by adding one line to the
> `volumes:` of `x-airflow-common` in `docker-compose.yaml` (see next section).

### 2. Mount the data folder
Add this line under `x-airflow-common → volumes:` in `docker-compose.yaml`:

```yaml
    - ${AIRFLOW_PROJ_DIR:-.}/data:/opt/airflow/data
```

This makes `data/flood_dataset.csv` available inside the containers at
`/opt/airflow/data/flood_dataset.csv` (the default the DAG looks for) and lets
trained artifacts persist back to your host.

### 3. Provide the dataset
Drop your real `flood_dataset.csv` into `data/`. To regenerate the synthetic
sample instead:

```bash
python include/generate_sample_data.py data/flood_dataset.csv
```

### 4. Trigger the DAG
Open the Airflow UI at <http://localhost:8080> (airflow / airflow), unpause
`flood_ml_pipeline`, and click **Trigger DAG**. Or from the CLI:

```bash
docker compose exec airflow-scheduler airflow dags trigger flood_ml_pipeline
```

---

## Configuration (optional Airflow Variables)

Set these in **Admin → Variables** to override the defaults:

| Variable               | Default                                   | Purpose |
|------------------------|-------------------------------------------|---------|
| `flood_raw_csv_path`   | `/opt/airflow/data/flood_dataset.csv`     | Input CSV path |
| `flood_artifacts_dir`  | `/opt/airflow/data/artifacts`             | Where run artifacts are written |
| `flood_min_f1`         | `0.50`                                    | Quality-gate F1 threshold for model registration |

To retrain on a schedule, change `schedule_interval=None` to e.g. `"@weekly"` in
`flood_ml_pipeline.py`.

---

## Outputs per run (`data/artifacts/<run_id>/`)

- `02_validation_report.json` — data quality report
- `03_scaler.joblib`, `03_label_encoder.joblib` — fitted transformers (for inference)
- `05_best_model.joblib`, `05_tuning_result.json` — tuned model + best params/CV score
- `06_metrics.json` — test-set metrics + confusion matrix + classification report
- `06_prediction_comparison.csv` — actual vs predicted with an error flag

A passing run also publishes `data/artifacts/flood_rf_model_latest.joblib`.

---

## Local smoke test (no Airflow)

```bash
python include/run_pipeline_local.py
```

Runs all six stages against `data/flood_dataset.csv` and prints best params, CV F1,
test metrics, and the confusion matrix — handy for fast iteration before deploying.

---

## Validation status

- DAG parses with **zero import errors** under real `apache-airflow==2.8.3` (Python 3.11, matching the base image).
- Full ML logic verified end-to-end on the sample dataset (all six stages + artifacts).
- All imports use stable Airflow 2.8.3 APIs (`PythonOperator`, `BranchPythonOperator`, `EmptyOperator`, `Variable`).

## Model

This pipeline uses **`RandomForestClassifier`** tuned over `n_estimators`,
`max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`
(`sqrt`/`log2`), with a shuffled 5-fold `StratifiedKFold` and `scoring='f1'`.

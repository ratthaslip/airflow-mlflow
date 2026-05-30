"""Local end-to-end test of the pipeline logic (no Airflow needed)."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dags"))
from flood_ml import ml_steps

RUN_DIR = "/tmp/flood_run_local"
RAW = "data/flood_dataset.csv"

loaded = ml_steps.load_data(RAW, RUN_DIR)
report = ml_steps.validate_data(loaded, RUN_DIR)
pre = ml_steps.preprocess_data(loaded, RUN_DIR)
split_paths = ml_steps.split_data(pre["features"], pre["target"], RUN_DIR)
tuning = ml_steps.tune_model(split_paths, RUN_DIR)
metrics = ml_steps.evaluate_model(tuning["model_path"], split_paths, pre["original"], RUN_DIR)

print("\n=== SUMMARY ===")
print("Best params:", tuning["best_params"])
print("CV F1:", round(tuning["best_cv_f1"], 4))
print("Test:", {k: round(metrics[k], 4) for k in ["accuracy","precision","recall","f1"]})
print("Confusion matrix:", metrics["confusion_matrix"])
print("Errors:", metrics["n_errors"], "/", metrics["n_test"])
print("Artifacts in", RUN_DIR, "->", sorted(os.listdir(RUN_DIR)))

"""Run with: python -m src.regression.knn_baseline (from repository root)."""

import json
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.data.load import load_raw, validate_schema
from src.data.split import split_predictor_groups
from src.preprocessing.regression import build_knn_pipeline, select_features
from src.utils.config import PROJECT_ROOT, load_config, set_seed


def metrics(y, predictions):
    return {
        "MAE": float(mean_absolute_error(y, predictions)),
        "RMSE": float(root_mean_squared_error(y, predictions)),
        "R2": float(r2_score(y, predictions)),
    }


def main():
    config = load_config()
    seed = config["random_seed"]
    set_seed(seed)
    regression = config["regression"]
    df = load_raw(PROJECT_ROOT / config["paths"]["raw_data"])
    validate_schema(df)
    X = select_features(df, regression)
    y = pd.to_numeric(df[regression["target"]], errors="raise")
    if not np.isfinite(y.to_numpy()).all():
        raise ValueError("ADR contains missing or infinite values; investigate first.")
    train_idx, val_idx, test_idx = split_predictor_groups(X, config["split"], seed)
    print(f"Training: {len(train_idx):,}; validation: {len(val_idx):,}; "
          f"reserved test: {len(test_idx):,}", flush=True)
    print("Fitting KNN with training data only...", flush=True)
    pipeline = build_knn_pipeline(regression)
    started = perf_counter()
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    fit_seconds = perf_counter() - started
    print("Predicting validation bookings (in batches to limit memory)...", flush=True)
    started = perf_counter()
    predictions = np.concatenate([
        pipeline.predict(X.iloc[val_idx[start:start + 256]])
        for start in range(0, len(val_idx), 256)
    ])
    predict_seconds = perf_counter() - started
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X.iloc[train_idx], y.iloc[train_idx])
    reference = dummy.predict(X.iloc[val_idx])
    rows = [
        {"model": "Training-mean reference", **metrics(y.iloc[val_idx], reference)},
        {"model": "KNN baseline", **metrics(y.iloc[val_idx], predictions)},
    ]
    table = pd.DataFrame(rows)
    table_dir = PROJECT_ROOT / config["paths"]["tables_dir"]
    artifact_dir = PROJECT_ROOT / config["paths"]["artifacts_dir"]
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]
    for directory in (table_dir, artifact_dir, processed_dir):
        directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_dir / "knn_baseline_validation.csv", index=False)
    pd.DataFrame({
        "source_row": val_idx, "actual_ADR": y.iloc[val_idx].to_numpy(),
        "predicted_ADR": predictions,
    }).to_csv(processed_dir / "knn_validation_predictions.csv", index=False)
    assignments = np.full(len(df), "test", dtype=object)
    assignments[train_idx] = "train"
    assignments[val_idx] = "validation"
    pd.DataFrame({"source_row": np.arange(len(df)), "split": assignments}).to_csv(
        processed_dir / "knn_split_assignments.csv", index=False
    )
    run = {
        "seed": seed, "regression": regression, "split": config["split"],
        "split_method": "GroupShuffleSplit on identical cleaned predictor rows",
        "rows": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "fit_seconds": fit_seconds, "validation_predict_seconds": predict_seconds,
        "test_evaluated": False, "validation_results": rows,
    }
    (table_dir / "knn_baseline_run.json").write_text(
        json.dumps(run, indent=2), encoding="utf-8"
    )
    joblib.dump(pipeline, artifact_dir / "knn_baseline.joblib", compress=3)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Final test set was not evaluated.")
    print("Saved validation metrics to outputs/tables/knn_baseline_validation.csv")


if __name__ == "__main__":
    main()

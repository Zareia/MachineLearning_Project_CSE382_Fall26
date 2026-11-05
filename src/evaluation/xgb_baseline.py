"""Train, cross-validate and save baseline results without scoring test rows."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cse382-matplotlib"))
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             mean_absolute_error, root_mean_squared_error, r2_score)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from src.preprocessing.booking import (TARGETS, ENGINEERED, REPLACED_RAW, DROP_REASONS,
                                       load_h2, task_features)
from src.utils.config import PROJECT_ROOT, load_config, set_seed


def evaluate(task, pipeline, X, y):
    predicted = pipeline.predict(X)
    if task == "regression":
        return {"MAE": mean_absolute_error(y, predicted),
                "RMSE": root_mean_squared_error(y, predicted), "R2": r2_score(y, predicted)}
    probability = pipeline.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, predicted),
            "precision": precision_score(y, predicted, zero_division=0),
            "recall": recall_score(y, predicted, zero_division=0),
            "F1": f1_score(y, predicted, zero_division=0),
            "ROC_AUC": roc_auc_score(y, probability),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def save_confusion_matrices(validation, folds, output_dir):
    names = ["TN", "FP", "FN", "TP"]
    pooled = folds[names].sum()
    rows = [{"evaluation": "validation", **{k: validation[k] for k in names}},
            {"evaluation": "cv_pooled", **pooled.to_dict()}]
    pd.DataFrame(rows).to_csv(output_dir / "xgb_classification_confusion_matrices.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, scores, title in zip(axes, [validation, pooled], ["Validation", f"{len(folds)}-fold CV (pooled)"]):
        matrix = np.asarray([scores[k] for k in names], dtype=int).reshape(2, 2)
        ConfusionMatrixDisplay(matrix, display_labels=["Not canceled", "Canceled"]).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / "xgb_classification_confusion_matrix.png", dpi=180)
    plt.close(fig)


def train_baseline(task, pipeline, data_path, output_dir, artifact_dir):
    output_dir, artifact_dir = Path(output_dir), Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()
    seed, split = config["random_seed"], config["split"]
    set_seed(seed)
    df, audit = load_h2(data_path)
    features = task_features(task)
    X, y = df[features], df[TARGETS[task]]
    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Missing or infinite target values must be investigated before training.")

    # Reserve test first; never pass these rows to fit, predict or evaluation.
    development, test = train_test_split(df.index, test_size=split["test_size"], random_state=seed,
                                         stratify=y if task == "classification" else None)
    train, validation = train_test_split(
        development, test_size=split["val_size"] / (split["train_size"] + split["val_size"]),
        random_state=seed, stratify=y.loc[development] if task == "classification" else None)
    X_train, y_train = X.loc[train], y.loc[train]
    cv_type = StratifiedKFold if task == "classification" else KFold
    cv = cv_type(n_splits=split["cv_folds"], shuffle=True, random_state=seed)
    folds = list(cv.split(X_train, y_train))
    assignments = pd.DataFrame({"split": "test", "cv_fold": 0}, index=df.index)
    assignments.index.name = "source_row"
    assignments.loc[train, "split"] = "train"
    assignments.loc[validation, "split"] = "validation"
    for fold, (_, held_out) in enumerate(folds, 1):
        assignments.loc[train[held_out], "cv_fold"] = fold
    assignments.to_csv(output_dir / f"xgb_{task}_splits.csv")

    started = perf_counter()
    # Limit CPU resources without changing estimator hyperparameters.
    with threadpool_limits(limits=4):
        pipeline.fit(X_train, y_train)
        validation_scores = evaluate(task, pipeline, X.loc[validation], y.loc[validation])
        fold_scores = []
        for fold, (fit_rows, held_out) in enumerate(folds, 1):
            model = clone(pipeline)  # Refit preprocessing as well as XGBoost.
            model.fit(X_train.iloc[fit_rows], y_train.iloc[fit_rows])
            scores = evaluate(task, model, X_train.iloc[held_out], y_train.iloc[held_out])
            fold_scores.append({"fold": fold, **scores})
            print(f"{task}: CV fold {fold}/{len(folds)} complete", flush=True)
    seconds = perf_counter() - started

    cv_results = pd.DataFrame(fold_scores).set_index("fold")
    results = pd.DataFrame({"validation": validation_scores, "cv_mean": cv_results.mean(),
                            "cv_std": cv_results.std(ddof=1)}).rename_axis("metric")
    results.to_csv(output_dir / f"xgb_{task}_metrics.csv")
    cv_results.to_csv(output_dir / f"xgb_{task}_cv_folds.csv")
    joblib.dump(pipeline, artifact_dir / f"xgb_{task}_pipeline.joblib")
    encoded = pipeline.named_steps["preprocessing"].get_feature_names_out().tolist()
    pd.Series(encoded, name="encoded_feature").to_csv(output_dir / f"xgb_{task}_features.csv", index=False)
    if task == "classification":
        save_confusion_matrices(validation_scores, cv_results, output_dir)
    import xgboost
    metadata = {
        "task": task, "target": TARGETS[task], "seed": seed, "split_config": split,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "test_evaluated": False, "cv_scope": "training only", "cv_std_ddof": 1,
        "raw_features": features, "engineered_features": ENGINEERED,
        "raw_replaced_after_engineering": REPLACED_RAW, "encoded_feature_count": len(encoded),
        "dropped_columns": DROP_REASONS[task], "data_audit": audit,
        "training_target_range": {"min": float(y_train.min()), "max": float(y_train.max())},
        "fit_and_evaluation_seconds": seconds, "runtime_thread_limit": 4,
        "model_parameters": {k: ("NaN" if isinstance(v, float) and np.isnan(v) else v)
                             for k, v in pipeline.named_steps["model"].get_params().items()},
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "notebook_sha256": hashlib.sha256((PROJECT_ROOT / "notebooks/Logistic regression_V2.ipynb").read_bytes()).hexdigest(),
        "versions": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib", "PyYAML", "threadpoolctl"]},
    }
    metadata["versions"]["xgboost"] = xgboost.__version__
    (output_dir / f"xgb_{task}_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    write_summary(output_dir)
    print(f"\n{task.upper()}\n{results.to_string(float_format=lambda v: f'{v:.6f}')}\n", flush=True)
    return results


def write_summary(output_dir):
    lines = ["# XGBoost baselines — H2 only", "",
             "## Method", "",
             "Intended prediction point: booking creation. Source attributes generally reflect the day before arrival, "
             "so these exploratory scores do not establish leakage-free booking-time performance.", "",
             "Reuse the current V2 notebook: remove Undefined market/channel rows and Adults <= 0; fill Country with Unknown; "
             "derive TotalGuests, IsFamily, hasCompany, hasAgent and TotalNights; use top-10 Country grouping and the notebook's "
             "feature exclusions. Country grouping is learned from training only. Missing numeric inputs use training medians; "
             "categorical inputs use constant Missing then one-hot encoding with unknown-category handling. "
             "Raw party/stay/agency columns are replaced by derived features. Trees need no scaling.", "",
             "Defaults except random_state from config.yaml (42). No search, early stopping or threshold tuning. "
             "Configured split: 70/15/15, test reserved first; classification is stratified. Five-fold CV uses only training: "
             "shuffled KFold for regression, StratifiedKFold for classification. Report unweighted fold mean and sample "
             "standard deviation (ddof=1). Saved pipelines are fitted on training only. Positive class = canceled (1); "
             "classification uses the default 0.5 threshold. Test rows are never predicted or scored.", "",
             "## Leakage and limitations", "",
             "ADR is calculated from lodging transactions, not established as a booking-time quote; exclude it from "
             "classification. DepositType is calculated from payments before arrival/cancellation; exclude it from both tasks. "
             "Exclude final status fields and realized post-booking changes. Remaining source attributes still need booking-time "
             "availability checks. The inherited row exclusions restrict coverage and do not prove the excluded bookings are errors.", "",
             "Duplicates are retained to match the notebook. Random row splits/CV can share identical or related bookings and be "
             "optimistic. KNN uses a different population, feature set and predictor-group split; these scores cannot rank models "
             "or establish a common untouched team test set. Freeze a shared evaluation policy before model comparison and acknowledge "
             "previous inspections. No target clipping or performance-driven row deletion is applied.", "",
             "Source: [António et al., specifications table and Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/)."]
    for task in ("regression", "classification"):
        path = output_dir / f"xgb_{task}_metadata.json"
        if not path.exists():
            continue
        meta = json.loads(path.read_text())
        table = pd.read_csv(output_dir / f"xgb_{task}_metrics.csv").set_index("metric")
        lines += ["", f"## {task.title()}", "", f"Target: {meta['target']}. Split sizes: {meta['split_sizes']}.", "",
                  "Raw inputs: " + ", ".join(meta["raw_features"]) + ".", "",
                  "Derived features: " + ", ".join(meta["engineered_features"]) + ".", "",
                  "Replaced raw columns: " + ", ".join(meta["raw_replaced_after_engineering"]) + ". "
                  "Exact estimator inputs are in the companion features CSV.", "", "Dropped predictors:", ""]
        lines += [f"- {column}: {reason}" for column, reason in meta["dropped_columns"].items()]
        lines += [f"- {meta['target']}: target, never a predictor.", "",
                  "| Metric | Validation | Training CV mean +/- std |", "|---|---:|---:|"]
        for name, row in table.iterrows():
            lines.append(f"| {name} | {row['validation']:.4f} | {row['cv_mean']:.4f} +/- {row['cv_std']:.4f} |")
        lines += ["", "Cleaning counts: `" + json.dumps(meta["data_audit"]) + "`.", "",
                  "Versions: `" + json.dumps(meta["versions"]) + "`."]
    if all((output_dir / f"xgb_{task}_metrics.csv").exists() for task in TARGETS):
        r = pd.read_csv(output_dir / "xgb_regression_metrics.csv").set_index("metric")
        c = pd.read_csv(output_dir / "xgb_classification_metrics.csv").set_index("metric")
        lines += ["", "## Interpretation (3 lines)", "",
                  f"1. Regression validation MAE is {r.loc['MAE', 'validation']:.2f} rate units, RMSE {r.loc['RMSE', 'validation']:.2f}, "
                  f"and R2 {r.loc['R2', 'validation']:.3f}; larger RMSE shows the influence of larger errors.",
                  f"2. Cancellation validation F1 is {c.loc['F1', 'validation']:.3f}, recall {c.loc['recall', 'validation']:.3f} "
                  f"and ROC-AUC {c.loc['ROC_AUC', 'validation']:.3f} at the default decision threshold.",
                  f"3. Regression CV RMSE is {r.loc['RMSE', 'cv_mean']:.2f} +/- {r.loc['RMSE', 'cv_std']:.2f}; "
                  "unusual target values are retained. These preliminary scores have the timing/duplicate limitations above; "
                  "final test performance is not known."]
    (output_dir / "xgb_baseline_summary.md").write_text("\n".join(lines) + "\n")

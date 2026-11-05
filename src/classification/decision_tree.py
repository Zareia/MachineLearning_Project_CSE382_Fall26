"""Untuned Decision Tree cancellation classification pipeline (Milestone 1 baseline)."""

import json
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import xgb_baseline
from src.preprocessing.booking import build_preprocessor
from src.utils.config import load_config

PREFIX = "decision_tree"


def build_pipeline():
    # Same shared preprocessing as XGBoost; configured parameters are the scikit-learn defaults
    # (no depth limit, pruning, class weighting or threshold tuning), plus the configured seed.
    config = load_config()
    params = config["classification"]["decision_tree"]
    return Pipeline([("preprocessing", build_preprocessor("classification")),
                     ("model", DecisionTreeClassifier(random_state=config["random_seed"], **params))])


def train(data_path, output_dir, artifact_dir):
    """Shared team runner (train fit, validation, training-only CV; test never scored) + tree size."""
    pipeline = build_pipeline()
    results = xgb_baseline.train_baseline("classification", pipeline, data_path, output_dir, artifact_dir,
                                          prefix=PREFIX)
    tree = pipeline.named_steps["model"]  # fitted in place on the training split by the runner
    path = Path(output_dir) / f"{PREFIX}_classification_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["fitted_tree"] = {"depth": int(tree.get_depth()), "leaves": int(tree.get_n_leaves()),
                               "fitted_on": "training split only"}
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return results

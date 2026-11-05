"""Shared paths, settings and random seeds used by the team's model modules."""

from pathlib import Path
import random

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path is not None else PROJECT_ROOT / "config.yaml"
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"random_seed", "paths", "split", "regression", "classification"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"Missing required configuration sections: {path}")
    fractions = [config["split"][key] for key in ("train_size", "val_size", "test_size")]
    if min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("Split fractions must be positive and sum to one.")
    if config["split"]["cv_folds"] < 2:
        raise ValueError("At least two CV folds are required.")
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

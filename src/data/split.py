"""Reproducible holdouts that keep identical predictor rows together."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_predictor_groups(X, split_config, seed):
    """Return positional indices; fractions apply to groups, not exact row counts.

    Group using X only: neither ADR nor future outcomes decide the split.
    Hash collisions would conservatively keep additional rows together.
    """
    train = split_config["train_size"]
    val = split_config["val_size"]
    test = split_config["test_size"]
    if min(train, val, test) <= 0 or not np.isclose(train + val + test, 1):
        raise ValueError("Split fractions must be positive and sum to one.")
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=test, random_state=seed)
    development, test_idx = next(first.split(X, groups=groups))
    second = GroupShuffleSplit(
        n_splits=1, test_size=val / (train + val), random_state=seed
    )
    train_rel, val_rel = next(
        second.split(X.iloc[development], groups=groups[development])
    )
    return development[train_rel], development[val_rel], test_idx

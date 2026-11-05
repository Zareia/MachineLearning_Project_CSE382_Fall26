import numpy as np
import pandas as pd

from src.data.split import split_predictor_groups
from src.preprocessing.regression import build_knn_pipeline, select_features


def test_identical_predictors_stay_in_one_partition():
    X = pd.DataFrame({"number": np.repeat(np.arange(40), 3)})
    config = {"train_size": .7, "val_size": .15, "test_size": .15}
    parts = split_predictor_groups(X, config, 42)
    groups = [set(X.iloc[part]["number"]) for part in parts]
    assert sum(len(part) for part in parts) == len(X)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    again = split_predictor_groups(X, config, 42)
    assert all(np.array_equal(a, b) for a, b in zip(parts, again))


def test_pipeline_uses_training_statistics_and_accepts_unseen_categories():
    config = {
        "target": "ADR", "features": ["Adults", "Meal"],
        "categorical_features": ["Meal"],
        "knn": {"n_neighbors": 1, "weights": "uniform", "p": 2},
    }
    train = select_features(pd.DataFrame({
        "Adults": [1., 3., np.nan], "Meal": ["BB  ", "HB", None],
    }), config)
    pipeline = build_knn_pipeline(config).fit(train, [50., 100., 75.])
    validation = select_features(pd.DataFrame({
        "Adults": [1000., np.nan], "Meal": ["new category", "NULL   "],
    }), config)
    predictions = pipeline.predict(validation)
    numeric = pipeline.named_steps["preprocessing"].named_transformers_["numeric"]
    assert numeric.named_steps["impute"].statistics_[0] == 2.
    assert numeric.named_steps["scale"].mean_[0] == 2.
    assert np.isfinite(predictions).all()

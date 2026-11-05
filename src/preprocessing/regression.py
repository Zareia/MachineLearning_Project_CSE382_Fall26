"""Feature preparation and a reusable KNN regression pipeline."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def select_features(df, config):
    """Stateless formatting only; fitted operations belong inside the pipeline."""
    features = config["features"]
    if config["target"] in features:
        raise ValueError("The target must not be included among the inputs.")
    categorical = config["categorical_features"]
    if not set(categorical).issubset(features):
        raise ValueError("Categorical features must be in the feature list.")
    X = df.loc[:, features].copy()
    for column in categorical:
        cleaned = X[column].astype("string").str.strip().replace(
            {"": pd.NA, "NULL": pd.NA}
        )
        X[column] = cleaned.astype(object).where(cleaned.notna(), np.nan)
    return X


def build_knn_pipeline(config):
    categorical = config["categorical_features"]
    numeric = [c for c in config["features"] if c not in categorical]
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", KNeighborsRegressor(algorithm="brute", **config["knn"])),
    ])

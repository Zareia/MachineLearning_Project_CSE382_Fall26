"""Untuned XGBoost cancellation classification pipeline."""

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing.booking import build_preprocessor
from src.utils.config import load_config


def build_pipeline():
    # No search, early stopping, class reweighting or threshold optimization.
    return Pipeline([("preprocessing", build_preprocessor("classification")),
                     ("model", XGBClassifier(random_state=load_config()["random_seed"]))])

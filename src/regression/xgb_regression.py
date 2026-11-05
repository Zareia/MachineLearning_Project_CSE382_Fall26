"""Untuned XGBoost ADR regression pipeline."""

from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.preprocessing.booking import build_preprocessor
from src.utils.config import load_config


def build_pipeline():
    # All estimator parameters are defaults except the fixed configured seed (42).
    return Pipeline([("preprocessing", build_preprocessor("regression")),
                     ("model", XGBRegressor(random_state=load_config()["random_seed"]))])

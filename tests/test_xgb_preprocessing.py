import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier, XGBRegressor

from src.preprocessing.booking import (CATEGORICAL, DROP_REASONS, BookingFeatures,
                                       build_preprocessor, task_features)
from src.regression.xgb_regression import build_pipeline as regression_pipeline
from src.classification.xgb_classification import build_pipeline as classification_pipeline


def example(task="regression", n=12):
    return pd.DataFrame({c: ([f"C{i:02d}" for i in range(n)] if c in CATEGORICAL else np.arange(n, dtype=float)+1)
                         for c in task_features(task)})


def test_train_only_country_grouping_and_imputation():
    train = example()
    train["LeadTime"] = [np.nan] + [10.] * 11
    preprocessing = build_preprocessor("regression").fit(train)
    held_out = train.iloc[[0]].copy()
    held_out["Country"], held_out["LeadTime"] = "HELD_OUT_ONLY", 1000000.
    features = preprocessing.named_steps["features"]
    countries = features.top_countries_.copy()
    assert len(countries) == 10
    assert features.transform(held_out)["Country"].tolist() == ["Other"]
    preprocessing.transform(held_out)
    assert features.top_countries_ == countries
    assert preprocessing.named_steps["columns"].named_transformers_["numeric"].statistics_[0] == 10.
    assert not any("HELD_OUT_ONLY" in name for name in preprocessing.get_feature_names_out())


def test_notebook_features_and_null_meaning():
    train = example(n=2)
    train["Adults"], train["Children"], train["Babies"] = [2, 1], [1, 0], [0, 0]
    train["Agent"], train["Company"], train["Country"] = [None, "9"], ["NULL", "40"], [None, " GBR "]
    output = BookingFeatures().fit_transform(train)
    assert output["TotalGuests"].tolist() == [3, 1]
    assert output["IsFamily"].tolist() == [1, 0]
    assert output["hasAgent"].tolist() == [0, 1]
    assert output["hasCompany"].tolist() == [0, 1]
    assert output["Country"].tolist() == ["Unknown", "GBR"]


@pytest.mark.parametrize("task,target", [("regression", "ADR"), ("classification", "IsCanceled")])
def test_outcomes_and_excluded_features_never_reach_estimator(task, target):
    train = example(task)
    for column in list(DROP_REASONS[task]) + [target]:
        train[column] = 1
    preprocessing = build_preprocessor(task).fit(train)
    encoded = [name.split("__", 1)[1] for name in preprocessing.get_feature_names_out()]
    for column in list(DROP_REASONS[task]) + [target]:
        assert not any(name == column or name.startswith(column + "_") for name in encoded)


def test_default_models_except_seed():
    for builder, model in [(regression_pipeline, XGBRegressor), (classification_pipeline, XGBClassifier)]:
        # repr handles the NaN missing-value parameter (NaN != NaN).
        assert repr(builder().named_steps["model"].get_params()) == repr(model(random_state=42).get_params())

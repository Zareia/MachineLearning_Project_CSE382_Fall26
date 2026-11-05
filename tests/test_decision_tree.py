"""Decision Tree baseline: shared preprocessing, leakage exclusions, outputs and CLI routing."""

import json
import sys
from contextlib import redirect_stdout
from io import StringIO

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import main
from src.classification import decision_tree
from src.evaluation import xgb_baseline as runner
from src.preprocessing.booking import DROP_REASONS, TARGETS, build_preprocessor, task_features
from tests.test_xgb_preprocessing import example


def labelled(n=60):
    df = example("classification", n=n)
    df.index += 1000
    df["IsCanceled"] = [0, 1] * (n // 2)
    return df


def test_pipeline_uses_shared_preprocessing_and_default_tree():
    pipeline = decision_tree.build_pipeline()
    assert list(pipeline.named_steps) == ["preprocessing", "model"]
    assert repr(pipeline.named_steps["preprocessing"]) == repr(build_preprocessor("classification"))
    assert pipeline.named_steps["model"].get_params() == DecisionTreeClassifier(random_state=42).get_params()


def test_target_and_excluded_columns_never_reach_the_tree():
    train = labelled(20)
    for column in DROP_REASONS["classification"]:
        train[column] = 1
    assert "ADR" in DROP_REASONS["classification"] and "DepositType" in DROP_REASONS["classification"]
    pipeline = decision_tree.build_pipeline().fit(train.drop(columns="IsCanceled"), train["IsCanceled"])
    encoded = [n.split("__", 1)[1] for n in pipeline.named_steps["preprocessing"].get_feature_names_out()]
    for column in list(DROP_REASONS["classification"]) + ["IsCanceled"]:
        assert not any(n == column or n.startswith(column + "_") for n in encoded)
    assert pipeline.named_steps["model"].n_features_in_ == len(encoded)


def test_missing_values_and_unseen_categories_predict():
    train = labelled(20)
    train.loc[train.index[:3], "LeadTime"] = np.nan
    train.loc[train.index[:3], "Meal"] = np.nan
    pipeline = decision_tree.build_pipeline().fit(train[task_features("classification")], train["IsCanceled"])
    new = train[task_features("classification")].iloc[:2].copy()
    new["Meal"], new["Country"], new["Children"] = "NEVER_SEEN", "ZZZ", np.nan
    proba = pipeline.predict_proba(new)
    assert np.isfinite(proba).all() and np.allclose(proba.sum(axis=1), 1)
    assert set(pipeline.predict(new)) <= {0, 1}


def test_outputs_metadata_and_reserved_test(tmp_path, monkeypatch):
    df = labelled()
    monkeypatch.setattr(runner, "load_h2", lambda path: (df, {"retained_rows": len(df)}))
    fitted, scored = [], []
    original_fit, original_evaluate = Pipeline.fit, runner.evaluate

    def observe_fit(self, X, y=None, **kwargs):
        if "model" in self.named_steps:
            fitted.append(set(X.index))
        return original_fit(self, X, y, **kwargs)

    def observe_evaluate(task, pipeline, X, y):
        scored.append(set(X.index))
        return original_evaluate(task, pipeline, X, y)

    monkeypatch.setattr(Pipeline, "fit", observe_fit)
    monkeypatch.setattr(runner, "evaluate", observe_evaluate)
    source, out, artifacts = tmp_path / "H2.csv", tmp_path / "outputs", tmp_path / "artifacts"
    df.to_csv(source, index=False)
    with redirect_stdout(StringIO()):
        decision_tree.train(source, out, artifacts)

    stem = "decision_tree_classification"
    for suffix in ["metrics.csv", "cv_folds.csv", "features.csv", "splits.csv", "confusion_matrices.csv",
                   "confusion_matrix.png", "metadata.json"]:
        assert (out / f"{stem}_{suffix}").stat().st_size > 0
    assert not list(out.glob("xgb_*"))
    test_rows = set(pd.read_csv(out / f"{stem}_splits.csv").query("split == 'test'")["source_row"])
    assert test_rows and all(part.isdisjoint(test_rows) for part in fitted + scored)

    meta = json.loads((out / f"{stem}_metadata.json").read_text())
    assert meta["test_evaluated"] is False and meta["cv_scope"] == "training only"
    assert meta["model_parameters"]["random_state"] == 42 and meta["model_parameters"]["max_depth"] is None
    assert meta["fitted_tree"]["depth"] >= 1 and meta["fitted_tree"]["leaves"] >= 2
    assert "xgboost" not in meta["versions"]

    restored = joblib.load(artifacts / f"{stem}_pipeline.joblib")
    validation = pd.read_csv(out / f"{stem}_splits.csv").query("split == 'validation'")["source_row"]
    expected = pd.read_csv(out / f"{stem}_metrics.csv").set_index("metric")["validation"]
    actual = original_evaluate("classification", restored, df.loc[validation, task_features("classification")],
                               df.loc[validation, TARGETS["classification"]])
    for metric, value in actual.items():
        np.testing.assert_allclose(value, expected[metric])


@pytest.mark.parametrize("task,model,expected", [
    ("classification", "xgboost", ("XGBClassifier", "xgb")),
    ("regression", "xgboost", ("XGBRegressor", "xgb")),
])
def test_existing_xgboost_commands_unchanged(task, model, expected, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "train_baseline", lambda t, pipeline, *paths, prefix="xgb":
                        calls.append((type(pipeline.named_steps["model"]).__name__, prefix)))
    monkeypatch.setattr(sys, "argv", ["main.py", "train", "--task", task, "--model", model])
    main.main()
    assert calls == [expected]


def test_decision_tree_command_routes_and_rejects_regression(monkeypatch):
    calls = []
    monkeypatch.setattr(decision_tree, "train", lambda *paths: calls.append(paths))
    monkeypatch.setattr(sys, "argv", ["main.py", "train", "--task", "classification", "--model", "decision_tree"])
    main.main()
    assert len(calls) == 1
    monkeypatch.setattr(sys, "argv", ["main.py", "train", "--task", "regression", "--model", "decision_tree"])
    with pytest.raises(SystemExit):
        main.main()

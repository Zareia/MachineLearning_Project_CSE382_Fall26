"""Observe every fit/score call to verify the reserved-test boundary."""

from contextlib import redirect_stdout
from io import StringIO

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline

from src.evaluation import xgb_baseline as runner
from src.preprocessing.booking import TARGETS, build_preprocessor, task_features
from tests.test_xgb_preprocessing import example


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_folds_and_saved_model_respect_holdouts(task, tmp_path, monkeypatch):
    df = example(task, n=60)
    df.index += 1000
    df["ADR"], df["IsCanceled"] = np.arange(60) + 50., [0, 1] * 30
    monkeypatch.setattr(runner, "load_h2", lambda path: (df, {"retained_rows": 60}))
    fits, scores = [], []
    original_fit, original_evaluate = Pipeline.fit, runner.evaluate

    def observe_fit(self, X, y=None, **kwargs):
        if "model" in self.named_steps:
            fits.append(set(X.index))
        return original_fit(self, X, y, **kwargs)

    def observe_evaluate(task, pipeline, X, y):
        scores.append(set(X.index))
        return original_evaluate(task, pipeline, X, y)

    monkeypatch.setattr(Pipeline, "fit", observe_fit)
    monkeypatch.setattr(runner, "evaluate", observe_evaluate)
    source = tmp_path / "H2.csv"
    df.to_csv(source, index=False)
    out, artifacts = tmp_path / "outputs", tmp_path / "artifacts"
    model = DummyRegressor() if task == "regression" else DummyClassifier()
    pipeline = Pipeline([("preprocessing", build_preprocessor(task)), ("model", model)])
    with redirect_stdout(StringIO()):
        runner.train_baseline(task, pipeline, source, out, artifacts)
    assignments = pd.read_csv(out / f"xgb_{task}_splits.csv").set_index("source_row")
    groups = {part: set(assignments.index[assignments['split'].eq(part)])
              for part in ("train", "validation", "test")}
    assert set.union(*groups.values()) == set(df.index)
    assert len(fits) == len(scores) == 6
    assert fits[0] == groups["train"] and scores[0] == groups["validation"]
    for fit, held_out in zip(fits[1:], scores[1:]):
        assert fit.isdisjoint(held_out) and fit | held_out == groups["train"]
    assert all(part.isdisjoint(groups["test"]) for part in fits + scores)
    assert set.union(*scores[1:]) == groups["train"]
    assert sum(map(len, scores[1:])) == len(groups["train"])
    restored = joblib.load(artifacts / f"xgb_{task}_pipeline.joblib")
    ids = sorted(groups["validation"])
    expected = pd.read_csv(out / f"xgb_{task}_metrics.csv").set_index("metric")
    actual = original_evaluate(task, restored, df.loc[ids, task_features(task)], df.loc[ids, TARGETS[task]])
    for metric, value in actual.items():
        np.testing.assert_allclose(value, expected.loc[metric, "validation"])

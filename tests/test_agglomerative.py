"""Agglomerative clustering: feature validation, degenerate metrics, outputs, reproducibility and CLI."""

import copy
import json
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

import main
from src.clustering import agglomerative as agg
from src.evaluation.clustering import cluster_metrics
from src.utils.config import load_config

FEATURES = ["LeadTime", "StaysInWeekendNights", "StaysInWeekNights", "Adults", "ADR"]


def bookings(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "LeadTime": rng.integers(0, 300, n).astype(float), "StaysInWeekendNights": rng.integers(0, 3, n),
        "StaysInWeekNights": rng.integers(0, 6, n), "Adults": rng.integers(1, 4, n),
        "ADR": rng.gamma(9, 12, n).round(2), "Children": rng.choice([0.0, 1.0, np.nan], n, p=[.85, .1, .05]),
        "MarketSegment": rng.choice(["Online TA", "Groups", "Direct"], n),
        "CustomerType": rng.choice(["Transient", "Contract"], n), "DepositType": rng.choice(["No Deposit", "Non Refund"], n),
        "Meal": rng.choice(["BB", "HB", np.nan], n), "IsRepeatedGuest": rng.integers(0, 2, n),
        "TotalOfSpecialRequests": rng.integers(0, 3, n), "IsCanceled": rng.integers(0, 2, n)})
    df.index += 5000
    return df


@pytest.fixture
def setup(tmp_path, monkeypatch):
    df = bookings()
    monkeypatch.setattr(agg, "load_h2", lambda path: (df, {"retained_rows": len(df)}))
    config = copy.deepcopy(load_config())
    settings = config["clustering"]["agglomerative"]
    settings.update(sample_size=150, stability_runs=2, dendrogram_leaves=10)
    settings["search"].update(n_clusters=[2, 3, 4], linkage=["ward", "average", "single"], metric=["euclidean", "cosine"])
    dirs = {k: tmp_path / k for k in ("tables", "figures", "artifacts")}
    return config, dirs


def run(config, dirs, **flags):
    return agg.run("unused.csv", dirs["tables"], dirs["figures"], dirs["artifacts"], config=config, **flags)


def test_configured_features_and_outcome_rejection():
    assert load_config()["clustering"]["agglomerative"]["features"] == FEATURES
    agg.check_features(FEATURES)
    for bad in (FEATURES + ["IsCanceled"], ["ReservationStatus"], ["ReservationStatusDate"], ["NotAColumn"], []):
        with pytest.raises(ValueError):
            agg.check_features(bad)


def test_metrics_undefined_for_degenerate_labelings():
    X = np.random.default_rng(0).normal(size=(10, 2))
    for labels in (np.zeros(10), np.arange(10)):
        m = cluster_metrics(X, labels, "manhattan")
        assert m["silhouette"] is None and m["silhouette_manhattan"] is None and m["davies_bouldin"] is None
        assert m["note"]
    with pytest.raises(ValueError):
        cluster_metrics(X, np.zeros(9))


def test_metrics_on_obvious_clusters():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, .1, (30, 2)), rng.normal(5, .1, (30, 2))]
    labels = agg.fit_labels(X, 2, "complete", "manhattan")
    m = cluster_metrics(X, labels, "manhattan")
    assert m["sizes"] == {0: 30, 1: 30} and m["silhouette"] > .9 and m["silhouette_manhattan"] > .9
    assert m["davies_bouldin"] < .1 and m["min_cluster_fraction"] == .5


def test_ward_only_euclidean_and_degenerate_configs_rejected():
    pairs = agg.valid_configs({"linkage": ["ward", "average"], "metric": ["euclidean", "cosine"]})
    assert ("ward", "cosine") not in pairs and {("ward", "euclidean"), ("average", "cosine")} <= set(pairs)
    grid = pd.DataFrame([
        {"n_clusters": 2, "linkage": "single", "metric": "euclidean", "silhouette": .9, "davies_bouldin": .1,
         "min_cluster_fraction": .001},
        {"n_clusters": 3, "linkage": "ward", "metric": "euclidean", "silhouette": .4, "davies_bouldin": .9,
         "min_cluster_fraction": .2},
        {"n_clusters": 4, "linkage": "average", "metric": "cosine", "silhouette": None, "davies_bouldin": None,
         "min_cluster_fraction": 0.0}])
    assert agg.select_best(grid, .02) == {"n_clusters": 3, "linkage": "ward", "metric": "euclidean"}


def test_preprocessing_handles_missing_values():
    df = bookings(50)
    df.loc[df.index[:5], "ADR"] = np.nan
    X = agg.build_preprocessor().fit_transform(df[FEATURES])
    assert np.isfinite(X).all() and np.allclose(X.mean(axis=0), 0)


def test_default_run_is_the_untuned_baseline_only(setup):
    config, dirs = setup
    results = run(config, dirs)
    assert list(results) == ["baseline"] and results["baseline"]["params"] == config["clustering"]["agglomerative"]["baseline"]
    tables, figures, artifacts = dirs["tables"], dirs["figures"], dirs["artifacts"]
    for path in [tables / "agglomerative_metrics.csv", tables / "agglomerative_baseline_profiles.csv",
                 tables / "agglomerative_results.json", tables / "agglomerative_metadata.json",
                 figures / "agglomerative_dendrogram.png", figures / "agglomerative_pca_2d.png",
                 figures / "agglomerative_cluster_sizes.png", artifacts / "agglomerative_sample_labels.csv",
                 artifacts / "agglomerative_preprocessor.joblib"]:
        assert path.stat().st_size > 0, path
    for absent in ["agglomerative_search.csv", "agglomerative_stability.csv", "agglomerative_optimized_profiles.csv"]:
        assert not (tables / absent).exists()
    meta = json.loads((tables / "agglomerative_metadata.json").read_text())
    assert meta["predict_supported"] is False and meta["search_run"] is False and meta["test_split_used"] is False
    assert meta["features"] == FEATURES and meta["sample"]["size"] == 150
    labels = pd.read_csv(artifacts / "agglomerative_sample_labels.csv", index_col="source_row")
    assert list(labels.columns) == FEATURES + ["baseline_cluster"]
    assert pd.read_csv(tables / "agglomerative_baseline_profiles.csv")["bookings"].sum() == 150
    restored = joblib.load(artifacts / "agglomerative_preprocessor.joblib")
    assert np.isfinite(restored.transform(labels[FEATURES])).all()


def test_reproducible_with_fixed_seed(setup, tmp_path):
    config, dirs = setup
    first = run(config, dirs)
    labels_first = (dirs["artifacts"] / "agglomerative_sample_labels.csv").read_text()
    second_dirs = {k: tmp_path / f"again_{k}" for k in dirs}
    second = run(config, second_dirs)
    assert labels_first == (second_dirs["artifacts"] / "agglomerative_sample_labels.csv").read_text()
    for key in ("silhouette", "davies_bouldin", "sizes"):
        assert first["baseline"][key] == second["baseline"][key]


def test_optional_search_and_stability(setup):
    config, dirs = setup
    results = run(config, dirs, run_search=True, run_stability=True)
    assert set(results) == {"baseline", "optimized"}
    assert results["optimized"]["min_cluster_fraction"] >= .02
    stability = pd.read_csv(dirs["tables"] / "agglomerative_stability.csv")
    assert sorted(stability["seed"].unique()) == [43, 44] and (stability["n_shared"] > 0).all()
    # 5 valid pairs (ward/euclidean + average and single with 2 metrics) x 3 cluster counts; ward/cosine skipped
    assert len(pd.read_csv(dirs["tables"] / "agglomerative_search.csv")) == 5 * 3
    assert (dirs["tables"] / "agglomerative_optimized_profiles.csv").exists()
    assert json.loads((dirs["tables"] / "agglomerative_metadata.json").read_text())["predict_supported"] is False


@pytest.mark.parametrize("flags,expected", [([], (False, False)), (["--search", "--stability"], (True, True))])
def test_cluster_command_routes_to_agglomerative(flags, expected, monkeypatch):
    calls = []
    monkeypatch.setattr(agg, "run", lambda *paths, run_search, run_stability: calls.append((run_search, run_stability)))
    monkeypatch.setattr(sys, "argv", ["main.py", "cluster", "--algorithm", "agglomerative", *flags])
    main.main()
    assert calls == [expected]

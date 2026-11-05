"""Agglomerative hierarchical clustering of H2 bookings.

Default (Milestone 1): the fixed, untuned baseline from config.yaml on a seeded random sample.
Optional: --search (parameter grid with unsupervised selection) and --stability (ARI across resamples).
Outcome columns are never inputs or selection criteria. AgglomerativeClustering has no `predict`
for unseen bookings, so the saved outputs are the sample's labels, profiles and metrics.
"""

import hashlib
import itertools
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.data.load import SOURCE_COLUMNS
from src.evaluation.clustering import cluster_metrics
from src.preprocessing.booking import load_h2
from src.utils.config import load_config, set_seed

OUTCOME_COLUMNS = {"IsCanceled", "ReservationStatus", "ReservationStatusDate"}
SCIPY_METRIC = {"euclidean": "euclidean", "manhattan": "cityblock", "cosine": "cosine"}


def check_features(features):
    """Known source columns only, and never an outcome column."""
    unknown = sorted(set(features) - set(SOURCE_COLUMNS))
    if unknown or not features:
        raise ValueError(f"Clustering features must be H2 source columns; unknown or empty: {unknown}")
    outcomes = sorted(OUTCOME_COLUMNS.intersection(features))
    if outcomes:
        raise ValueError(f"Outcome columns must not be clustering inputs: {outcomes}")


def log1p_nonneg(X):
    """log(1 + x) after clipping at 0, so long right tails (LeadTime, ADR) do not dominate distances."""
    return np.log1p(np.clip(X, 0, None))


def build_preprocessor():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("log1p", FunctionTransformer(log1p_nonneg, feature_names_out="one-to-one")),
                     ("scale", StandardScaler())])


def draw_sample(df, n, seed):
    """Simple random sample without replacement (numpy RandomState(seed) via DataFrame.sample)."""
    return df.sample(n=min(n, len(df)), random_state=seed)


def fingerprint(index):
    return hashlib.sha256(np.asarray(index, dtype=np.int64).tobytes()).hexdigest()[:16]


def fit_labels(X, n_clusters, linkage, metric, memory=None):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric,
                                    memory=memory, compute_full_tree=True)
    return model.fit_predict(X)


def valid_configs(space):
    """(linkage, metric) pairs scikit-learn accepts: Ward requires Euclidean distance."""
    return [(l, m) for l, m in itertools.product(space["linkage"], space["metric"]) if l != "ward" or m == "euclidean"]


def evaluate(X, labels, params):
    return {"params": dict(params), **cluster_metrics(X, labels, params["metric"])}


def search(X, space):
    """Every valid (linkage, metric, n_clusters); the tree is cached per (linkage, metric)."""
    rows, cache = [], tempfile.mkdtemp(prefix="agglomerative_cache_")
    try:
        for linkage, metric in valid_configs(space):
            for k in space["n_clusters"]:
                params = {"n_clusters": k, "linkage": linkage, "metric": metric}
                started = perf_counter()
                labels = fit_labels(X, **params, memory=cache)
                seconds = perf_counter() - started
                m = cluster_metrics(X, labels, metric)
                rows.append({**params, "fit_seconds": seconds, "silhouette": m["silhouette"],
                             "silhouette_fit_metric": m[f"silhouette_{metric}"], "davies_bouldin": m["davies_bouldin"],
                             "min_cluster_fraction": m["min_cluster_fraction"], "note": m["note"]})
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    return pd.DataFrame(rows)


def select_best(grid, min_cluster_fraction):
    """Highest Euclidean silhouette (Davies-Bouldin tie-break) among configs whose every cluster holds at
    least `min_cluster_fraction` of the sample; falls back to all defined configs if none qualify."""
    defined = grid.dropna(subset=["silhouette"])
    eligible = defined[defined["min_cluster_fraction"] >= min_cluster_fraction]
    best = (eligible if len(eligible) else defined).sort_values(["silhouette", "davies_bouldin"],
                                                                  ascending=[False, True]).iloc[0]
    return {"n_clusters": int(best["n_clusters"]), "linkage": best["linkage"], "metric": best["metric"]}


def stability(df, sample, labels, preprocessor, features, params, runs, sample_size, seed):
    """ARI between the reference labels and refits on `runs` other random samples (seeds seed+1..seed+runs),
    computed on the bookings both samples share. Clusters are matched by row index; ARI ignores label names."""
    reference = pd.Series(labels, index=sample.index)
    per_run = []
    for r in range(1, runs + 1):
        other = draw_sample(df, sample_size, seed + r)
        refit = pd.Series(fit_labels(preprocessor.transform(other[features]), **params), index=other.index)
        shared = reference.index.intersection(other.index)
        ari = float(adjusted_rand_score(reference[shared], refit[shared])) if len(shared) > 1 else None
        per_run.append({"run": r, "seed": seed + r, "sample_fingerprint": fingerprint(other.index),
                        "n_shared": int(len(shared)), "ari": ari})
    aris = np.array([p["ari"] for p in per_run if p["ari"] is not None])
    summary = ({"mean": float(aris.mean()), "median": float(np.median(aris)), "min": float(aris.min()),
                "max": float(aris.max())} if len(aris) else None)
    return {"runs": runs, "summary": summary, "per_run": per_run}


def profile(sample, labels, features, profile_columns):
    """Per-cluster statistics on ORIGINAL values; for interpretation only."""
    data = sample.assign(cluster=labels)
    groups = data.groupby("cluster")
    table = pd.DataFrame({"bookings": groups.size(), "share": groups.size() / len(data)})
    for column in features:
        table[f"{column}_mean"] = groups[column].mean()
        table[f"{column}_median"] = groups[column].median()
    for column in profile_columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            table[f"{column}_mean"] = groups[column].mean()
        else:
            table[f"{column}_top"] = groups[column].agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
            table[f"{column}_top_share"] = groups[column].agg(
                lambda s: s.value_counts(normalize=True).iat[0] if s.notna().any() else 0.0)
    # Computed after clustering; never an input or a selection criterion.
    table["IsCanceled_rate_posthoc"] = groups["IsCanceled"].mean()
    return table.round(4)


def run(data_path, tables_dir, figures_dir, artifact_dir, run_search=False, run_stability=False, config=None):
    config = config or load_config()
    seed = config["random_seed"]
    settings = config["clustering"]["agglomerative"]
    features = list(settings["features"])
    check_features(features)
    set_seed(seed)

    df, audit = load_h2(data_path)  # Team eligibility rules; duplicates retained.
    sample = draw_sample(df, settings["sample_size"], seed)
    preprocessor = build_preprocessor().fit(sample[features])
    X = preprocessor.transform(sample[features])
    print(f"Agglomerative: {len(sample):,} of {len(df):,} eligible bookings (seed {seed}); features {features}", flush=True)

    baseline = dict(settings["baseline"])
    started = perf_counter()
    labels = {"baseline": fit_labels(X, **baseline)}
    results = {"baseline": {**evaluate(X, labels["baseline"], baseline), "fit_seconds": perf_counter() - started}}

    grid = None
    if run_search:
        space = settings["search"]
        started = perf_counter()
        grid = search(X, space)
        best = select_best(grid, space["min_cluster_fraction"])
        labels["optimized"] = fit_labels(X, **best)
        results["optimized"] = {**evaluate(X, labels["optimized"], best),
                                "search_seconds": perf_counter() - started, "configs_evaluated": len(grid),
                                "selection": "max Euclidean silhouette, Davies-Bouldin tie-break, every cluster >= "
                                             f"{space['min_cluster_fraction']:.0%} of the sample; no labels used"}
    if run_stability:
        for name, result in results.items():
            result["stability"] = stability(df, sample, labels[name], preprocessor, features, result["params"],
                                            settings["stability_runs"], settings["sample_size"], seed)

    shown = "optimized" if run_search else "baseline"
    metadata = {
        "algorithm": "AgglomerativeClustering", "features": features,
        "preprocessing": "median impute -> log1p(clip at 0) -> StandardScaler, fitted on the sample",
        "population": {"source": "src.preprocessing.booking.load_h2 (team eligibility; duplicates retained)",
                       "eligible_rows": len(df), "data_audit": audit},
        "sample": {"size": len(sample), "seed": seed, "method": "simple random, no replacement",
                   "fingerprint": fingerprint(sample.index)},
        "configurations": {name: r["params"] for name, r in results.items()},
        "search_run": run_search, "stability_run": run_stability,
        "predict_supported": False,
        "predict_note": "AgglomerativeClustering cannot assign unseen bookings; use the saved sample labels.",
        "test_split_used": False,
    }
    _save(Path(tables_dir), Path(figures_dir), Path(artifact_dir), settings, metadata, results, grid,
          preprocessor, sample, X, labels, features, shown)
    for name, r in results.items():
        print(f"{name}: {r['params']} silhouette={r['silhouette']} davies_bouldin={r['davies_bouldin']} "
              f"sizes={r['sizes']}", flush=True)
    return results


def _write_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
                    + "\n", encoding="utf-8")


def _save(tables, figures, artifacts, settings, metadata, results, grid, preprocessor, sample, X, labels,
          features, shown):
    for directory in (tables, figures, artifacts):
        directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, artifacts / "agglomerative_preprocessor.joblib")
    sample[features].assign(**{f"{n}_cluster": l for n, l in labels.items()}).to_csv(
        artifacts / "agglomerative_sample_labels.csv", index_label="source_row")
    _write_json(metadata, tables / "agglomerative_metadata.json")
    _write_json(results, tables / "agglomerative_results.json")

    rows = [{"configuration": n, **r["params"], "silhouette_euclidean": r["silhouette"],
             "silhouette_fit_metric": r[f"silhouette_{r['params']['metric']}"], "davies_bouldin": r["davies_bouldin"],
             "min_cluster_fraction": r["min_cluster_fraction"], "sizes": json.dumps(r["sizes"]),
             "stability_ari_median": _median_ari(r)}
            for n, r in results.items()]
    pd.DataFrame(rows).to_csv(tables / "agglomerative_metrics.csv", index=False)
    for name, l in labels.items():
        profile(sample, l, features, settings["profile_columns"]).to_csv(tables / f"agglomerative_{name}_profiles.csv")
    if grid is not None:
        grid.to_csv(tables / "agglomerative_search.csv", index=False)
        _plot_search(grid, figures / "agglomerative_search_silhouette.png")
        _plot_comparison(pd.DataFrame(rows), figures / "agglomerative_metrics_comparison.png")
    if any("stability" in r for r in results.values()):
        pd.DataFrame([{"configuration": n, **p} for n, r in results.items() for p in r["stability"]["per_run"]]).to_csv(
            tables / "agglomerative_stability.csv", index=False)

    _plot_dendrogram(X, results[shown]["params"], settings["dendrogram_leaves"], figures / "agglomerative_dendrogram.png")
    _plot_pca(X, labels, results, metadata["sample"]["seed"], figures / "agglomerative_pca_2d.png")
    _plot_sizes(labels, figures / "agglomerative_cluster_sizes.png")
    print(f"Saved tables to {tables}, figures to {figures}, preprocessor and sample labels to {artifacts}", flush=True)


def _median_ari(result):
    summary = (result.get("stability") or {}).get("summary")
    return summary["median"] if summary else None


def _savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_dendrogram(X, params, leaves, path):
    # SciPy builds the same hierarchy that scikit-learn uses for these linkages and exposes merge heights.
    Z = scipy_linkage(X, method=params["linkage"], metric=SCIPY_METRIC[params["metric"]])
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(Z, truncate_mode="lastp", p=leaves, show_leaf_counts=True, ax=ax)
    k = params["n_clusters"]
    if k > 1:  # horizontal cut between the merges that leave k and k-1 clusters
        ax.axhline((Z[-k, 2] + Z[-k + 1, 2]) / 2, color="r", ls="--", lw=1, label=f"cut -> {k} clusters")
        ax.legend()
    ax.set_title(f"Dendrogram ({params['linkage']}, {params['metric']}): last {leaves} merges of {len(X):,} sampled "
                 "bookings; bracketed leaf labels = booking counts")
    ax.set_ylabel("Merge distance")
    _savefig(fig, path)


def _plot_pca(X, labels, results, seed, path):
    pca = PCA(n_components=2, random_state=seed).fit(X)
    Z, ev = pca.transform(X), pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, len(labels), figsize=(max(9, 6 * len(labels)), 5), squeeze=False, sharex=True,
                             sharey=True)
    for ax, (name, l) in zip(axes[0], labels.items()):
        points = ax.scatter(Z[:, 0], Z[:, 1], c=l, cmap="tab10", s=4, alpha=0.6)
        ax.legend(*points.legend_elements(), title="cluster", fontsize=7)
        ax.set_title(f"{name}: {results[name]['params']}", fontsize=9)
        ax.set_xlabel(f"PC1 ({ev[0]:.1%} variance)")
    axes[0][0].set_ylabel(f"PC2 ({ev[1]:.1%} variance)")
    fig.suptitle("Clusters on 2 principal components\n(visualisation only; clustering used all scaled features)",
                 fontsize=11)
    _savefig(fig, path)


def _plot_sizes(labels, path):
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 3.5), squeeze=False)
    for ax, (name, l) in zip(axes[0], labels.items()):
        values, counts = np.unique(l, return_counts=True)
        ax.bar(values.astype(str), counts, color="#3b6ea5")
        ax.set_title(f"{name} cluster sizes")
        ax.set_xlabel("cluster")
    axes[0][0].set_ylabel("bookings in sample")
    _savefig(fig, path)


def _plot_comparison(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, column, note in ((axes[0], "silhouette_euclidean", "higher is better"),
                             (axes[1], "davies_bouldin", "lower is better")):
        ax.bar(rows["configuration"], rows[column], color=["#999999", "#3b6ea5"][:len(rows)])
        ax.set_title(f"{column} ({note})")
        for i, v in enumerate(rows[column]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    _savefig(fig, path)


def _plot_search(grid, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (l, m), g in grid.groupby(["linkage", "metric"]):
        ax.plot(g["n_clusters"], g["silhouette"], marker="o", ms=3, label=f"{l}/{m}")
    ax.set(xlabel="n_clusters", ylabel="silhouette (Euclidean)", title="Silhouette across the search space")
    ax.legend(fontsize=7, ncol=2)
    _savefig(fig, path)

from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from time import perf_counter
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.utils.config import PROJECT_ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd

from src.data.load import load_raw, validate_schema


FEATURES = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
]


def prepare_features():
    raw = load_raw()
    validate_schema(raw)

    df = raw.copy()

    # Normalize padded strings and missing-value markers.
    text_columns = df.select_dtypes(
        include=["object", "string", "str"]
    ).columns

    for column in text_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace({"NULL": pd.NA, "": pd.NA})
        )

    # Reproduce the notebook's baseline row-selection rules.
    # These rules still need review and documentation.
    keep = (
        df["MarketSegment"].ne("Undefined").fillna(True)
        & df["DistributionChannel"].ne("Undefined").fillna(True)
        & df["Adults"].gt(0)
    )

    cleaned = df.loc[keep].copy()
    X = cleaned[FEATURES].copy()

    if X.isna().any().any():
        raise ValueError("Selected features contain missing values.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Selected features contain non-finite values.")

    if (X < 0).any().any():
        raise ValueError("Selected features contain negative values.")

    print("Raw rows:", len(raw))
    print("Retained rows:", len(cleaned))
    print("Excluded rows:", len(raw) - len(cleaned))
    print("Feature matrix:", X.shape)

    return X


def run_baseline(X):
    preprocessor = Pipeline([
        ("scaler", StandardScaler()),
    ])
    X_scaled = preprocessor.fit_transform(X)

    X_unique, inverse, counts = np.unique(
        X_scaled,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    model = DBSCAN(
        eps=0.5,
        min_samples=5,
        algorithm="ball_tree",
        n_jobs=1,
    )

    start = perf_counter()
    model.fit(X_unique, sample_weight=counts)
    fit_seconds = perf_counter() - start

    labels = model.labels_[inverse]
    n_clusters = len(set(labels) - {-1})
    noise_count = int((labels == -1).sum())

    print("Unique feature rows:", len(X_unique))
    print("Clusters:", n_clusters)
    print("Noise points:", noise_count)
    print(f"Noise percentage: {noise_count / len(labels):.2%}")
    print(f"Fit time: {fit_seconds:.2f} seconds")

    return X_scaled, labels, preprocessor, model

def evaluate_baseline(X_scaled, labels, seed=42):
    mask = labels != -1
    X_eval = X_scaled[mask]
    labels_eval = labels[mask]

    results = {
        "silhouette_sample": np.nan,
        "silhouette_sample_size": 0,
        "silhouette_clusters_represented": 0,
        "davies_bouldin": np.nan,
    }

    n_clusters = len(np.unique(labels_eval))

    if not 2 <= n_clusters < len(labels_eval):
        print("Evaluation metrics undefined for this clustering.")
        return results

    rng = np.random.default_rng(seed)
    idx = rng.choice(
        len(X_eval),
        size=min(5000, len(X_eval)),
        replace=False,
    )

    sampled_labels = labels_eval[idx]
    sample_clusters = len(np.unique(sampled_labels))

    results["silhouette_sample_size"] = len(idx)
    results["silhouette_clusters_represented"] = sample_clusters

    if 2 <= sample_clusters < len(idx):
        results["silhouette_sample"] = silhouette_score(
            X_eval[idx], sampled_labels
        )

    results["davies_bouldin"] = davies_bouldin_score(
        X_eval, labels_eval
    )

    print(f"Silhouette: {results['silhouette_sample']:.4f}")
    print("Clusters represented in sample:", sample_clusters)
    print(f"Davies-Bouldin: {results['davies_bouldin']:.4f}")

    return results

def save_results(X, labels, model, metrics):
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": "DBSCAN",
        "configuration": "baseline",
        "eps": model.eps,
        "min_samples": model.min_samples,
        "rows": len(X),
        "features": ", ".join(X.columns),
        "clusters": len(set(labels) - {-1}),
        "noise_points": int((labels == -1).sum()),
        "noise_percentage": 100 * float((labels == -1).mean()),
        "evaluation_excludes_noise": True,
        **metrics,
    }

    pd.DataFrame([results]).to_csv(
        tables_dir / "dbscan_baseline_script_metrics.csv",
        index=False,
    )

    profiles = X.copy()
    profiles["Cluster"] = labels

    grouped = profiles.groupby("Cluster")
    summary = grouped.size().rename("Bookings").to_frame()
    summary = summary.join(grouped[list(X.columns)].mean())
    summary["Percentage"] = 100 * summary["Bookings"] / len(X)

    summary.to_csv(
        tables_dir / "dbscan_baseline_script_profiles.csv"
    )

    print("Script metrics and cluster profiles saved.")
def save_pca_plot(X_scaled, labels, seed=42):
    pca = PCA(n_components=2, svd_solver="full")
    X_2d = pca.fit_transform(X_scaled)

    rng = np.random.default_rng(seed)
    idx = rng.choice(
        len(X_2d),
        size=min(10000, len(X_2d)),
        replace=False,
    )

    points = X_2d[idx]
    plot_labels = labels[idx]
    noise = plot_labels == -1

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        points[~noise, 0],
        points[~noise, 1],
        c=plot_labels[~noise],
        cmap="turbo",
        s=8,
        alpha=0.5,
    )

    ax.scatter(
        points[noise, 0],
        points[noise, 1],
        color="black",
        marker="x",
        s=20,
        label="Noise",
    )

    explained = pca.explained_variance_ratio_.sum()

    ax.set(
        title=f"DBSCAN baseline — PCA projection "
              f"({explained:.2%} explained variance)",
        xlabel="Principal Component 1",
        ylabel="Principal Component 2",
    )
    ax.legend()
    fig.tight_layout()

    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    output_path = figures_dir / "dbscan_baseline_script_pca.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"PCA explained variance: {explained:.2%}")
    print("Script PCA figure saved.")

    return explained

if __name__ == "__main__":
    X = prepare_features()
    X_scaled, labels, preprocessor, model = run_baseline(X)
    metrics = evaluate_baseline(X_scaled, labels)

    metrics["pca_2d_explained_variance"] = save_pca_plot(
        X_scaled, labels
    )

    save_results(X, labels, model, metrics)
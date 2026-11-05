"""Internal (label-free) clustering metrics that stay defined for degenerate labelings."""

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score


def cluster_metrics(X, labels, fit_metric="euclidean"):
    """Cluster sizes, silhouette (Euclidean, and under the model's own distance) and Davies-Bouldin.

    Silhouette (higher is better) and Davies-Bouldin (lower is better; Euclidean by definition) need
    2 <= distinct labels <= n_samples - 1. Outside that range they are None with a note, not an error.
    """
    X, labels = np.asarray(X), np.asarray(labels)
    if X.ndim != 2 or len(X) != len(labels):
        raise ValueError(f"X has shape {X.shape} but {len(labels)} labels were given")
    unique, counts = np.unique(labels, return_counts=True)
    result = {"n_clusters_found": int(len(unique)),
              "sizes": {int(u): int(c) for u, c in zip(unique, counts)},
              "min_cluster_fraction": float(counts.min() / len(labels)) if len(labels) else 0.0,
              "silhouette": None, f"silhouette_{fit_metric}": None, "davies_bouldin": None, "note": None}
    if not 2 <= len(unique) <= len(labels) - 1:
        result["note"] = f"metrics undefined for {len(unique)} distinct label(s) on {len(labels)} samples"
        return result
    result["silhouette"] = float(silhouette_score(X, labels, metric="euclidean"))
    result[f"silhouette_{fit_metric}"] = float(silhouette_score(X, labels, metric=fit_metric))
    result["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    return result

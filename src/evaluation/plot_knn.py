"""Plot saved baseline results without fitting or predicting again."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.load import load_raw
from src.utils.config import PROJECT_ROOT, load_config


def main():
    config = load_config()
    paths = {key: PROJECT_ROOT / value for key, value in config["paths"].items()}
    run = json.loads((paths["tables_dir"] / "knn_baseline_run.json").read_text())
    if any(run[key] != config[key] for key in ("regression", "split")) or run["seed"] != config["random_seed"]:
        raise ValueError("Settings differ from saved run; regenerate baseline first.")
    split = pd.read_csv(paths["processed_dir"] / "knn_split_assignments.csv")
    predictions = pd.read_csv(paths["processed_dir"] / "knn_validation_predictions.csv")
    validation_rows = split.loc[split["split"].eq("validation"), "source_row"]
    if (predictions.source_row.duplicated().any()
            or set(predictions.source_row) != set(validation_rows)):
        raise ValueError("Saved predictions do not match validation assignments.")
    training_rows = split.loc[split["split"].eq("train"), "source_row"]
    train_adr = load_raw(paths["raw_data"]).iloc[training_rows]["ADR"]
    actual = predictions.actual_ADR.to_numpy()
    predicted = predictions.predicted_ADR.to_numpy()
    error = predicted - actual
    stats = {
        "validation_rows": len(actual),
        "mean_error_predicted_minus_actual": float(error.mean()),
        "median_absolute_error": float(np.median(np.abs(error))),
        "p95_absolute_error": float(np.quantile(np.abs(error), .95)),
        "max_absolute_error": float(np.abs(error).max()),
        "training_ADR_99th_percentile": float(train_adr.quantile(.99)),
    }
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "figure.facecolor": "white",
    })
    paths["figures_dir"].mkdir(parents=True, exist_ok=True)
    # Separate training-only figure; the zoom does not filter the model's data.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    cutoff = stats["training_ADR_99th_percentile"]
    for ax, values, title in [
        (axes[0], train_adr, "Full range | logarithmic count axis"),
        (axes[1], train_adr[train_adr <= cutoff], f"Detail view | ADR <= {cutoff:g}"),
    ]:
        ax.hist(values, bins=60, color="#247a91", edgecolor="white", linewidth=.3)
        ax.set(xlabel="ADR (rate units)", ylabel="Training bookings", title=title)
    axes[0].set_yscale("log")
    axes[1].text(.97, .94, f"{int((train_adr > cutoff).sum()):,} higher-rate rows\noutside this view",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=9)
    fig.suptitle(f"Training ADR distribution | {len(train_adr):,} bookings", fontsize=15)
    fig.savefig(paths["figures_dir"] / "knn_training_adr.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    ax = axes[0, 0]
    points = ax.hexbin(actual, predicted, gridsize=55, bins="log", mincnt=1, cmap="Blues")
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    ax.plot([low, high], [low, high], "--", color="#bd4a36", label="Perfect prediction")
    ax.set(xlabel="Actual ADR", ylabel="Predicted ADR", title="Actual versus predicted | full range")
    ax.legend(fontsize=8)
    fig.colorbar(points, ax=ax, label="Bookings per hexagon")
    ax = axes[0, 1]
    ax.scatter(predicted, error, s=7, alpha=.22, color="#247a91", rasterized=True)
    ax.axhline(0, color="#bd4a36", linestyle="--")
    ax.set(xlabel="Predicted ADR", ylabel="Predicted - actual ADR", title="Errors versus predictions | full range")
    ax = axes[1, 0]
    ax.hist(error, bins=70, color="#247a91", edgecolor="white", linewidth=.3)
    ax.axvline(0, color="#bd4a36", linestyle="--")
    ax.set_yscale("log")
    ax.set(xlabel="Predicted - actual ADR", ylabel="Validation bookings (log scale)",
           title="Error distribution | full range")
    ax = axes[1, 1]
    results = pd.DataFrame(run["validation_results"])
    positions = np.arange(2)
    for offset, metric, color in [(-.18, "MAE", "#247a91"), (.18, "RMSE", "#dc9a39")]:
        bars = ax.bar(positions + offset, results[metric], width=.34, label=metric, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks(positions, ["Training-mean reference", "KNN (k=5)"])
    ax.set(ylabel="ADR error units (lower is better)", title="Validation error comparison")
    ax.set_ylim(0, results[["MAE", "RMSE"]].to_numpy().max() * 1.25)
    ax.legend()
    knn_r2 = results.loc[results.model.eq("KNN baseline"), "R2"].iloc[0]
    fig.suptitle(f"KNN baseline | {len(actual):,} validation bookings | R2 = {knn_r2:.3f}", fontsize=15)
    fig.supxlabel("Saved validation predictions only. Positive error = overprediction. Final test set reserved.", fontsize=10)
    fig.savefig(paths["figures_dir"] / "knn_validation_diagnostics.png", dpi=180)
    plt.close(fig)
    (paths["tables_dir"] / "knn_error_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print("Saved two plots to outputs/figures. No model retrained; no test evaluation.")


if __name__ == "__main__":
    main()

"""Training-only audit: python -m src.data.audit_regression.

Flags are for investigation, never automatic deletion or target correction.
"""

import pandas as pd

from src.data.load import load_raw, validate_schema
from src.data.split import split_predictor_groups
from src.preprocessing.regression import select_features
from src.utils.config import PROJECT_ROOT, load_config


def main():
    config = load_config()
    df = load_raw(PROJECT_ROOT / config["paths"]["raw_data"])
    validate_schema(df)
    X = select_features(df, config["regression"])
    train_idx, _, _ = split_predictor_groups(X, config["split"], config["random_seed"])
    training = df.iloc[train_idx].copy()
    guests = training[["Adults", "Children", "Babies"]].sum(axis=1, min_count=3)
    nights = training["StaysInWeekNights"] + training["StaysInWeekendNights"]
    # Fixed exploratory cutoffs, not claims of invalidity or cleaning thresholds.
    flags = pd.DataFrame({
        "zero_guests": guests.eq(0),
        "zero_nights": nights.eq(0),
        "zero_ADR": training["ADR"].eq(0),
        "ADR_over_500_review_only": training["ADR"].gt(500),
        "babies_over_3_review_only": training["Babies"].gt(3),
    }, index=training.index)
    summary = []
    examples = []
    for name in flags:
        subset = training.loc[flags[name]]
        summary.append({
            "flag": name, "training_rows": len(subset),
            "percent_training": 100 * len(subset) / len(training),
            "ADR_min": subset["ADR"].min(),
            "ADR_median": subset["ADR"].median(),
            "ADR_max": subset["ADR"].max(),
            "decision": "retain; no verified correction",
        })
        columns = config["regression"]["features"] + [config["regression"]["target"]]
        sample = subset[columns].head(12).copy()
        sample.insert(0, "source_row", sample.index)
        sample.insert(0, "flag", name)
        examples.append(sample)

    tables = PROJECT_ROOT / config["paths"]["tables_dir"]
    processed = PROJECT_ROOT / config["paths"]["processed_dir"]
    tables.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(summary)
    table.to_csv(tables / "regression_training_audit.csv", index=False)
    X.iloc[train_idx].isna().sum().rename("missing_training_rows").to_csv(
        tables / "regression_training_missing.csv", index_label="feature"
    )
    pd.concat(examples, ignore_index=True).to_csv(
        processed / "regression_training_flagged_examples.csv", index=False
    )
    print(f"Audited {len(training):,} training rows only.")
    print(table.to_string(index=False))
    print("Zero-guests AND zero-nights:", int((flags.zero_guests & flags.zero_nights).sum()))
    print("All zero-night rows have ADR zero:", bool(training.loc[flags.zero_nights, "ADR"].eq(0).all()))
    print("No rows deleted, no values corrected, no model retrained.")
    print("Saved outputs/tables/regression_training_audit.csv")


if __name__ == "__main__":
    main()

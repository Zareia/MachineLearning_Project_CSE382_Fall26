"""Run the Milestone 1 XGBoost baselines."""

import argparse
from pathlib import Path

from src.utils.config import PROJECT_ROOT, load_config


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--task", choices=["regression", "classification"], required=True)
    train.add_argument("--model", choices=["xgboost"], required=True)
    train.add_argument("--data", type=Path, default=PROJECT_ROOT / config["paths"]["raw_data"])
    train.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs")
    train.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / config["paths"]["artifacts_dir"])
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}")
    if args.task == "regression":
        from src.regression.xgb_regression import build_pipeline
    else:
        from src.classification.xgb_classification import build_pipeline
    from src.evaluation.xgb_baseline import train_baseline
    train_baseline(args.task, build_pipeline(), args.data, args.outputs, args.artifacts)


if __name__ == "__main__":
    main()

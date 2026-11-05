"""Run the Milestone 1 baselines (XGBoost, Decision Tree, Agglomerative clustering)."""

import argparse
from pathlib import Path

from src.utils.config import PROJECT_ROOT, load_config


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--task", choices=["regression", "classification"], required=True)
    train.add_argument("--model", choices=["xgboost", "decision_tree"], required=True)
    train.add_argument("--data", type=Path, default=PROJECT_ROOT / config["paths"]["raw_data"])
    train.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs")
    train.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / config["paths"]["artifacts_dir"])
    cluster = commands.add_parser("cluster")
    cluster.add_argument("--algorithm", choices=["agglomerative"], required=True)
    cluster.add_argument("--search", action="store_true", help="optional parameter search (not the baseline)")
    cluster.add_argument("--stability", action="store_true", help="optional resampling stability check")
    cluster.add_argument("--data", type=Path, default=PROJECT_ROOT / config["paths"]["raw_data"])
    cluster.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs")
    cluster.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / config["paths"]["artifacts_dir"])
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}")
    if args.command == "cluster":
        from src.clustering.agglomerative import run
        run(args.data, args.outputs / "tables", args.outputs / "figures", args.artifacts,
            run_search=args.search, run_stability=args.stability)
        return
    if args.model == "decision_tree":
        if args.task != "classification":
            parser.error("--model decision_tree is only available for --task classification")
        from src.classification.decision_tree import train as train_decision_tree
        train_decision_tree(args.data, args.outputs, args.artifacts)
        return
    if args.task == "regression":
        from src.regression.xgb_regression import build_pipeline
    else:
        from src.classification.xgb_classification import build_pipeline
    from src.evaluation.xgb_baseline import train_baseline
    train_baseline(args.task, build_pipeline(), args.data, args.outputs, args.artifacts)


if __name__ == "__main__":
    main()

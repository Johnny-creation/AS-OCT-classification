"""Command-line interface for training leak-free stacking meta-learners."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.stacking.pipeline import save_model, train_meta_model
from src.utils.io import load_prediction_tables

LOGGER = logging.getLogger("train_stacking")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for stacking training."""

    parser = argparse.ArgumentParser(description="Train stacking meta learner")
    parser.add_argument("--ens-file", required=True, help="CSV/Parquet with ensemble features")
    parser.add_argument("--labels-col", required=True, help="Label column name")
    parser.add_argument("--feature-type", choices=["proba", "logit"], default="logit", help="Feature type")
    parser.add_argument("--meta", choices=["lr", "svm", "rf", "knn", "xgb", "gnb"], default="lr", help="Meta learner type")
    parser.add_argument("--outdir", required=True, help="Directory to store trained model")
    parser.add_argument("--prefixes", help="Comma-separated prefixes to include (optional)")
    return parser.parse_args()


def main() -> None:
    """Entry-point for training stacking meta-learners."""

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_prediction_tables([args.ens_file])
    df = data.frame
    prefixes = None
    if args.prefixes:
        prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]

    model, feature_cols = train_meta_model(
        df=df,
        labels_col=args.labels_col,
        feature_type=args.feature_type,
        meta=args.meta,
        prefixes=prefixes,
    )
    model_path = outdir / f"stacking_{args.meta}.pkl"
    save_model(model, feature_cols, args.feature_type, model_path)
    LOGGER.info("Saved meta model to %s", model_path)


if __name__ == "__main__":
    main()

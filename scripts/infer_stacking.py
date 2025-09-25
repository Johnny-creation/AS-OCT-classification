"""Command-line interface for running stacking inference."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.stacking.pipeline import load_model, predict_meta_model
from src.utils.io import load_prediction_tables, save_dataframe

LOGGER = logging.getLogger("infer_stacking")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for stacking inference."""

    parser = argparse.ArgumentParser(description="Run stacking inference")
    parser.add_argument("--stacker", required=True, help="Path to trained stacking model (joblib)")
    parser.add_argument("--inputs", nargs="+", required=True, help="Prediction file(s) to combine")
    parser.add_argument(
        "--feature-type",
        choices=["proba", "logit"],
        required=True,
        help="Feature representation expected by the stacker",
    )
    parser.add_argument("--out", required=True, help="Output path for fused probabilities")
    return parser.parse_args()


def main() -> None:
    """Entry-point for stacking inference."""

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    model, feature_cols, stored_feature_type = load_model(Path(args.stacker))
    if stored_feature_type and stored_feature_type != args.feature_type:
        raise ValueError(
            f"Stacker expects feature type '{stored_feature_type}' but '--feature-type' was set to '{args.feature_type}'"
        )
    data = load_prediction_tables(args.inputs)
    df = data.frame
    probas = predict_meta_model(model, df, feature_cols)

    base_cols: List[str] = []
    for col in ["patient_id", "image_id", "y_true"]:
        if col in df.columns:
            base_cols.append(col)
    out_df = df[base_cols].copy() if base_cols else pd.DataFrame(index=df.index)
    for i in range(probas.shape[1]):
        out_df[f"proba_{i}"] = probas[:, i]
    save_dataframe(out_df, Path(args.out))
    LOGGER.info("Saved stacked predictions to %s", args.out)


if __name__ == "__main__":
    main()

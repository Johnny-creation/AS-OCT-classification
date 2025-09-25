"""Command-line interface for fitting and applying temperature scaling."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.metrics.calibration import TemperatureScaler
from src.utils.io import (
    infer_logit_prefixes,
    infer_prefixes,
    load_prediction_tables,
    logit_column,
    probability_column,
    save_dataframe,
    save_json,
)

LOGGER = logging.getLogger("fit_temperature")
NUM_CLASSES = 4


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Temperature scaling for model predictions")
    parser.add_argument("--ens-file", required=True, help="Predictions on ensemble dataset (with logits)")
    parser.add_argument("--apply-to", nargs="+", required=True, help="Prediction file(s) to calibrate")
    parser.add_argument("--by", choices=["per_model", "per_ensemble"], default="per_model", help="Temperature strategy")
    parser.add_argument("--outdir", required=True, help="Directory for calibrated outputs")
    parser.add_argument("--proba-prefixes", help="Optional prefixes to limit calibration")
    return parser.parse_args()


def get_scalers(df: pd.DataFrame, prefixes: List[str]) -> Dict[str, TemperatureScaler]:
    """Fit temperature scalers for the requested prefixes."""

    scalers: Dict[str, TemperatureScaler] = {}
    for prefix in prefixes:
        cols = [logit_column(prefix, i) for i in range(NUM_CLASSES)]
        if not all(col in df.columns for col in cols):
            raise KeyError(f"Missing logits for prefix '{prefix}' in ensemble file")
        logits = df[cols].to_numpy()
        labels_col = "y_true"
        if labels_col not in df.columns:
            raise KeyError("Ensemble file must contain y_true column for fitting temperature")
        labels = df[labels_col].to_numpy(dtype=int)
        scaler = TemperatureScaler()
        scaler.fit(logits, labels)
        scalers[prefix] = scaler
    return scalers


def apply_scalers(df: pd.DataFrame, scalers: Dict[str, TemperatureScaler]) -> pd.DataFrame:
    """Apply fitted scalers to logits/probabilities in the provided dataframe."""

    calibrated = df.copy()
    for prefix, scaler in scalers.items():
        logit_cols = [logit_column(prefix, i) for i in range(NUM_CLASSES)]
        if all(col in calibrated.columns for col in logit_cols):
            logits = calibrated[logit_cols].to_numpy()
        else:
            proba_cols = [probability_column(prefix, i) for i in range(NUM_CLASSES)]
            if not all(col in calibrated.columns for col in proba_cols):
                raise KeyError(
                    f"Cannot find logits or probabilities for prefix '{prefix}' in apply file"
                )
            LOGGER.info(
                "Prefix '%s' lacks logits; deriving pseudo-logits from probabilities", prefix or "model"
            )
            probas = calibrated[proba_cols].to_numpy()
            logits = np.log(np.clip(probas, 1e-6, 1.0))
        probas = scaler.transform(logits)
        for i in range(NUM_CLASSES):
            calibrated[f"{prefix}_proba_{i}"] = probas[:, i]
    return calibrated


def main() -> None:
    """Script entry-point."""

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ens_df = load_prediction_tables([args.ens_file]).frame
    if args.proba_prefixes:
        prefixes = [p.strip() for p in args.proba_prefixes.split(",") if p.strip()]
    else:
        prefixes = infer_logit_prefixes(ens_df)
        if not prefixes:
            prefixes = infer_prefixes(ens_df)
    if not prefixes:
        raise ValueError("Unable to infer prefixes for temperature scaling")

    if args.by == "per_model":
        scalers = get_scalers(ens_df, prefixes)
    else:
        if len(prefixes) != 1:
            raise ValueError(
                "Per-ensemble temperature scaling expects exactly one prefix. Use --proba-prefixes to select the target predictions."
            )
        prefix = prefixes[0]
        LOGGER.info("Using prefix '%s' for per_ensemble temperature", prefix or "model")
        scalers = {prefix: get_scalers(ens_df, [prefix])[prefix]}

    temps = {prefix: scaler.temperature for prefix, scaler in scalers.items()}
    save_json(temps, outdir / "temperatures.json")

    inputs = load_prediction_tables(args.apply_to)
    calibrated = apply_scalers(inputs.frame, scalers)
    save_dataframe(calibrated, outdir / "calibrated_predictions.csv")
    LOGGER.info("Saved calibrated predictions to %s", outdir / "calibrated_predictions.csv")


if __name__ == "__main__":
    main()

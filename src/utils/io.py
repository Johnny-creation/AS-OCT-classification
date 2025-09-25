"""Utility functions for loading and validating prediction tables."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".pq"}


def probability_column(prefix: str, idx: int) -> str:
    """Return the canonical probability column name for ``prefix`` and class ``idx``."""

    return f"{prefix}_proba_{idx}" if prefix else f"proba_{idx}"


def logit_column(prefix: str, idx: int) -> str:
    """Return the canonical logit column name for ``prefix`` and class ``idx``."""

    return f"{prefix}_logit_{idx}" if prefix else f"logit_{idx}"


@dataclass
class PredictionData:
    """Container for aligned prediction tables.

    Attributes:
        frame: DataFrame containing aligned data for all inputs.
        source_files: Ordered list of files that were loaded.
    """

    frame: pd.DataFrame
    source_files: List[Path]


def _discover_files(path_like: str) -> List[Path]:
    path = Path(path_like)
    if path.is_dir():
        files: List[Path] = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(sorted(path.glob(f"*{ext}")))
        if not files:
            raise FileNotFoundError(f"No supported prediction files found in directory: {path}")
        return files
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension for {path}. Supported: {SUPPORTED_EXTENSIONS}")
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


def load_prediction_tables(inputs: Sequence[str]) -> PredictionData:
    """Load CSV/Parquet prediction files and align them row-wise.

    Args:
        inputs: Iterable of file paths or directories.

    Returns:
        PredictionData with concatenated DataFrame.
    """

    files: List[Path] = []
    for item in inputs:
        files.extend(_discover_files(item))
    if not files:
        raise FileNotFoundError("No prediction files discovered from inputs")

    frames: List[pd.DataFrame] = []
    key_cols: Optional[List[str]] = None
    for file in files:
        if file.suffix.lower() == ".csv":
            df = pd.read_csv(file)
        else:
            df = pd.read_parquet(file)
        df["__source_file"] = str(file)
        frames.append(df)
        LOGGER.info("Loaded %s with %d rows and columns: %s", file, len(df), list(df.columns))
        if key_cols is None:
            key_cols = _infer_key_columns(df)

    aligned = _align_frames(frames, key_cols)
    return PredictionData(frame=aligned, source_files=files)


def _infer_key_columns(df: pd.DataFrame) -> List[str]:
    candidates = ["patient_id", "image_id", "study_id", "filename"]
    keys = []
    for col in candidates:
        if col in df.columns:
            keys.append(col)
    if not keys:
        # Fall back to row index alignment later.
        keys = []
    else:
        LOGGER.debug("Using key columns for alignment: %s", keys)
    return keys


def _align_frames(frames: Sequence[pd.DataFrame], key_cols: Optional[Sequence[str]]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No frames provided for alignment")

    if not key_cols:
        LOGGER.warning("No key columns found; aligning frames by row order")
        base = frames[0].copy()
        base["__row_id"] = np.arange(len(base))
        merge_keys = ["__row_id"]
    else:
        merge_keys = list(dict.fromkeys(key_cols))
        base = frames[0].copy()

    for other in frames[1:]:
        if not merge_keys:
            other = other.copy()
            other["__row_id"] = np.arange(len(other))
        missing_keys = [col for col in merge_keys if col not in other.columns]
        if missing_keys:
            raise KeyError(f"Missing key columns {missing_keys} in file {other['__source_file'].iat[0]}")
        base = base.merge(other, on=merge_keys, how="inner", suffixes=("", "_dup"))
        dup_cols = [col for col in base.columns if col.endswith("_dup")]
        if dup_cols:
            LOGGER.warning("Dropping duplicated columns during merge: %s", dup_cols)
            base = base.drop(columns=dup_cols)
    if "__row_id" in base.columns:
        base = base.drop(columns=["__row_id"])
    return base


def ensure_probability_columns(
    df: pd.DataFrame,
    proba_prefixes: Sequence[str],
    num_classes: int,
) -> Dict[str, List[str]]:
    """Validate probability columns and return mapping of prefix to sorted column list."""

    mapping: Dict[str, List[str]] = {}
    for prefix in proba_prefixes:
        cols = [probability_column(prefix, i) for i in range(num_classes)]
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise KeyError(f"Missing probability columns for prefix '{prefix or 'default'}': {missing}")
        mapping[prefix] = cols
    return mapping


def ensure_logit_columns(
    df: pd.DataFrame,
    prefixes: Sequence[str],
    num_classes: int,
) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for prefix in prefixes:
        cols = [logit_column(prefix, i) for i in range(num_classes)]
        if all(col in df.columns for col in cols):
            mapping[prefix] = cols
    return mapping


def infer_logit_prefixes(df: pd.DataFrame) -> List[str]:
    prefixes = set()
    for col in df.columns:
        if "_logit_" in col:
            prefixes.add(col.split("_logit_")[0])
        elif col.startswith("logit_"):
            prefixes.add("")
    return sorted(prefixes)


def normalise_probabilities(probas: np.ndarray) -> np.ndarray:
    """Renormalise probability rows, guarding against degenerate inputs."""

    row_sums = probas.sum(axis=1, keepdims=True)
    zero_rows = np.isclose(row_sums, 0.0)
    if zero_rows.any():
        LOGGER.warning("Found %d rows with zero probability mass; replacing with uniform distribution", zero_rows.sum())
        probas[zero_rows] = 1.0 / probas.shape[1]
        row_sums = probas.sum(axis=1, keepdims=True)
    probas = probas / row_sums
    return probas


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format for {path}")


def save_json(data: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def infer_prefixes(df: pd.DataFrame) -> List[str]:
    prefixes = set()
    for col in df.columns:
        if "_proba_" in col:
            prefixes.add(col.split("_proba_")[0])
        elif col.startswith("proba_"):
            prefixes.add("")
    return sorted(prefixes)

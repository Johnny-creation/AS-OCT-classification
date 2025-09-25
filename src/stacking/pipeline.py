"""Stacking ensemble utilities without data leakage."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None

LOGGER = logging.getLogger(__name__)

META_MODELS = {"lr", "svm", "rf", "knn", "xgb", "gnb"}


def detect_feature_columns(df: pd.DataFrame, feature_type: str) -> List[str]:
    """Detect all columns of the requested feature type."""

    marker = f"{feature_type}_"
    cols = [col for col in df.columns if marker in col]
    if not cols:
        raise KeyError(f"No columns matching feature type '{feature_type}' were found")
    return sorted(cols)


def extract_features(
    df: pd.DataFrame,
    feature_type: str,
    prefixes: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Extract stacked features for the requested prefixes and feature type."""

    if prefixes:
        cols = []
        for prefix in prefixes:
            if prefix:
                pattern = f"{prefix}_{feature_type}_"
                cols.extend(sorted([col for col in df.columns if col.startswith(pattern)]))
            else:
                cols.extend(sorted([col for col in df.columns if col.startswith(f"{feature_type}_")]))
        if not cols:
            raise KeyError(f"No columns found for prefixes {prefixes} and feature_type {feature_type}")
    else:
        cols = detect_feature_columns(df, feature_type)
    features = df[cols].to_numpy(dtype=float)
    return features, cols


def build_meta_model(name: str, random_state: int = 42):
    """Construct the meta-learner specified by ``name``."""

    name = name.lower()
    if name not in META_MODELS:
        raise ValueError(f"Unsupported meta model '{name}'. Options: {sorted(META_MODELS)}")
    if name == "lr":
        return LogisticRegression(max_iter=1000, multi_class="auto", random_state=random_state)
    if name == "svm":
        base = LinearSVC(max_iter=5000, random_state=random_state)
        return CalibratedClassifierCV(base, cv=3)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=random_state)
    if name == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ])
    if name == "gnb":
        return GaussianNB()
    if name == "xgb":
        if XGBClassifier is None:
            raise RuntimeError("XGBoost is not available. Install xgboost to enable this meta model.")
        return XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        )
    raise AssertionError("Unhandled meta model")


def train_meta_model(
    df: pd.DataFrame,
    labels_col: str,
    feature_type: str,
    meta: str,
    prefixes: Optional[Sequence[str]] = None,
    random_state: int = 42,
) -> Tuple[object, List[str]]:
    """Train a leak-free stacking meta model on D_ens predictions."""

    if labels_col not in df.columns:
        raise KeyError(f"Label column '{labels_col}' not found")
    y = df[labels_col].to_numpy(dtype=int)
    X, feature_cols = extract_features(df, feature_type, prefixes)
    model = build_meta_model(meta, random_state=random_state)
    model.fit(X, y)
    LOGGER.info("Trained meta model %s on %d samples", meta, len(df))
    return model, feature_cols


def predict_meta_model(model, df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    """Generate probabilities using a fitted meta model."""

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns during inference: {missing}")
    X = df[list(feature_cols)].to_numpy(dtype=float)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
    else:
        logits = model.decision_function(X)
        if logits.ndim == 1:
            logits = np.stack([-logits, logits], axis=1)
        probas = softmax(logits)
    return probas


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax over the final axis."""

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def save_model(model, feature_cols: Sequence[str], feature_type: str, path: Path) -> None:
    """Persist meta model and feature column metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_cols": list(feature_cols), "feature_type": feature_type},
        path,
    )


def load_model(path: Path):
    """Load a previously persisted meta model."""

    data = joblib.load(path)
    return data["model"], data["feature_cols"], data.get("feature_type")


def weighted_average(probas_list: Sequence[np.ndarray], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Combine probabilities via weighted averaging with normalised weights."""

    if not probas_list:
        raise ValueError("No probability arrays provided")
    num_models = len(probas_list)
    num_classes = probas_list[0].shape[1]
    for arr in probas_list:
        if arr.shape[1] != num_classes:
            raise ValueError("All probability arrays must have the same number of classes")
    if weights is None:
        weights = [1.0] * num_models
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    stacked = np.stack(probas_list, axis=0)
    combined = np.tensordot(weights, stacked, axes=(0, 0))
    return combined


def hard_vote(probas_list: Sequence[np.ndarray]) -> np.ndarray:
    """Perform hard voting with deterministic tie-breaking."""

    votes = [np.argmax(p, axis=1) for p in probas_list]
    votes = np.stack(votes, axis=0)
    num_samples = votes.shape[1]
    num_classes = probas_list[0].shape[1]
    result = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        counts = np.bincount(votes[:, i], minlength=num_classes)
        winners = np.where(counts == counts.max())[0]
        if len(winners) == 1:
            chosen = winners[0]
        else:
            # Tie-breaker: choose class with highest average probability
            class_scores = np.mean([probas_list[j][i, winners] for j in range(len(probas_list))], axis=0)
            chosen = winners[int(np.argmax(class_scores))]
        result[i, chosen] = 1.0
    return result

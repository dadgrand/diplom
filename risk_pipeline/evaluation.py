from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

CLASSES = ["low", "medium", "high"]
CLASS_ORDER = {"low": 0, "medium": 1, "high": 2}


def classification_metrics(y_true, y_pred, labels: list[str] | None = None) -> dict[str, float]:
    labels = labels or CLASSES
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    high_mask = y_true == "high"
    high_fn = ((y_true == "high") & (y_pred != "high")).sum()
    adjacent_or_exact = np.mean([abs(CLASS_ORDER.get(t, 99) - CLASS_ORDER.get(p, 99)) <= 1 for t, p in zip(y_true, y_pred)]) if len(y_true) else np.nan
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "high_recall": float(recall_score(y_true, y_pred, labels=["high"], average="macro", zero_division=0)),
        "high_precision": float(precision_score(y_true, y_pred, labels=["high"], average="macro", zero_division=0)),
        "high_false_negative_rate": float(high_fn / max(high_mask.sum(), 1)),
        "ordinal_adjacent_accuracy": float(adjacent_or_exact),
    }


def probability_diagnostics(y_true, proba: np.ndarray, labels: list[str] | None = None, n_bins: int = 10) -> dict[str, float]:
    labels = labels or CLASSES
    y_true = np.asarray(y_true).astype(str)
    proba = np.asarray(proba, dtype=float)
    pred_idx = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    pred = np.array([labels[i] for i in pred_idx])
    accuracy = (pred == y_true).astype(float)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(accuracy[mask].mean()) - float(confidence[mask].mean()))
    high_idx = labels.index("high") if "high" in labels else len(labels) - 1
    return {
        "mean_confidence": float(np.nanmean(confidence)) if len(confidence) else np.nan,
        "ece_10_bins": float(ece),
        "mean_high_probability": float(np.nanmean(proba[:, high_idx])) if len(proba) else np.nan,
    }


def confusion_as_frame(y_true, y_pred, labels: list[str] | None = None) -> pd.DataFrame:
    labels = labels or CLASSES
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=[f"actual_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])


def economic_monotonicity(
    df: pd.DataFrame,
    pred_col: str = "predicted_class",
    components: list[str] | None = None,
) -> pd.DataFrame:
    components = components or ["future_max_drawdown", "future_downside_volatility", "future_cvar_95", "future_illiquidity"]
    tmp = df.copy()
    grouped = tmp.groupby(pred_col)[components].mean().reindex(CLASSES)
    monotone = {}
    for col in components:
        values = grouped[col].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        monotone[col + "_is_monotone"] = bool(len(finite) <= 1 or np.all(np.diff(finite) >= -1e-12))
    grouped["class_order"] = [0, 1, 2]
    for key, value in monotone.items():
        grouped[key] = value
    return grouped.reset_index()


def classwise_metrics_frame(y_true, y_pred, labels: list[str] | None = None) -> pd.DataFrame:
    labels = labels or CLASSES
    rows = []
    for cls in labels:
        rows.append(
            {
                "class": cls,
                "precision": float(precision_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0)),
                "support": int((np.asarray(y_true) == cls).sum()),
            }
        )
    return pd.DataFrame(rows)

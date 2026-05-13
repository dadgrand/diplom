from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .schemas import SchemaError, require_columns

TARGET_COMPONENTS = ["future_max_drawdown", "future_downside_volatility", "future_cvar_95", "future_illiquidity"]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataframe_fingerprint(df: pd.DataFrame, *, sample_rows: int = 2000) -> str:
    """Stable-ish fingerprint for run manifests without storing data itself."""
    if len(df) > sample_rows:
        data = df.sort_index().head(sample_rows).copy()
    else:
        data = df.copy()
    payload = data.to_json(orient="split", date_format="iso", default_handler=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jsonable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def make_run_manifest(*, cfg, input_frame: pd.DataFrame, artifact_files: Iterable[str | Path] = ()) -> dict[str, object]:
    files = []
    for path in artifact_files:
        p = Path(path)
        if p.exists() and p.is_file():
            files.append({"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "input_rows": int(len(input_frame)),
        "input_columns": list(map(str, input_frame.columns)),
        "input_fingerprint": dataframe_fingerprint(input_frame),
        "config": _jsonable(cfg),
        "artifact_files": files,
    }


def save_json(obj: dict[str, object], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def validate_no_duplicate_keys(df: pd.DataFrame, keys: list[str]) -> dict[str, object]:
    require_columns(df, keys, "duplicate_key_check")
    duplicates = df.duplicated(keys).sum()
    return {"check": "no_duplicate_keys", "keys": keys, "passed": bool(duplicates == 0), "duplicates": int(duplicates)}


def validate_temporal_split(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, date_col: str = "decision_date") -> dict[str, object]:
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        require_columns(part, [date_col], name)
    train_max = pd.to_datetime(train[date_col]).max()
    val_min = pd.to_datetime(validation[date_col]).min()
    val_max = pd.to_datetime(validation[date_col]).max()
    test_min = pd.to_datetime(test[date_col]).min()
    passed = bool(train_max < val_min and val_max < test_min) if len(train) and len(validation) and len(test) else False
    return {
        "check": "temporal_split_order",
        "passed": passed,
        "train_max": str(train_max),
        "validation_min": str(val_min),
        "validation_max": str(val_max),
        "test_min": str(test_min),
    }


def validate_target_components(df: pd.DataFrame) -> dict[str, object]:
    missing = [c for c in TARGET_COMPONENTS if c not in df.columns]
    if missing:
        return {"check": "target_components_available", "passed": False, "missing": missing}
    rates = {c: float(pd.to_numeric(df[c], errors="coerce").notna().mean()) for c in TARGET_COMPONENTS}
    return {"check": "target_components_available", "passed": all(v > 0 for v in rates.values()), "non_null_rates": rates}


def data_quality_report(df: pd.DataFrame, *, date_col: str = "decision_date", key_cols: list[str] | None = None) -> pd.DataFrame:
    key_cols = key_cols or [date_col, "ticker"]
    checks = []
    try:
        checks.append(validate_no_duplicate_keys(df, key_cols))
    except SchemaError as exc:
        checks.append({"check": "no_duplicate_keys", "passed": False, "error": str(exc)})
    checks.append(validate_target_components(df))
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        checks.append({"check": "date_parse", "passed": bool(dates.notna().all()), "min_date": str(dates.min()), "max_date": str(dates.max())})
    return pd.DataFrame(checks)


def population_stability_index(expected: pd.Series, actual: pd.Series, *, bins: int = 10) -> float:
    """Calculate PSI using train quantile bins."""
    e = pd.to_numeric(expected, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    a = pd.to_numeric(actual, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(e) < 2 or len(a) < 2:
        return np.nan
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.nanquantile(e, quantiles))
    if len(edges) <= 2:
        lo = min(e.min(), a.min())
        hi = max(e.max(), a.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return 0.0
        edges = np.linspace(lo, hi, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected_counts = pd.cut(e, bins=edges, include_lowest=True).value_counts(sort=False).to_numpy(dtype=float)
    actual_counts = pd.cut(a, bins=edges, include_lowest=True).value_counts(sort=False).to_numpy(dtype=float)
    expected_pct = np.maximum(expected_counts / max(expected_counts.sum(), 1), 1e-6)
    actual_pct = np.maximum(actual_counts / max(actual_counts.sum(), 1), 1e-6)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def feature_drift_report(train: pd.DataFrame, other: pd.DataFrame, features: list[str], *, other_name: str = "test", bins: int = 10, warn_threshold: float = 0.20) -> pd.DataFrame:
    rows = []
    for col in features:
        if col not in train.columns or col not in other.columns:
            continue
        train_x = pd.to_numeric(train[col], errors="coerce")
        other_x = pd.to_numeric(other[col], errors="coerce")
        psi = population_stability_index(train_x, other_x, bins=bins)
        rows.append(
            {
                "feature": col,
                "period": other_name,
                "train_missing_rate": float(train_x.isna().mean()),
                f"{other_name}_missing_rate": float(other_x.isna().mean()),
                "train_mean": float(train_x.mean()) if train_x.notna().any() else np.nan,
                f"{other_name}_mean": float(other_x.mean()) if other_x.notna().any() else np.nan,
                "psi": psi,
                "drift_warning": bool(np.isfinite(psi) and psi >= warn_threshold),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last") if rows else pd.DataFrame()

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import augment_monthly_features
from .models import CLASSES


def load_model_package(path: str | Path) -> dict:
    return joblib.load(path)


def predict_model_ready(df: pd.DataFrame, model_package: dict) -> pd.DataFrame:
    """Predict risk classes for model-ready monthly observations.

    The input must contain the same current-date raw features as the training panel;
    target components are not required for inference.
    """
    cfg = model_package["config"]
    data, _ = augment_monthly_features(
        df,
        rank_features=cfg.features.cross_sectional_rank_features,
        enable_derived=cfg.features.enable_derived_features,
    )
    regime = model_package["regime_clusterer"]
    data = data.copy()
    data["regime_cluster"] = regime.transform(data)
    enriched_bundle = model_package["enriched_bundle"]
    ann_bundle = model_package["ann_bundle"]
    ae = model_package["autoencoder"]
    overlay = model_package["overlay"]

    enriched_pred = enriched_bundle.predict(data)
    enriched_proba = enriched_bundle.predict_proba(data)
    data_ae = pd.concat([data.reset_index(drop=True), ae.transform(data).reset_index(drop=True)], axis=1)
    ann_pred = ann_bundle.predict(data_ae)
    ann_proba = ann_bundle.predict_proba(data_ae)
    use_expert = data["sector"].astype(str).isin(overlay.selected_sectors_).to_numpy() if "sector" in data.columns else np.zeros(len(data), dtype=bool)
    final_pred = np.where(use_expert, ann_pred, enriched_pred)
    final_proba = enriched_proba.copy()
    final_proba[use_expert] = ann_proba[use_expert]

    out_cols = [c for c in ["decision_date", "ticker", "sector"] if c in df.columns]
    out = df[out_cols].copy() if out_cols else pd.DataFrame(index=df.index)
    out["predicted_risk_class"] = final_pred
    for i, cls in enumerate(CLASSES):
        out[f"p_{cls}"] = final_proba[:, i]
    out["used_sector_expert"] = use_expert
    return out

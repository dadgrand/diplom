from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import ProjectConfig
from .diagnostics import (
    data_quality_report,
    feature_drift_report,
    make_run_manifest,
    save_json,
    validate_temporal_split,
)
from .ensemble import CandidateBundle, fit_candidate_bundle
from .evaluation import classification_metrics, classwise_metrics_frame, confusion_as_frame, economic_monotonicity, probability_diagnostics
from .feature_engineering import augment_monthly_features
from .models import AutoencoderFactors, CLASSES, RegimeClusterer, SectorExpertOverlay
from .reporting import write_run_report
from .splits import temporal_train_val_test, walk_forward_splits
from .targets import COMPONENTS, TargetRanker


@dataclass
class PipelineResult:
    metrics: dict[str, object]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    artifacts: dict[str, str]


def _unique(seq: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _present(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [c for c in _unique(features) if c in df.columns]


def _prepare_target(df: pd.DataFrame, cfg: ProjectConfig) -> tuple[pd.DataFrame, TargetRanker]:
    split = temporal_train_val_test(df, cfg.split)
    if split.train.empty:
        raise ValueError("Train split is empty; adjust split dates or input data")
    ranker = TargetRanker(weights=cfg.target.weights).fit(split.train)
    labeled = ranker.transform(df).dropna(subset=["risk_class"]).reset_index(drop=True)
    return labeled, ranker


def _fit_bundle(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    cfg: ProjectConfig,
) -> CandidateBundle:
    return fit_candidate_bundle(
        name,
        train,
        validation,
        features=_present(train, features),
        categorical=[c for c in categorical if c in train.columns],
        candidates=cfg.model.candidates,
        n_estimators=cfg.model.n_estimators,
        min_samples_leaf=cfg.model.min_samples_leaf,
        class_weight=cfg.model.class_weight,
        random_state=cfg.random_state,
        n_jobs=cfg.model.n_jobs,
        ensemble_weight_step=cfg.model.ensemble_weight_step,
        calibration_grid=cfg.model.calibration_grid,
        high_recall_bonus=cfg.model.high_recall_bonus,
    )


def _architecture_report(bundle: CandidateBundle, validation: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    val_proba = bundle.predict_proba(validation)
    test_proba = bundle.predict_proba(test)
    val_pred = bundle.policy.predict_from_proba(val_proba)
    test_pred = bundle.policy.predict_from_proba(test_proba)
    return {
        "validation": classification_metrics(validation["risk_class"], val_pred),
        "test": classification_metrics(test["risk_class"], test_pred),
        "validation_probability": probability_diagnostics(validation["risk_class"], val_proba),
        "test_probability": probability_diagnostics(test["risk_class"], test_proba),
        "ensemble_weights": bundle.ensemble.weights_,
        "calibration_gamma": bundle.calibrator.gamma_,
        "policy_thresholds": {"low": bundle.policy.low_threshold_, "high": bundle.policy.high_threshold_},
    }


def _report_feature_selection_score(bundle: CandidateBundle, validation: pd.DataFrame) -> dict[str, float]:
    pred = bundle.predict(validation)
    metrics = classification_metrics(validation["risk_class"], pred)
    # High-risk recall is rewarded because the business task is risk detection,
    # but macro-F1 remains the main optimization target.
    score = float(metrics.get("macro_f1", 0.0) or 0.0) + 0.03 * float(metrics.get("high_recall", 0.0) or 0.0)
    return {
        "selection_score": score,
        "macro_f1": float(metrics.get("macro_f1", 0.0) or 0.0),
        "weighted_f1": float(metrics.get("weighted_f1", 0.0) or 0.0),
        "high_recall": float(metrics.get("high_recall", 0.0) or 0.0),
        "high_precision": float(metrics.get("high_precision", 0.0) or 0.0),
    }


def _risk_detection_selection_score(metrics: dict[str, Any]) -> float:
    """Validation objective for the final architecture.

    Macro-F1 is the main quality signal, while high-risk recall receives a small
    bonus because the applied task is conservative risk detection.
    """
    return float(metrics.get("macro_f1", 0.0) or 0.0) + 0.03 * float(metrics.get("high_recall", 0.0) or 0.0)


def _is_report_feature(name: str) -> bool:
    return name.startswith("report_") or name in {"fundamental_report_gap"}


def _split_walk_forward_train_validation(wf_train: pd.DataFrame, validation_months: int, date_col: str = "decision_date") -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = wf_train.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp["_month"] = tmp[date_col].dt.to_period("M")
    months = sorted(tmp["_month"].unique())
    if len(months) <= validation_months:
        return pd.DataFrame(), pd.DataFrame()
    val_months = months[-validation_months:]
    train_months = months[:-validation_months]
    train = tmp[tmp["_month"].isin(train_months)].drop(columns=["_month"]).reset_index(drop=True)
    validation = tmp[tmp["_month"].isin(val_months)].drop(columns=["_month"]).reset_index(drop=True)
    return train, validation


def _walk_forward_enriched_scores(labeled: pd.DataFrame, enriched_features: list[str], categorical: list[str], cfg: ProjectConfig) -> pd.DataFrame:
    rows = []
    for i, (wf_train_all, wf_test) in enumerate(
        walk_forward_splits(
            labeled,
            initial_months=cfg.walk_forward.initial_months,
            test_months=cfg.walk_forward.test_months,
            step_months=cfg.walk_forward.step_months,
        ),
        start=1,
    ):
        wf_train, wf_val = _split_walk_forward_train_validation(wf_train_all, cfg.walk_forward.validation_months)
        if wf_train.empty or wf_val.empty or wf_test.empty:
            continue
        if min(wf_train["risk_class"].nunique(), wf_val["risk_class"].nunique(), wf_test["risk_class"].nunique()) < 2:
            continue
        wf_regime = RegimeClusterer(cfg.features.regime, n_clusters=cfg.model.regime_k, random_state=cfg.random_state + i).fit(wf_train)
        wf_train = wf_train.copy(); wf_val = wf_val.copy(); wf_test = wf_test.copy()
        wf_train["regime_cluster"] = wf_regime.transform(wf_train)
        wf_val["regime_cluster"] = wf_regime.transform(wf_val)
        wf_test["regime_cluster"] = wf_regime.transform(wf_test)
        # Keep walk-forward lighter than the main fit but still use the same architecture.
        cfg_light = cfg
        bundle = fit_candidate_bundle(
            "wf_enriched",
            wf_train,
            wf_val,
            features=_present(wf_train, enriched_features),
            categorical=[c for c in categorical if c in wf_train.columns],
            candidates=[c for c in cfg.model.candidates if c in {"rf", "extra"}] or ["rf"],
            n_estimators=max(40, cfg.model.n_estimators // 2),
            min_samples_leaf=cfg.model.min_samples_leaf,
            class_weight=cfg.model.class_weight,
            random_state=cfg.random_state + i,
            n_jobs=cfg.model.n_jobs,
            ensemble_weight_step=max(cfg.model.ensemble_weight_step, 0.20),
            calibration_grid=cfg.model.calibration_grid,
            high_recall_bonus=cfg.model.high_recall_bonus,
        )
        pred = bundle.predict(wf_test)
        rows.append({"fold": i, "train_rows": len(wf_train), "validation_rows": len(wf_val), "test_rows": len(wf_test), **classification_metrics(wf_test["risk_class"], pred)})
    return pd.DataFrame(rows)


def run_modeling_pipeline(df: pd.DataFrame, cfg: ProjectConfig, artifact_dir: str | Path | None = None) -> PipelineResult:
    """Run the full modeling stack on a model-ready monthly panel."""
    artifact_dir = Path(artifact_dir or cfg.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    raw_input = df.copy()
    data_quality = data_quality_report(raw_input)
    data_quality.to_csv(artifact_dir / "data_quality_report.csv", index=False)

    augmented, added_features = augment_monthly_features(
        raw_input,
        rank_features=cfg.features.cross_sectional_rank_features,
        enable_derived=cfg.features.enable_derived_features,
    )
    labeled, target_ranker = _prepare_target(augmented, cfg)
    split = temporal_train_val_test(labeled, cfg.split)
    train, validation, test = split.train.copy(), split.validation.copy(), split.test.copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"Non-empty train/validation/test required, got {len(train)}/{len(validation)}/{len(test)}")

    split_check = validate_temporal_split(train, validation, test)

    # Regime labels are fitted only on train and then attached to future periods.
    regime = RegimeClusterer(cfg.features.regime, n_clusters=cfg.model.regime_k, random_state=cfg.random_state).fit(train)
    for part in [train, validation, test]:
        part["regime_cluster"] = regime.transform(part)

    rank_added = [c for c in added_features if c.endswith("_cs_rank") or c.endswith("_sector_z")]
    derived_added = [c for c in cfg.features.derived_features if c in train.columns]
    enhanced_numeric = derived_added + rank_added

    v1_features = _present(train, cfg.features.model_v1 + ["sector"])
    regime_features = _present(train, cfg.features.model_v1 + ["sector", "regime_cluster"])
    enriched_features = _present(train, cfg.features.model_v1 + cfg.features.model_v2_extra + enhanced_numeric + ["sector", "regime_cluster"])

    baseline_bundle = _fit_bundle("baseline_v1", train, validation, v1_features, ["sector"], cfg)
    regime_bundle = _fit_bundle("regime_v1", train, validation, regime_features, ["sector", "regime_cluster"], cfg)

    report_candidate_features = [f for f in enriched_features if _is_report_feature(f)]
    report_layer_selection: dict[str, Any] = {
        "report_features_present": bool(report_candidate_features),
        "selected": "not_available",
        "report_feature_count": int(len(report_candidate_features)),
    }
    enriched_candidates: list[CandidateBundle] = []
    if report_candidate_features:
        enriched_no_report_features = [f for f in enriched_features if not _is_report_feature(f)]
        enriched_no_report_bundle = _fit_bundle(
            "enriched_v2_without_reports",
            train,
            validation,
            enriched_no_report_features,
            ["sector", "regime_cluster"],
            cfg,
        )
        enriched_report_bundle = _fit_bundle(
            "enriched_v2_with_reports",
            train,
            validation,
            enriched_features,
            ["sector", "regime_cluster"],
            cfg,
        )
        no_report_score = _report_feature_selection_score(enriched_no_report_bundle, validation)
        with_report_score = _report_feature_selection_score(enriched_report_bundle, validation)
        min_gain = 0.005
        use_reports = with_report_score["selection_score"] >= no_report_score["selection_score"] + min_gain
        enriched_bundle = enriched_report_bundle if use_reports else enriched_no_report_bundle
        selected_enriched_features = enriched_features if use_reports else enriched_no_report_features
        report_layer_selection.update(
            {
                "selected": "with_reports" if use_reports else "without_reports",
                "min_gain": min_gain,
                "without_reports": no_report_score,
                "with_reports": with_report_score,
            }
        )
        enriched_candidates.extend([enriched_no_report_bundle, enriched_report_bundle])
    else:
        enriched_bundle = _fit_bundle("enriched_v2", train, validation, enriched_features, ["sector", "regime_cluster"], cfg)
        selected_enriched_features = enriched_features
        enriched_candidates.append(enriched_bundle)

    # Autoencoder branch: fit on train numeric features and append latent factors.
    ae_numeric = _present(train, [f for f in cfg.features.model_v1 + cfg.features.model_v2_extra + derived_added if f in selected_enriched_features])
    ae = AutoencoderFactors(
        numeric_features=ae_numeric,
        latent_dim=cfg.model.autoencoder_latent_dim,
        hidden_dim=cfg.model.autoencoder_hidden_dim,
        random_state=cfg.random_state,
        epochs=cfg.model.autoencoder_epochs,
    ).fit(train)
    train_ae = pd.concat([train.reset_index(drop=True), ae.transform(train).reset_index(drop=True)], axis=1)
    val_ae = pd.concat([validation.reset_index(drop=True), ae.transform(validation).reset_index(drop=True)], axis=1)
    test_ae = pd.concat([test.reset_index(drop=True), ae.transform(test).reset_index(drop=True)], axis=1)
    latent_features = [c for c in train_ae.columns if c.startswith("latent_factor_")]
    ann_features = _present(train_ae, selected_enriched_features + latent_features)
    ann_bundle = _fit_bundle("ann_regime_v3", train_ae, val_ae, ann_features, ["sector", "regime_cluster"], cfg)

    val_pred = {
        "baseline_pred": baseline_bundle.predict(validation),
        "regime_pred": regime_bundle.predict(validation),
        "enriched_pred": enriched_bundle.predict(validation),
        "ann_regime_pred": ann_bundle.predict(val_ae),
    }
    test_pred = {
        "baseline_pred": baseline_bundle.predict(test),
        "regime_pred": regime_bundle.predict(test),
        "enriched_pred": enriched_bundle.predict(test),
        "ann_regime_pred": ann_bundle.predict(test_ae),
    }

    val_overlay_frame = validation[["decision_date", "ticker", "sector", "risk_class"]].copy()
    val_overlay_frame["global_pred"] = val_pred["enriched_pred"]
    val_overlay_frame["expert_pred"] = val_pred["ann_regime_pred"]
    overlay = SectorExpertOverlay(cfg.model.overlay_min_sector_rows, cfg.model.overlay_min_gain).fit(
        val_overlay_frame,
        sector_col="sector",
        y_col="risk_class",
        global_pred_col="global_pred",
        expert_pred_col="expert_pred",
    )
    if overlay.sector_report_ is not None:
        overlay.sector_report_.to_csv(artifact_dir / "sector_overlay_report.csv", index=False)

    val_overlay_frame["sector_overlay_pred"] = overlay.predict(
        val_overlay_frame,
        sector_col="sector",
        global_pred_col="global_pred",
        expert_pred_col="expert_pred",
    )

    test_pred_frame = test[["decision_date", "ticker", "sector", "risk_class"]].copy()
    for col, pred in test_pred.items():
        test_pred_frame[col] = pred
    test_pred_frame["sector_overlay_pred"] = overlay.predict(test_pred_frame, sector_col="sector", global_pred_col="enriched_pred", expert_pred_col="ann_regime_pred")

    p_baseline = baseline_bundle.predict_proba(test)
    p_regime = regime_bundle.predict_proba(test)
    p_enriched_val = enriched_bundle.predict_proba(validation)
    p_enriched = enriched_bundle.predict_proba(test)
    p_ann_val = ann_bundle.predict_proba(val_ae)
    p_ann = ann_bundle.predict_proba(test_ae)
    use_expert_val = val_overlay_frame["sector"].astype(str).isin(overlay.selected_sectors_).to_numpy()
    use_expert = test_pred_frame["sector"].astype(str).isin(overlay.selected_sectors_).to_numpy()
    p_sector_overlay_val = p_enriched_val.copy()
    p_sector_overlay_val[use_expert_val] = p_ann_val[use_expert_val]
    p_sector_overlay = p_enriched.copy()
    p_sector_overlay[use_expert] = p_ann[use_expert]

    architecture_metrics = {
        "baseline_rf": _architecture_report(baseline_bundle, validation, test),
        "regime_only": _architecture_report(regime_bundle, validation, test),
        "enriched_reference": _architecture_report(enriched_bundle, validation, test),
        "ann_plus_regime": _architecture_report(ann_bundle, val_ae, test_ae),
    }
    sector_overlay_validation_metrics = classification_metrics(validation["risk_class"], val_overlay_frame["sector_overlay_pred"])
    sector_overlay_test_metrics = classification_metrics(test["risk_class"], test_pred_frame["sector_overlay_pred"])
    sector_overlay_report = {
        "validation": sector_overlay_validation_metrics,
        "test": sector_overlay_test_metrics,
        "validation_probability": probability_diagnostics(validation["risk_class"], p_sector_overlay_val),
        "test_probability": probability_diagnostics(test["risk_class"], p_sector_overlay),
        "selected_sectors": sorted(overlay.selected_sectors_),
    }
    architecture_selection = {
        name: {
            "selection_score": _risk_detection_selection_score(report["validation"]),
            **report["validation"],
        }
        for name, report in architecture_metrics.items()
    }
    architecture_selection["sector_overlay"] = {
        "selection_score": _risk_detection_selection_score(sector_overlay_validation_metrics),
        **sector_overlay_validation_metrics,
    }
    final_architecture = max(
        architecture_selection,
        key=lambda name: (
            architecture_selection[name]["selection_score"],
            architecture_selection[name]["macro_f1"],
            architecture_selection[name]["high_recall"],
        ),
    )
    final_pred_by_arch = {
        "baseline_rf": test_pred["baseline_pred"],
        "regime_only": test_pred["regime_pred"],
        "enriched_reference": test_pred["enriched_pred"],
        "ann_plus_regime": test_pred["ann_regime_pred"],
        "sector_overlay": test_pred_frame["sector_overlay_pred"].to_numpy(),
    }
    final_proba_by_arch = {
        "baseline_rf": p_baseline,
        "regime_only": p_regime,
        "enriched_reference": p_enriched,
        "ann_plus_regime": p_ann,
        "sector_overlay": p_sector_overlay,
    }
    test_pred_frame["predicted_risk_class"] = final_pred_by_arch[final_architecture]
    test_pred_frame["final_architecture"] = final_architecture
    test_pred_frame["used_sector_expert"] = use_expert
    final_proba = final_proba_by_arch[final_architecture]
    for i, cls in enumerate(CLASSES):
        test_pred_frame[f"p_{cls}"] = final_proba[:, i]

    component_cols = [c for c in COMPONENTS if c in test.columns]
    test_pred_frame = test_pred_frame.merge(test[["decision_date", "ticker"] + component_cols], on=["decision_date", "ticker"], how="left")

    drift_features = [c for c in selected_enriched_features if c not in {"sector", "regime_cluster"}]
    drift_val = feature_drift_report(train, validation, drift_features, other_name="validation", bins=cfg.diagnostics.psi_bins, warn_threshold=cfg.diagnostics.drift_psi_warn)
    drift_test = feature_drift_report(train, test, drift_features, other_name="test", bins=cfg.diagnostics.psi_bins, warn_threshold=cfg.diagnostics.drift_psi_warn)
    drift_report = pd.concat([drift_val, drift_test], ignore_index=True)
    drift_report.to_csv(artifact_dir / "feature_drift_report.csv", index=False)

    wf_frame = _walk_forward_enriched_scores(labeled, selected_enriched_features, ["sector", "regime_cluster"], cfg)
    wf_frame.to_csv(artifact_dir / "walk_forward_report.csv", index=False)

    final_metrics = classification_metrics(test["risk_class"], test_pred_frame["predicted_risk_class"])
    metrics: dict[str, object] = {
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "n_test": int(len(test)),
        "class_distribution": labeled["risk_class"].value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict(),
        "target_thresholds": {"low_upper": target_ranker.thresholds_[0], "medium_upper": target_ranker.thresholds_[1]} if target_ranker.thresholds_ else {},
        "added_features_count": int(len(added_features)),
        "added_features": added_features,
        "split_check": split_check,
        "data_quality": data_quality.to_dict(orient="records"),
        "validation_selected_overlay_sectors": sorted(overlay.selected_sectors_),
        "report_layer_selection": report_layer_selection,
        "autoencoder_backend": ae.fitted_with_,
        "autoencoder_reconstruction_loss": ae.reconstruction_loss_,
        "architectures": architecture_metrics,
        "sector_overlay": sector_overlay_report,
        "final_architecture": final_architecture,
        "final_architecture_selection": architecture_selection,
        "test": {
            "baseline_rf": architecture_metrics["baseline_rf"]["test"],
            "regime_only": architecture_metrics["regime_only"]["test"],
            "ann_plus_regime": architecture_metrics["ann_plus_regime"]["test"],
            "enriched_reference": architecture_metrics["enriched_reference"]["test"],
            "sector_overlay": sector_overlay_test_metrics,
            "final_selected": final_metrics,
        },
        "sector_overlay_probability": probability_diagnostics(test["risk_class"], p_sector_overlay),
        "final_probability": probability_diagnostics(test["risk_class"], final_proba),
        "confusion_sector_overlay": confusion_as_frame(test["risk_class"], test_pred_frame["sector_overlay_pred"]).to_dict(),
        "confusion_final": confusion_as_frame(test["risk_class"], test_pred_frame["predicted_risk_class"]).to_dict(),
        "classwise_sector_overlay": classwise_metrics_frame(test["risk_class"], test_pred_frame["sector_overlay_pred"]).to_dict(orient="records"),
        "classwise_final": classwise_metrics_frame(test["risk_class"], test_pred_frame["predicted_risk_class"]).to_dict(orient="records"),
        "economic_monotonicity": economic_monotonicity(test_pred_frame.rename(columns={"predicted_risk_class": "predicted_class"})).to_dict(orient="records"),
        "walk_forward": {
            "folds": int(len(wf_frame)),
            "macro_f1_mean": float(wf_frame["macro_f1"].mean()) if not wf_frame.empty else None,
            "macro_f1_std": float(wf_frame["macro_f1"].std(ddof=0)) if not wf_frame.empty else None,
            "high_recall_mean": float(wf_frame["high_recall"].mean()) if not wf_frame.empty else None,
        },
        "drift_warning_count": int(drift_report.get("drift_warning", pd.Series(dtype=bool)).sum()) if not drift_report.empty else 0,
    }

    leaderboard_frames = []
    for bundle in [baseline_bundle, regime_bundle, *enriched_candidates, ann_bundle]:
        if bundle.validation_leaderboard is not None:
            frame = bundle.validation_leaderboard.copy()
            frame.insert(0, "architecture", bundle.name)
            leaderboard_frames.append(frame)
    model_leaderboard = pd.concat(leaderboard_frames, ignore_index=True) if leaderboard_frames else pd.DataFrame()
    model_leaderboard.to_csv(artifact_dir / "model_leaderboard.csv", index=False)

    feature_importance = enriched_bundle.feature_importance.rename(columns={"weighted_importance": "enriched_importance"})
    feature_importance = feature_importance.merge(
        regime_bundle.feature_importance[["feature", "weighted_importance"]].rename(columns={"weighted_importance": "regime_importance"}),
        on="feature",
        how="outer",
    ).merge(
        ann_bundle.feature_importance[["feature", "weighted_importance"]].rename(columns={"weighted_importance": "ann_branch_importance"}),
        on="feature",
        how="outer",
    ).fillna(0.0)
    feature_importance = feature_importance.sort_values("enriched_importance", ascending=False)

    metrics_path = artifact_dir / "metrics.json"
    predictions_path = artifact_dir / "predictions.csv"
    fi_path = artifact_dir / "feature_importance.csv"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    report_path = artifact_dir / "RUN_REPORT.md"
    write_run_report(metrics, report_path)
    test_pred_frame.to_csv(predictions_path, index=False)
    feature_importance.to_csv(fi_path, index=False)

    artifacts = {
        "metrics": str(metrics_path),
        "predictions": str(predictions_path),
        "feature_importance": str(fi_path),
        "data_quality_report": str(artifact_dir / "data_quality_report.csv"),
        "feature_drift_report": str(artifact_dir / "feature_drift_report.csv"),
        "walk_forward_report": str(artifact_dir / "walk_forward_report.csv"),
        "model_leaderboard": str(artifact_dir / "model_leaderboard.csv"),
        "run_report": str(artifact_dir / "RUN_REPORT.md"),
    }

    if cfg.diagnostics.save_model_package:
        model_package = {
            "classes": CLASSES,
            "features": {"enriched": selected_enriched_features, "ann": ann_features},
            "final_architecture": final_architecture,
            "target_ranker": target_ranker,
            "regime_clusterer": regime,
            "autoencoder": ae,
            "baseline_bundle": baseline_bundle,
            "regime_bundle": regime_bundle,
            "enriched_bundle": enriched_bundle,
            "ann_bundle": ann_bundle,
            "overlay": overlay,
            "config": cfg,
        }
        model_path = artifact_dir / "model_package.joblib"
        joblib.dump(model_package, model_path, compress=3)
        artifacts["model_package"] = str(model_path)

    manifest = make_run_manifest(cfg=cfg, input_frame=raw_input, artifact_files=artifacts.values())
    manifest_path = artifact_dir / "run_manifest.json"
    save_json(manifest, manifest_path)
    artifacts["run_manifest"] = str(manifest_path)

    return PipelineResult(metrics=metrics, predictions=test_pred_frame, feature_importance=feature_importance, artifacts=artifacts)

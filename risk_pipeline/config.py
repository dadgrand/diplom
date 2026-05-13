from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SplitConfig:
    train_start: str = "2023-07-01"
    train_end: str = "2024-08-31"
    validation_start: str = "2024-09-01"
    validation_end: str = "2025-02-28"
    test_start: str = "2025-03-01"
    test_end: str = "2025-08-31"


@dataclass(frozen=True)
class TargetConfig:
    horizon_days: int = 126
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "future_max_drawdown": 0.35,
            "future_downside_volatility": 0.30,
            "future_cvar_95": 0.20,
            "future_illiquidity": 0.15,
        }
    )


@dataclass(frozen=True)
class FeatureConfig:
    categorical: list[str] = field(default_factory=lambda: ["sector", "regime_cluster"])
    regime: list[str] = field(
        default_factory=lambda: [
            "cbr_key_rate",
            "usd_rub",
            "ofz_slope_10y_2y",
            "imoex_realized_vol_20d",
            "average_market_correlation_60d",
        ]
    )
    model_v1: list[str] = field(
        default_factory=lambda: [
            "log_market_cap",
            "book_to_market",
            "beta_60d",
            "rolling_vol_20d",
            "downside_vol_60d",
            "momentum_6m",
            "amihud_20d",
            "avg_daily_traded_value_20d",
            "spread_proxy",
            "turnover_ratio",
            "cbr_key_rate",
            "usd_rub",
            "ofz_slope_10y_2y",
            "imoex_realized_vol_20d",
            "average_market_correlation_60d",
        ]
    )
    model_v2_extra: list[str] = field(
        default_factory=lambda: [
            "net_debt_to_ebitda",
            "interest_coverage",
            "operating_cash_flow",
            "free_cash_flow",
            "ebitda_margin",
            "net_debt_to_ebitda_missing",
            "interest_coverage_missing",
            "cash_flow_missing",
            "report_available",
            "report_stale_flag",
            "report_lag_days",
            "report_extraction_quality",
            "report_document_available",
            "report_financial_pressure",
            "report_revenue",
            "report_ebitda",
            "report_net_profit",
            "report_operating_cash_flow",
            "report_free_cash_flow",
            "report_capex",
            "report_total_debt",
            "report_cash_and_equivalents",
            "report_net_debt",
            "report_net_debt_to_ebitda",
            "report_interest_coverage",
            "report_ebitda_margin",
            "report_net_margin",
            "report_operating_cf_margin",
            "report_free_cf_margin",
            "report_capex_intensity",
            "report_equity_ratio",
            "report_debt_to_assets",
            "report_short_debt_share",
            "report_text_risk_density",
            "report_text_risk_terms_total",
            "report_sanctions_count",
            "report_currency_fx_count",
            "report_liquidity_refinancing_count",
            "report_covenant_flag",
            "report_impairment_count",
            "report_litigation_count",
            "report_auditor_emphasis_flag",
            "report_dividend_pressure_flag",
            "report_revenue_yoy_change",
            "report_ebitda_yoy_change",
            "report_net_profit_yoy_change",
            "report_free_cash_flow_yoy_change",
            "report_total_debt_yoy_change",
            "report_net_debt_yoy_change",
        ]
    )
    enable_derived_features: bool = True
    cross_sectional_rank_features: list[str] = field(
        default_factory=lambda: [
            "log_market_cap",
            "beta_60d",
            "rolling_vol_20d",
            "downside_vol_60d",
            "amihud_20d",
            "avg_daily_traded_value_20d",
            "spread_proxy",
            "turnover_ratio",
            "net_debt_to_ebitda",
            "interest_coverage",
            "free_cash_flow",
            "report_financial_pressure",
            "report_net_debt_to_ebitda",
            "report_interest_coverage",
            "report_free_cf_margin",
            "report_text_risk_density",
            "report_lag_days",
            "report_integrated_stress",
        ]
    )
    derived_features: list[str] = field(
        default_factory=lambda: [
            "downside_to_total_vol",
            "liquidity_stress",
            "liquidity_vol_interaction",
            "beta_vol_interaction",
            "macro_pressure",
            "rate_fx_pressure",
            "curve_inversion_flag",
            "debt_service_stress",
            "cashflow_buffer",
            "size_liquidity_interaction",
            "momentum_reversal_risk",
            "fundamental_fragility",
            "report_leverage_pressure",
            "report_cashflow_pressure",
            "report_staleness_weight",
            "report_integrated_stress",
            "fundamental_report_gap",
        ]
    )


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int = 500
    min_samples_leaf: int = 4
    class_weight: str | None = "balanced_subsample"
    regime_k: int = 2
    autoencoder_latent_dim: int = 3
    autoencoder_hidden_dim: int = 8
    autoencoder_epochs: int = 80
    overlay_min_sector_rows: int = 18
    overlay_min_gain: float = 0.10
    candidates: list[str] = field(default_factory=lambda: ["rf", "extra", "gb"])
    ensemble_weight_step: float = 0.10
    calibration_grid: list[float] = field(default_factory=lambda: [0.65, 0.80, 1.00, 1.25, 1.50, 1.80])
    high_recall_bonus: float = 0.02
    n_jobs: int = -1
    random_state: int = 42


@dataclass(frozen=True)
class WalkForwardConfig:
    initial_months: int = 12
    validation_months: int = 3
    test_months: int = 3
    step_months: int = 3


@dataclass(frozen=True)
class DiagnosticsConfig:
    psi_bins: int = 10
    drift_psi_warn: float = 0.20
    run_data_quality_checks: bool = True
    save_model_package: bool = True


@dataclass(frozen=True)
class ProjectConfig:
    random_state: int = 42
    artifact_dir: str = "results/run"
    split: SplitConfig = field(default_factory=SplitConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


def _deep_get(d: dict[str, Any], path: list[str], default: Any) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _dataclass_kwargs(cls: type, raw: dict[str, Any], section: str, default: Any) -> dict[str, Any]:
    return {k: _deep_get(raw, [section, k], getattr(default, k)) for k in cls.__dataclass_fields__}


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load YAML config and merge it with conservative defaults."""
    raw: dict[str, Any] = {}
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    default = ProjectConfig()
    split = SplitConfig(**_dataclass_kwargs(SplitConfig, raw, "split", default.split))

    target_weights = _deep_get(raw, ["target", "weights"], default.target.weights)
    target = TargetConfig(
        horizon_days=int(_deep_get(raw, ["data", "horizon_days"], _deep_get(raw, ["target", "horizon_days"], default.target.horizon_days))),
        weights={k: float(v) for k, v in target_weights.items()},
    )

    features = FeatureConfig(
        categorical=list(_deep_get(raw, ["features", "categorical"], default.features.categorical)),
        regime=list(_deep_get(raw, ["features", "regime"], default.features.regime)),
        model_v1=list(_deep_get(raw, ["features", "model_v1"], default.features.model_v1)),
        model_v2_extra=list(_deep_get(raw, ["features", "model_v2_extra"], default.features.model_v2_extra)),
        enable_derived_features=bool(_deep_get(raw, ["features", "enable_derived_features"], default.features.enable_derived_features)),
        cross_sectional_rank_features=list(_deep_get(raw, ["features", "cross_sectional_rank_features"], default.features.cross_sectional_rank_features)),
        derived_features=list(_deep_get(raw, ["features", "derived_features"], default.features.derived_features)),
    )

    random_state = int(_deep_get(raw, ["project", "random_state"], default.random_state))
    model = ModelConfig(
        n_estimators=int(_deep_get(raw, ["model", "n_estimators"], default.model.n_estimators)),
        min_samples_leaf=int(_deep_get(raw, ["model", "min_samples_leaf"], default.model.min_samples_leaf)),
        class_weight=_deep_get(raw, ["model", "class_weight"], default.model.class_weight),
        regime_k=int(_deep_get(raw, ["model", "regime_k"], default.model.regime_k)),
        autoencoder_latent_dim=int(_deep_get(raw, ["model", "autoencoder_latent_dim"], default.model.autoencoder_latent_dim)),
        autoencoder_hidden_dim=int(_deep_get(raw, ["model", "autoencoder_hidden_dim"], default.model.autoencoder_hidden_dim)),
        autoencoder_epochs=int(_deep_get(raw, ["model", "autoencoder_epochs"], default.model.autoencoder_epochs)),
        overlay_min_sector_rows=int(_deep_get(raw, ["model", "overlay_min_sector_rows"], default.model.overlay_min_sector_rows)),
        overlay_min_gain=float(_deep_get(raw, ["model", "overlay_min_gain"], default.model.overlay_min_gain)),
        candidates=list(_deep_get(raw, ["model", "candidates"], default.model.candidates)),
        ensemble_weight_step=float(_deep_get(raw, ["model", "ensemble_weight_step"], default.model.ensemble_weight_step)),
        calibration_grid=[float(x) for x in _deep_get(raw, ["model", "calibration_grid"], default.model.calibration_grid)],
        high_recall_bonus=float(_deep_get(raw, ["model", "high_recall_bonus"], default.model.high_recall_bonus)),
        n_jobs=int(_deep_get(raw, ["model", "n_jobs"], default.model.n_jobs)),
        random_state=random_state,
    )

    walk_forward = WalkForwardConfig(
        initial_months=int(_deep_get(raw, ["walk_forward", "initial_months"], default.walk_forward.initial_months)),
        validation_months=int(_deep_get(raw, ["walk_forward", "validation_months"], default.walk_forward.validation_months)),
        test_months=int(_deep_get(raw, ["walk_forward", "test_months"], default.walk_forward.test_months)),
        step_months=int(_deep_get(raw, ["walk_forward", "step_months"], default.walk_forward.step_months)),
    )
    diagnostics = DiagnosticsConfig(
        psi_bins=int(_deep_get(raw, ["diagnostics", "psi_bins"], default.diagnostics.psi_bins)),
        drift_psi_warn=float(_deep_get(raw, ["diagnostics", "drift_psi_warn"], default.diagnostics.drift_psi_warn)),
        run_data_quality_checks=bool(_deep_get(raw, ["diagnostics", "run_data_quality_checks"], default.diagnostics.run_data_quality_checks)),
        save_model_package=bool(_deep_get(raw, ["diagnostics", "save_model_package"], default.diagnostics.save_model_package)),
    )
    return ProjectConfig(
        random_state=random_state,
        artifact_dir=str(_deep_get(raw, ["project", "artifact_dir"], default.artifact_dir)),
        split=split,
        target=target,
        features=features,
        model=model,
        walk_forward=walk_forward,
        diagnostics=diagnostics,
    )

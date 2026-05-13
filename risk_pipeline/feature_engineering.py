from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
TARGET_COLUMNS = {
    "risk_class",
    "risk_score",
    "future_max_drawdown",
    "future_downside_volatility",
    "future_cvar_95",
    "future_illiquidity",
}


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


def add_domain_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add economically interpretable features available at decision time only.

    The function deliberately refuses to use target columns. All features are built
    from contemporaneous market, liquidity, macro and fundamental variables.
    """
    data = df.copy()
    forbidden = TARGET_COLUMNS.intersection(data.columns)
    # Target columns may be present in a model-ready table. They are never read in
    # this function; the variable is kept to make static checks easier.
    _ = forbidden

    rolling_vol = _num(data, "rolling_vol_20d", 0.0).clip(lower=0)
    downside_vol = _num(data, "downside_vol_60d", 0.0).clip(lower=0)
    beta = _num(data, "beta_60d", 0.0)
    momentum = _num(data, "momentum_6m", 0.0)
    amihud = _num(data, "amihud_20d", 0.0).clip(lower=0)
    traded_value = _num(data, "avg_daily_traded_value_20d", 0.0).clip(lower=0)
    spread = _num(data, "spread_proxy", 0.0).clip(lower=0)
    turnover = _num(data, "turnover_ratio", 0.0).clip(lower=0)
    log_market_cap = _num(data, "log_market_cap", 0.0)
    cbr_key_rate = _num(data, "cbr_key_rate", 0.0)
    usd_rub = _num(data, "usd_rub", 0.0)
    ofz_slope = _num(data, "ofz_slope_10y_2y", 0.0)
    market_vol = _num(data, "imoex_realized_vol_20d", 0.0).clip(lower=0)
    market_corr = _num(data, "average_market_correlation_60d", 0.0)
    debt = _num(data, "net_debt_to_ebitda", 0.0)
    coverage = _num(data, "interest_coverage", 0.0)
    op_cf = _num(data, "operating_cash_flow", 0.0)
    free_cf = _num(data, "free_cash_flow", 0.0)
    ebitda_margin = _num(data, "ebitda_margin", 0.0)
    has_report_layer = any(
        c in data.columns
        for c in [
            "report_financial_pressure",
            "report_net_debt_to_ebitda",
            "report_text_risk_density",
            "report_available",
        ]
    )

    data["downside_to_total_vol"] = _safe_div(downside_vol, rolling_vol + EPS)
    data["liquidity_stress"] = np.log1p(amihud * 1e10) + np.log1p(spread * 100.0) - np.log1p(traded_value / 1e7)
    data["liquidity_vol_interaction"] = data["liquidity_stress"] * rolling_vol
    data["beta_vol_interaction"] = beta * rolling_vol
    data["macro_pressure"] = 0.05 * cbr_key_rate + 10.0 * market_vol + 0.50 * market_corr - 0.10 * ofz_slope
    data["rate_fx_pressure"] = cbr_key_rate * np.log1p(usd_rub.clip(lower=0))
    data["curve_inversion_flag"] = (ofz_slope < 0).astype(int)
    data["debt_service_stress"] = debt.clip(lower=0) / (1.0 + coverage.clip(lower=0))
    data["cashflow_buffer"] = free_cf.fillna(0.0) + 0.50 * op_cf.fillna(0.0) + 0.25 * ebitda_margin.fillna(0.0)
    data["size_liquidity_interaction"] = log_market_cap * np.log1p(traded_value)
    data["momentum_reversal_risk"] = (-momentum).clip(lower=0) * (1.0 + beta.clip(lower=0))
    data["fundamental_fragility"] = data["debt_service_stress"] - data["cashflow_buffer"]

    report_cols: list[str] = []
    if has_report_layer:
        report_pressure = _num(data, "report_financial_pressure", 0.0).clip(lower=0)
        report_net_debt_to_ebitda = _num(data, "report_net_debt_to_ebitda", 0.0)
        report_interest_coverage = _num(data, "report_interest_coverage", 0.0)
        report_free_cf_margin = _num(data, "report_free_cf_margin", 0.0)
        report_risk_density = _num(data, "report_text_risk_density", 0.0).clip(lower=0)
        report_lag_days = _num(data, "report_lag_days", 365.0).clip(lower=0)
        report_quality = _num(data, "report_extraction_quality", 0.0).clip(lower=0, upper=1)
        report_available = _num(data, "report_available", 0.0).clip(lower=0, upper=1)
        data["report_leverage_pressure"] = report_net_debt_to_ebitda.clip(lower=0) / (1.0 + report_interest_coverage.clip(lower=0))
        data["report_cashflow_pressure"] = (-report_free_cf_margin).clip(lower=0) + report_risk_density * 25.0
        data["report_staleness_weight"] = np.exp(-report_lag_days / 365.0) * report_available * report_quality
        data["report_integrated_stress"] = (
            0.45 * report_pressure
            + 0.30 * data["report_leverage_pressure"]
            + 0.25 * data["report_cashflow_pressure"]
        ) * data["report_staleness_weight"].fillna(0.0)
        data["fundamental_report_gap"] = data["fundamental_fragility"] - data["report_integrated_stress"]
        report_cols = [
            "report_leverage_pressure",
            "report_cashflow_pressure",
            "report_staleness_weight",
            "report_integrated_stress",
            "fundamental_report_gap",
        ]

    for col in [
        "downside_to_total_vol",
        "liquidity_stress",
        "liquidity_vol_interaction",
        "beta_vol_interaction",
        "macro_pressure",
        "rate_fx_pressure",
        "debt_service_stress",
        "cashflow_buffer",
        "size_liquidity_interaction",
        "momentum_reversal_risk",
        "fundamental_fragility",
        *report_cols,
    ]:
        data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return data


def add_cross_sectional_features(
    df: pd.DataFrame,
    rank_features: list[str],
    *,
    date_col: str = "decision_date",
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Add per-month ranks and optional sector z-scores.

    Cross-sectional ranks reduce sensitivity to market-wide level shifts: the model
    learns whether a stock is relatively risky among peers at the same decision date.
    """
    data = df.copy()
    if date_col not in data.columns:
        return data
    data[date_col] = pd.to_datetime(data[date_col])
    for col in rank_features:
        if col not in data.columns:
            continue
        x = pd.to_numeric(data[col], errors="coerce")
        data[f"{col}_cs_rank"] = x.groupby(data[date_col]).rank(pct=True, method="average")
        if sector_col in data.columns:
            group_keys = [data[date_col], data[sector_col].astype(str)]
            mean = x.groupby(group_keys).transform("mean")
            std = x.groupby(group_keys).transform("std").replace(0, np.nan)
            data[f"{col}_sector_z"] = ((x - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return data


def augment_monthly_features(
    df: pd.DataFrame,
    *,
    rank_features: list[str] | None = None,
    enable_derived: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply all safe feature enrichments and return names of newly added columns."""
    before = set(df.columns)
    data = add_domain_derived_features(df) if enable_derived else df.copy()
    if rank_features:
        data = add_cross_sectional_features(data, rank_features)
    added = [c for c in data.columns if c not in before and c not in TARGET_COLUMNS]
    return data, added

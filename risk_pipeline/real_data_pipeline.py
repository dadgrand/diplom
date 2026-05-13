from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_sources import load_local_csv, save_frame
from .financial_reports import build_financial_report_features, coverage_report, merge_financial_report_features_pit, write_report_layer_summary
from .features import add_fundamental_ratios, compute_daily_features, make_monthly_snapshots, merge_fundamentals_pit, merge_macro_asof
from .schemas import DAILY_PRICE_SCHEMA, MACRO_SCHEMA, require_columns
from .targets import compute_future_components


def build_model_ready_panel(
    daily_prices: pd.DataFrame,
    macro: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    *,
    universe: pd.DataFrame | None = None,
    market: pd.DataFrame | None = None,
    report_features: pd.DataFrame | None = None,
    horizon_days: int = 126,
) -> pd.DataFrame:
    """Build a monthly point-in-time panel from raw tabular inputs.

    The function is intentionally explicit: raw data are transformed through daily
    features, monthly decision snapshots, macro as-of merge, fundamental as-of merge
    and finally future target-component computation.
    """
    DAILY_PRICE_SCHEMA.validate(daily_prices)
    MACRO_SCHEMA.validate(macro)
    daily = daily_prices.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    if universe is not None and not universe.empty:
        require_columns(universe, ["ticker"], "universe")
        include = universe.copy()
        if "include_flag" in include.columns:
            include = include[include["include_flag"].astype(int) == 1]
        daily = daily[daily["ticker"].isin(include["ticker"].astype(str))]

    daily_features = compute_daily_features(daily, market=market)
    monthly = make_monthly_snapshots(daily_features)
    monthly = merge_macro_asof(monthly, macro)

    if universe is not None and "sector" in universe.columns:
        monthly = monthly.merge(universe[["ticker", "sector"]].drop_duplicates("ticker"), on="ticker", how="left", suffixes=("", "_universe"))
        if "sector_universe" in monthly.columns:
            monthly["sector"] = monthly["sector"].fillna(monthly["sector_universe"])
            monthly = monthly.drop(columns=["sector_universe"])
    if "sector" not in monthly.columns:
        monthly["sector"] = "unknown"

    if fundamentals is not None and not fundamentals.empty:
        monthly = merge_fundamentals_pit(monthly, fundamentals)
    monthly = add_fundamental_ratios(monthly)
    if report_features is not None and not report_features.empty:
        monthly = merge_financial_report_features_pit(monthly, report_features)
    monthly = compute_future_components(daily, monthly, horizon_days=horizon_days)
    return monthly.sort_values(["decision_date", "ticker"]).reset_index(drop=True)


def build_model_ready_panel_from_paths(
    *,
    daily_prices_path: str | Path,
    macro_path: str | Path,
    output_path: str | Path,
    fundamentals_path: str | Path | None = None,
    universe_path: str | Path | None = None,
    market_path: str | Path | None = None,
    report_features_path: str | Path | None = None,
    reports_registry_path: str | Path | None = None,
    reports_dir: str | Path | None = None,
    report_features_output_path: str | Path | None = None,
    report_coverage_output_path: str | Path | None = None,
    horizon_days: int = 126,
) -> pd.DataFrame:
    daily = load_local_csv(daily_prices_path)
    macro = load_local_csv(macro_path)
    fundamentals = load_local_csv(fundamentals_path) if fundamentals_path else None
    universe = load_local_csv(universe_path) if universe_path else None
    market = load_local_csv(market_path) if market_path else None
    report_features = None
    if report_features_path:
        report_features = load_local_csv(report_features_path)
    elif reports_registry_path:
        report_features = build_financial_report_features(reports_registry_path, reports_dir=reports_dir, registry_path=reports_registry_path)
        if report_features_output_path:
            save_frame(report_features, report_features_output_path)
    panel_no_reports = build_model_ready_panel(daily, macro, fundamentals, universe=universe, market=market, horizon_days=horizon_days)
    if report_features is not None and not report_features.empty:
        panel = merge_financial_report_features_pit(panel_no_reports, report_features)
        if report_coverage_output_path:
            coverage_report(panel_no_reports, panel).to_csv(report_coverage_output_path, index=False)
            write_report_layer_summary(report_features, panel, Path(report_coverage_output_path).with_suffix(".json"))
    else:
        panel = panel_no_reports
    save_frame(panel, output_path)
    return panel

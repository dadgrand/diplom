from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data_sources import save_frame
from .financial_reports import build_financial_report_features, coverage_report, merge_financial_report_features_pit
from .pipeline import run_modeling_pipeline
from .synthetic import make_synthetic_monthly_panel


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    if not np.isfinite(value):
        return default
    return value


def _make_synthetic_report_text(row: pd.Series, period_end: pd.Timestamp, rng: np.random.Generator) -> str:
    ticker = str(row["ticker"])
    leverage = max(_safe_float(row.get("net_debt_to_ebitda"), 1.0), -1.0)
    coverage = max(_safe_float(row.get("interest_coverage"), 4.0), 0.1)
    fcf_margin = _safe_float(row.get("free_cash_flow"), 0.05)
    op_cf_margin = _safe_float(row.get("operating_cash_flow"), 0.12)
    ebitda_margin = np.clip(_safe_float(row.get("ebitda_margin"), 0.25), 0.03, 0.65)
    size = np.clip(_safe_float(row.get("log_market_cap"), 10.0), 6, 16)
    base_revenue = float(np.exp(size - 5.0) * 1e6 * rng.uniform(0.8, 1.2))
    revenue = max(base_revenue, 1e7)
    ebitda = revenue * ebitda_margin
    net_profit = revenue * np.clip(ebitda_margin - 0.08 - 0.02 * max(leverage, 0), -0.30, 0.40)
    ocf = revenue * op_cf_margin
    fcf = revenue * fcf_margin
    cash = max(0.05 * revenue * rng.uniform(0.4, 1.4), 1e6)
    total_debt = max(cash + leverage * max(ebitda, 1.0), 0.0)
    interest = abs(ebitda / max(coverage, 0.1))
    capex = abs(ocf - fcf)
    total_assets = max(revenue * rng.uniform(1.1, 2.6), revenue)
    equity = max(total_assets - total_debt * rng.uniform(0.8, 1.3), total_assets * 0.05)

    warnings: list[str] = []
    if leverage > 3.0:
        warnings.append("Компания отмечает риск рефинансирования и повышение долговой нагрузки.")
    if coverage < 2.0:
        warnings.append("В отчете раскрыт риск процентных расходов и необходимость контроля ковенантов.")
    if fcf_margin < 0:
        warnings.append("Свободный денежный поток отрицателен из-за инвестиционной программы и роста капитальных затрат.")
    if _safe_float(row.get("rolling_vol_20d"), 0) > 0.035:
        warnings.append("Менеджмент указывает на валютный риск, рыночную волатильность и санкционные ограничения.")
    if not warnings:
        warnings.append("Существенных событий, влияющих на непрерывность деятельности, не выявлено.")

    period = period_end.strftime("%d.%m.%Y")
    return f"""
    Консолидированная финансовая отчетность {ticker} за период, закончившийся {period}.
    Выручка {revenue:,.0f} рублей.
    EBITDA {ebitda:,.0f} рублей.
    Чистая прибыль {net_profit:,.0f} рублей.
    Денежный поток от операционной деятельности {ocf:,.0f} рублей.
    Свободный денежный поток {fcf:,.0f} рублей.
    Капитальные затраты {capex:,.0f} рублей.
    Общий долг {total_debt:,.0f} рублей.
    Денежные средства и их эквиваленты {cash:,.0f} рублей.
    Процентные расходы {interest:,.0f} рублей.
    Итого активы {total_assets:,.0f} рублей.
    Итого капитал {equity:,.0f} рублей.
    {' '.join(warnings)}
    """.replace(",", " ")


def make_synthetic_report_registry(panel: pd.DataFrame, *, random_state: int = 42) -> pd.DataFrame:
    """Create synthetic published reports from point-in-time base features.

    This is a demo/ablation helper only. It uses contemporaneous fundamentals and
    market features, never future target columns.
    """
    rng = np.random.default_rng(random_state)
    data = panel.copy()
    data["decision_date"] = pd.to_datetime(data["decision_date"])
    months = sorted(data["decision_date"].dt.to_period("M").unique())
    quarter_months = [m for m in months if m.month in {3, 6, 9, 12}]
    rows: list[dict[str, Any]] = []
    for ticker, group in data.groupby("ticker"):
        group = group.sort_values("decision_date")
        for period in quarter_months:
            period_end = period.to_timestamp(how="end").normalize()
            hist = group[group["decision_date"] <= period_end]
            if hist.empty:
                continue
            base = hist.iloc[-1]
            publish_delay = 45 if period.month in {3, 6, 9} else 100
            publish_date = period_end + pd.Timedelta(days=publish_delay)
            rows.append(
                {
                    "ticker": ticker,
                    "company_name": ticker,
                    "report_period_end": period_end,
                    "publish_date": publish_date,
                    "report_type": "synthetic_ifrs_interim" if period.month != 12 else "synthetic_ifrs_annual",
                    "accounting_standard": "IFRS",
                    "language": "ru",
                    "source_name": "synthetic_demo",
                    "source_url": None,
                    "local_path": None,
                    "report_text": _make_synthetic_report_text(base, period_end, rng),
                }
            )
    return pd.DataFrame(rows)


def _extract_core_metrics(result) -> dict[str, Any]:
    test = result.metrics.get("test", {})
    overlay = test.get("sector_overlay", {}) if isinstance(test, dict) else {}
    return {
        "macro_f1": overlay.get("macro_f1"),
        "weighted_f1": overlay.get("weighted_f1"),
        "accuracy": overlay.get("accuracy"),
        "high_recall": overlay.get("high_recall"),
        "high_precision": overlay.get("high_precision"),
    }


def run_report_layer_ablation(
    cfg: ProjectConfig,
    *,
    out_dir: str | Path,
    n_tickers: int = 24,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run a compact ablation of the report layer on controlled synthetic data."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_panel = make_synthetic_monthly_panel(n_tickers=n_tickers, random_state=random_state)
    registry = make_synthetic_report_registry(base_panel, random_state=random_state)
    report_features = build_financial_report_features(registry, include_evidence=False)
    report_panel = merge_financial_report_features_pit(base_panel, report_features)

    # Controlled synthetic experiment: accounting/narrative warnings add an
    # unobserved risk component to future outcomes. The baseline gets the same
    # targets but cannot see the report-derived signal; the enhanced run can.
    stress = (
        pd.to_numeric(report_panel.get("report_financial_pressure"), errors="coerce").fillna(0.0)
        + 0.10 * pd.to_numeric(report_panel.get("report_sanctions_flag"), errors="coerce").fillna(0.0)
        + 0.08 * pd.to_numeric(report_panel.get("report_liquidity_refinancing_flag"), errors="coerce").fillna(0.0)
        + 0.06 * pd.to_numeric(report_panel.get("report_covenant_flag"), errors="coerce").fillna(0.0)
    )
    stress_rank = stress.groupby(pd.to_datetime(report_panel["decision_date"]).dt.to_period("M")).rank(pct=True).fillna(0.0)
    adjustment = 1.0 + 0.22 * stress_rank
    target_cols = [
        "future_max_drawdown",
        "future_downside_volatility",
        "future_cvar_95",
        "future_illiquidity",
    ]
    baseline_panel = base_panel.copy()
    for col in target_cols:
        if col in report_panel.columns:
            report_panel[col] = report_panel[col].astype(float) * adjustment
            baseline_panel[col] = report_panel[col].values

    save_frame(baseline_panel, out / "baseline_panel.csv")
    save_frame(registry.drop(columns=["report_text"]), out / "synthetic_report_registry.csv")
    save_frame(report_features, out / "financial_report_features.csv")
    save_frame(report_panel, out / "report_enhanced_panel.csv")
    coverage_report(base_panel, report_panel).to_csv(out / "report_coverage.csv", index=False)

    cfg_light = replace(
        cfg,
        model=replace(
            cfg.model,
            candidates=[c for c in cfg.model.candidates if c in {"rf", "extra"}] or ["rf"],
            n_estimators=min(cfg.model.n_estimators, 24),
            autoencoder_epochs=min(cfg.model.autoencoder_epochs, 6),
            ensemble_weight_step=max(cfg.model.ensemble_weight_step, 0.25),
            n_jobs=1,
        ),
        # The ablation is a quick controlled experiment, not the production
        # validation protocol. The full pipeline still uses walk-forward checks.
        walk_forward=replace(cfg.walk_forward, initial_months=120, validation_months=3, test_months=3, step_months=3),
        diagnostics=replace(cfg.diagnostics, save_model_package=False),
    )
    baseline = run_modeling_pipeline(baseline_panel, cfg_light, artifact_dir=out / "baseline_run")
    enhanced = run_modeling_pipeline(report_panel, cfg_light, artifact_dir=out / "report_enhanced_run")
    b = _extract_core_metrics(baseline)
    e = _extract_core_metrics(enhanced)
    rows = []
    for key in sorted(set(b) | set(e)):
        bv = b.get(key)
        ev = e.get(key)
        rows.append({"metric": key, "baseline": bv, "report_enhanced": ev, "delta": (ev - bv) if bv is not None and ev is not None else None})
    ablation = pd.DataFrame(rows)
    ablation.to_csv(out / "report_layer_ablation.csv", index=False)
    summary = {
        "baseline": b,
        "report_enhanced": e,
        "delta": {r["metric"]: r["delta"] for r in rows},
        "artifacts": {
            "ablation": str(out / "report_layer_ablation.csv"),
            "coverage": str(out / "report_coverage.csv"),
            "features": str(out / "financial_report_features.csv"),
        },
    }
    (out / "REPORT_LAYER_ABLATION_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary

from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = [
    "banks_financials",
    "oil_gas",
    "metals_mining",
    "retail",
    "telecom",
    "technology",
    "utilities",
    "transport",
    "chemicals",
    "real_estate",
    "consumer",
]


def make_synthetic_monthly_panel(n_tickers: int = 54, random_state: int = 42) -> pd.DataFrame:
    """Generate a model-ready monthly panel for a full end-to-end smoke test.

    The generator intentionally embeds nonlinear interactions between macro regime,
    liquidity and fundamentals so that tree models, clustering and sector overlay
    have meaningful work to do. The data are synthetic and must not be interpreted
    as financial evidence.
    """
    rng = np.random.default_rng(random_state)
    dates = pd.date_range("2023-07-31", "2025-08-31", freq="ME")
    tickers = [f"TCK{i:02d}" for i in range(1, n_tickers + 1)]
    sector_map = {ticker: SECTORS[i % len(SECTORS)] for i, ticker in enumerate(tickers)}

    macro = pd.DataFrame({"decision_date": dates})
    t = np.linspace(0, 1, len(macro))
    macro["cbr_key_rate"] = 7 + 5 * t + rng.normal(0, 0.25, len(macro))
    macro["usd_rub"] = 78 + 10 * t + 3 * np.sin(6 * t) + rng.normal(0, 1.0, len(macro))
    macro["ofz_slope_10y_2y"] = 0.8 - 0.6 * t + rng.normal(0, 0.15, len(macro))
    macro["imoex_realized_vol_20d"] = 0.012 + 0.014 * (t > 0.55) + rng.normal(0, 0.002, len(macro))
    macro["average_market_correlation_60d"] = 0.25 + 0.35 * (t > 0.55) + rng.normal(0, 0.04, len(macro))

    rows = []
    for ticker in tickers:
        sector = sector_map[ticker]
        size = rng.normal(11.5, 1.2)
        base_liquidity = rng.normal(8.0, 1.0)
        debt_style = rng.normal(0.0, 0.8) + (sector == "real_estate") * 1.2 + (sector == "banks_financials") * 0.5
        for _, m in macro.iterrows():
            regime_pressure = 0.09 * (m.cbr_key_rate - 10) + 1.7 * m.imoex_realized_vol_20d + 0.30 * m.average_market_correlation_60d
            liquidity_noise = rng.normal(0, 0.4)
            log_market_cap = size + rng.normal(0, 0.25)
            spread_proxy = np.exp(-base_liquidity / 3 + 0.5 * regime_pressure + liquidity_noise) / 100
            amihud_20d = np.exp(-base_liquidity + 1.5 * regime_pressure + rng.normal(0, 0.6)) / 1e6
            turnover_ratio = np.exp(rng.normal(-4.5, 0.7) + 0.2 * (sector == "banks_financials"))
            beta_60d = 0.6 + rng.normal(0, 0.25) + 0.25 * (sector in {"oil_gas", "metals_mining", "banks_financials"})
            rolling_vol_20d = 0.018 + 0.018 * regime_pressure + 0.006 * beta_60d + rng.normal(0, 0.003)
            downside_vol_60d = rolling_vol_20d * (0.75 + 0.20 * rng.random())
            momentum_6m = rng.normal(0.03, 0.12) - 0.03 * regime_pressure
            book_to_market = np.exp(rng.normal(-0.1, 0.35))
            net_debt_to_ebitda = 1.2 + debt_style + rng.normal(0, 0.8)
            interest_coverage = np.exp(rng.normal(1.6, 0.7) - 0.25 * max(net_debt_to_ebitda, 0))
            operating_cash_flow = rng.normal(0.15, 0.12) - 0.04 * debt_style
            free_cash_flow = operating_cash_flow - rng.normal(0.05, 0.08)
            ebitda_margin = rng.normal(0.25, 0.09) - 0.02 * (sector == "retail")
            missing_fund = rng.random() < (0.07 + 0.04 * (sector in {"technology", "consumer"}))

            risk_latent = (
                1.2 * regime_pressure
                + 0.65 * beta_60d
                + 9.5 * spread_proxy
                + 0.20 * max(net_debt_to_ebitda, 0)
                - 0.17 * log_market_cap
                - 0.35 * interest_coverage / (1 + interest_coverage)
                - 0.40 * momentum_6m
                + 0.20 * (sector == "banks_financials") * regime_pressure
                + rng.normal(0, 0.25)
            )
            future_max_drawdown = np.clip(0.06 + 0.055 * risk_latent + rng.normal(0, 0.015), 0.0, 0.55)
            future_downside_volatility = np.clip(0.010 + 0.010 * risk_latent + rng.normal(0, 0.003), 0.001, 0.12)
            future_cvar_95 = np.clip(0.015 + 0.014 * risk_latent + rng.normal(0, 0.004), 0.001, 0.16)
            future_illiquidity = np.clip(1e-10 * np.exp(risk_latent + rng.normal(0, 0.45)), 1e-12, 1e-7)

            rows.append(
                {
                    "decision_date": m.decision_date,
                    "ticker": ticker,
                    "sector": sector,
                    "log_market_cap": log_market_cap,
                    "book_to_market": book_to_market,
                    "beta_60d": beta_60d,
                    "rolling_vol_20d": rolling_vol_20d,
                    "downside_vol_60d": downside_vol_60d,
                    "momentum_6m": momentum_6m,
                    "amihud_20d": amihud_20d,
                    "avg_daily_traded_value_20d": np.exp(base_liquidity + rng.normal(0, 0.7)) * 1e5,
                    "spread_proxy": spread_proxy,
                    "turnover_ratio": turnover_ratio,
                    "cbr_key_rate": m.cbr_key_rate,
                    "usd_rub": m.usd_rub,
                    "ofz_slope_10y_2y": m.ofz_slope_10y_2y,
                    "imoex_realized_vol_20d": m.imoex_realized_vol_20d,
                    "average_market_correlation_60d": m.average_market_correlation_60d,
                    "net_debt_to_ebitda": np.nan if missing_fund else net_debt_to_ebitda,
                    "interest_coverage": np.nan if missing_fund else interest_coverage,
                    "operating_cash_flow": np.nan if missing_fund else operating_cash_flow,
                    "free_cash_flow": np.nan if missing_fund else free_cash_flow,
                    "ebitda_margin": np.nan if missing_fund else ebitda_margin,
                    "net_debt_to_ebitda_missing": int(missing_fund),
                    "interest_coverage_missing": int(missing_fund),
                    "cash_flow_missing": int(missing_fund),
                    "future_max_drawdown": future_max_drawdown,
                    "future_downside_volatility": future_downside_volatility,
                    "future_cvar_95": future_cvar_95,
                    "future_illiquidity": future_illiquidity,
                }
            )
    df = pd.DataFrame(rows)
    # Drop a reproducible fraction to mimic unavailable targets and readiness flags.
    keep = rng.random(len(df)) > 0.08
    return df.loc[keep].sort_values(["decision_date", "ticker"]).reset_index(drop=True)

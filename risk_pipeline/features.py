from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import DAILY_PRICE_SCHEMA, FUNDAMENTAL_SCHEMA, MACRO_SCHEMA, require_columns

EPS = 1e-12


def _downside_std(x: pd.Series) -> float:
    values = x[x < 0]
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1))


def _rolling_beta(stock_ret: pd.Series, market_ret: pd.Series, window: int = 60) -> pd.Series:
    cov = stock_ret.rolling(window, min_periods=max(20, window // 3)).cov(market_ret)
    var = market_ret.rolling(window, min_periods=max(20, window // 3)).var()
    return cov / (var + EPS)


def add_market_return(daily: pd.DataFrame, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add stock and market returns.

    If a separate market dataframe is not supplied, the equally weighted average
    stock return by date is used as a market proxy. For production use, pass IMOEX
    or another benchmark series as `market` with columns `date` and `market_close`.
    """
    DAILY_PRICE_SCHEMA.validate(daily)
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    df["return_1d"] = df.groupby("ticker")["close"].pct_change()

    if market is not None:
        m = market.copy()
        m["date"] = pd.to_datetime(m["date"])
        close_col = "market_close" if "market_close" in m.columns else "close"
        m = m.sort_values("date")[["date", close_col]].drop_duplicates("date")
        m["market_return"] = m[close_col].pct_change()
        df = df.merge(m[["date", "market_return"]], on="date", how="left")
    elif "market_return" not in df.columns:
        df["market_return"] = df.groupby("date")["return_1d"].transform("mean")
    return df


def compute_daily_features(daily: pd.DataFrame, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute rolling market, liquidity and price-behaviour features."""
    df = add_market_return(daily, market)
    df = df.sort_values(["ticker", "date"]).copy()

    if "high" not in df.columns:
        df["high"] = df["close"]
    if "low" not in df.columns:
        df["low"] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = np.nan
    if "shares_outstanding" not in df.columns:
        df["shares_outstanding"] = np.nan

    grouped = df.groupby("ticker", group_keys=False)
    df["rolling_vol_20d"] = grouped["return_1d"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    df["downside_vol_60d"] = grouped["return_1d"].rolling(60, min_periods=20).apply(_downside_std, raw=False).reset_index(level=0, drop=True)
    df["momentum_6m"] = grouped["close"].pct_change(126)
    df["avg_daily_traded_value_20d"] = grouped["value"].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
    df["amihud_raw"] = df["return_1d"].abs() / (df["value"].abs() + EPS)
    df["amihud_20d"] = grouped["amihud_raw"].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
    df["spread_proxy"] = ((df["high"] - df["low"]).abs() / (df["close"].abs() + EPS)).clip(lower=0)
    df["turnover_ratio"] = df["volume"] / (df["shares_outstanding"] + EPS)
    beta_parts = []
    for _, g in df.groupby("ticker", sort=False):
        beta = _rolling_beta(g["return_1d"], g["market_return"], 60)
        beta_parts.append(beta)
    df["beta_60d"] = pd.concat(beta_parts).sort_index() if beta_parts else np.nan
    df["market_cap"] = df["close"] * df["shares_outstanding"]
    df["log_market_cap"] = np.log1p(df["market_cap"].clip(lower=0))
    return df.drop(columns=["amihud_raw"])


def make_monthly_snapshots(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Take the last available trading observation for each ticker-month."""
    require_columns(daily_features, ["date", "ticker"], "daily_features")
    df = daily_features.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")
    idx = df.sort_values("date").groupby(["ticker", "month"]).tail(1).index
    monthly = df.loc[idx].copy().sort_values(["date", "ticker"])
    monthly = monthly.rename(columns={"date": "decision_date"})
    monthly = monthly.drop(columns=["month"])
    return monthly.reset_index(drop=True)


def merge_macro_asof(monthly: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time merge of monthly observations and macro rows by date."""
    MACRO_SCHEMA.validate(macro)
    require_columns(monthly, ["decision_date"], "monthly")
    left = monthly.copy()
    right = macro.copy()
    left["decision_date"] = pd.to_datetime(left["decision_date"])
    right["date"] = pd.to_datetime(right["date"])
    left = left.sort_values("decision_date")
    right = right.sort_values("date")
    merged = pd.merge_asof(left, right, left_on="decision_date", right_on="date", direction="backward")
    return merged.drop(columns=["date"])


def merge_fundamentals_pit(monthly: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Attach latest published fundamentals for each ticker.

    A row with `report_date=2024-03-31` and `publish_date=2024-05-15` becomes
    available only for decisions on or after 2024-05-15.
    """
    FUNDAMENTAL_SCHEMA.validate(fundamentals)
    require_columns(monthly, ["decision_date", "ticker"], "monthly")
    m = monthly.copy()
    f = fundamentals.copy()
    m["decision_date"] = pd.to_datetime(m["decision_date"])
    f["publish_date"] = pd.to_datetime(f["publish_date"])
    f["report_date"] = pd.to_datetime(f["report_date"])
    pieces: list[pd.DataFrame] = []
    for ticker, left in m.groupby("ticker", sort=False):
        right = f[f["ticker"] == ticker].sort_values("publish_date")
        if right.empty:
            pieces.append(left)
            continue
        merged = pd.merge_asof(
            left.sort_values("decision_date"),
            right.drop(columns=["ticker"]).sort_values("publish_date"),
            left_on="decision_date",
            right_on="publish_date",
            direction="backward",
        )
        merged["ticker"] = ticker
        pieces.append(merged)
    result = pd.concat(pieces, ignore_index=True).sort_values(["decision_date", "ticker"])
    return result.reset_index(drop=True)


def add_fundamental_ratios(monthly: pd.DataFrame) -> pd.DataFrame:
    """Compute robust fundamental features and missingness flags."""
    df = monthly.copy()
    for col in ["net_debt", "ebitda", "interest_expense", "operating_cash_flow", "free_cash_flow", "book_equity", "market_cap"]:
        if col not in df.columns:
            df[col] = np.nan
    df["net_debt_to_ebitda_missing"] = df[["net_debt", "ebitda"]].isna().any(axis=1).astype(int)
    df["interest_coverage_missing"] = df[["ebitda", "interest_expense"]].isna().any(axis=1).astype(int)
    df["cash_flow_missing"] = df[["operating_cash_flow", "free_cash_flow"]].isna().any(axis=1).astype(int)
    df["net_debt_to_ebitda"] = df["net_debt"] / (df["ebitda"].replace(0, np.nan))
    df["interest_coverage"] = df["ebitda"] / (df["interest_expense"].abs().replace(0, np.nan))
    df["ebitda_margin"] = df["ebitda"] / (df.get("revenue", pd.Series(np.nan, index=df.index)).replace(0, np.nan))
    df["book_to_market"] = df["book_equity"] / (df["market_cap"].replace(0, np.nan))
    return df

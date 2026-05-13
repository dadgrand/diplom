from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from risk_pipeline.data_sources import MOEX_BASE_URL, fetch_moex_history


CBR_FX_DYNAMIC_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
CBR_KEY_RATE_URL = "https://www.cbr.ru/hd_base/KeyRate/"
CBR_ZCYC_URL = "https://www.cbr.ru/hd_base/zcyc_params/zcyc/"


def _request_json(url: str, params: dict[str, Any] | None = None, retries: int = 4, sleep: float = 1.0) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params or {}, timeout=35)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(sleep * attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def _moex_table(payload: dict[str, Any], name: str) -> pd.DataFrame:
    table = payload.get(name, {})
    if "columns" not in table or "data" not in table:
        return pd.DataFrame()
    return pd.DataFrame(table["data"], columns=table["columns"])


def load_universe(path: Path) -> pd.DataFrame:
    universe = pd.read_csv(path)
    universe = universe[["ticker", "company_name", "sector"]].drop_duplicates("ticker")
    universe["include_flag"] = 1
    return universe


def fetch_moex_board_metadata(board: str = "TQBR") -> pd.DataFrame:
    url = f"{MOEX_BASE_URL}/engines/stock/markets/shares/boards/{board}/securities.json"
    payload = _request_json(url, params={"iss.meta": "off"})
    securities = _moex_table(payload, "securities")
    if securities.empty:
        return securities
    securities.columns = [c.lower() for c in securities.columns]
    keep = [c for c in ["secid", "shortname", "secname", "issuesize", "isin", "prevdate"] if c in securities.columns]
    return securities[keep].rename(columns={"secid": "ticker", "issuesize": "shares_outstanding"})


def fetch_daily_prices(universe: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = fetch_moex_board_metadata()
    shares = metadata[["ticker", "shares_outstanding"]].drop_duplicates("ticker") if not metadata.empty else pd.DataFrame()
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for row in universe.itertuples(index=False):
        ticker = str(row.ticker)
        try:
            frame = fetch_moex_history(ticker, start, end)
            if not shares.empty:
                frame = frame.merge(shares, on="ticker", how="left")
            else:
                frame["shares_outstanding"] = np.nan
            if not frame.empty:
                frames.append(frame)
            audit_rows.append(
                {
                    "ticker": ticker,
                    "rows": int(len(frame)),
                    "min_date": frame["date"].min() if not frame.empty else pd.NaT,
                    "max_date": frame["date"].max() if not frame.empty else pd.NaT,
                    "shares_outstanding_known": bool(frame.get("shares_outstanding", pd.Series(dtype=float)).notna().any()),
                    "status": "ok" if not frame.empty else "empty",
                    "error": "",
                }
            )
        except Exception as exc:
            audit_rows.append(
                {
                    "ticker": ticker,
                    "rows": 0,
                    "min_date": pd.NaT,
                    "max_date": pd.NaT,
                    "shares_outstanding_known": False,
                    "status": "failed",
                    "error": str(exc)[:500],
                }
            )
        time.sleep(0.15)
    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    return daily, pd.DataFrame(audit_rows)


def fetch_market_index(start: str, end: str, ticker: str = "IMOEX") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        url = f"{MOEX_BASE_URL}/history/engines/stock/markets/index/securities/{ticker}.json"
        payload = _request_json(
            url,
            params={
                "from": start,
                "till": end,
                "start": offset,
                "iss.meta": "off",
                "history.columns": "TRADEDATE,SECID,OPEN,HIGH,LOW,CLOSE,VALUE,CAPITALIZATION",
            },
        )
        frame = _moex_table(payload, "history")
        if frame.empty:
            break
        frames.append(frame)
        if len(frame) < 100:
            break
        offset += 100
        time.sleep(0.05)
    if not frames:
        return pd.DataFrame(columns=["date", "market_close"])
    result = pd.concat(frames, ignore_index=True).rename(
        columns={
            "TRADEDATE": "date",
            "SECID": "ticker",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "market_close",
            "VALUE": "value",
            "CAPITALIZATION": "capitalization",
        }
    )
    result["date"] = pd.to_datetime(result["date"])
    for col in ["open", "high", "low", "market_close", "value", "capitalization"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result.sort_values("date").reset_index(drop=True)


def fetch_cbr_fx(start: str, end: str, code: str = "R01235") -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    response = requests.get(
        CBR_FX_DYNAMIC_URL,
        params={
            "date_req1": start_dt.strftime("%d/%m/%Y"),
            "date_req2": end_dt.strftime("%d/%m/%Y"),
            "VAL_NM_RQ": code,
        },
        timeout=45,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    rows = []
    for rec in root.findall("Record"):
        value = rec.findtext("VunitRate") or rec.findtext("Value")
        nominal = rec.findtext("Nominal") or "1"
        if value is None:
            continue
        rows.append(
            {
                "date": pd.to_datetime(rec.attrib["Date"], dayfirst=True),
                "usd_rub": float(value.replace(",", ".")) / float(nominal.replace(",", ".")),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def fetch_cbr_key_rate(start: str, end: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    url = (
        f"{CBR_KEY_RATE_URL}?UniDbQuery.Posted=True"
        f"&UniDbQuery.From={start_dt.strftime('%d.%m.%Y')}"
        f"&UniDbQuery.To={end_dt.strftime('%d.%m.%Y')}"
    )
    table = pd.read_html(url, decimal=",", thousands=" ")[0]
    table = table.rename(columns={"Дата": "date", "Ставка": "cbr_key_rate"})
    table["date"] = pd.to_datetime(table["date"], dayfirst=True)
    table["cbr_key_rate"] = pd.to_numeric(table["cbr_key_rate"], errors="coerce")
    return table[["date", "cbr_key_rate"]].dropna().sort_values("date").reset_index(drop=True)


def _zcyc_for_date(day: pd.Timestamp) -> float | None:
    # The CBR table contains dashes for weekends and some holidays. Try nearby
    # previous dates so the curve remains point-in-time.
    for shift in range(0, 8):
        candidate = day - pd.Timedelta(days=shift)
        url = f"{CBR_ZCYC_URL}?DateTo={candidate.strftime('%d.%m.%Y')}"
        try:
            table = pd.read_html(url, decimal=",", thousands=" ")[0]
        except Exception:
            continue
        cols = {str(c).strip(): c for c in table.columns}
        c2 = cols.get("2.00")
        c10 = cols.get("10.00")
        if c2 is None or c10 is None:
            continue
        y2 = pd.to_numeric(table.loc[0, c2], errors="coerce")
        y10 = pd.to_numeric(table.loc[0, c10], errors="coerce")
        if pd.notna(y2) and pd.notna(y10):
            return float(y10 - y2)
        time.sleep(0.05)
    return None


def fetch_monthly_ofz_slope(dates: pd.Series) -> pd.DataFrame:
    month_end_dates = (
        pd.DataFrame({"date": pd.to_datetime(dates).dropna().sort_values().unique()})
        .assign(month=lambda x: x["date"].dt.to_period("M"))
        .groupby("month", as_index=False)["date"]
        .max()
    )
    rows = []
    for day in month_end_dates["date"]:
        rows.append({"date": day, "ofz_slope_10y_2y": _zcyc_for_date(day)})
        time.sleep(0.20)
    return pd.DataFrame(rows)


def compute_average_market_correlation(daily: pd.DataFrame, market_dates: pd.Series) -> pd.DataFrame:
    close = daily.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    returns = close.pct_change(fill_method=None)
    values = []
    for idx, day in enumerate(returns.index):
        start_idx = max(0, idx - 59)
        window = returns.iloc[start_idx : idx + 1]
        if window.notna().sum().max() < 20:
            values.append(np.nan)
            continue
        corr = window.corr(min_periods=20).to_numpy(dtype=float)
        if corr.shape[0] < 2:
            values.append(np.nan)
            continue
        tri = corr[np.triu_indices_from(corr, k=1)]
        values.append(float(np.nanmean(tri)) if np.isfinite(tri).any() else np.nan)
    corr = pd.DataFrame({"date": returns.index, "average_market_correlation_60d": values})
    frame = pd.DataFrame({"date": pd.to_datetime(market_dates).dropna().sort_values().unique()})
    frame = frame.merge(corr, on="date", how="left")
    frame["average_market_correlation_60d"] = frame["average_market_correlation_60d"].ffill().bfill()
    return frame


def build_macro(daily: pd.DataFrame, market: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.DataFrame({"date": market["date"].dropna().sort_values().unique()})
    fx = fetch_cbr_fx(start, end)
    key = fetch_cbr_key_rate(start, end)
    ofz = fetch_monthly_ofz_slope(dates["date"])
    corr = compute_average_market_correlation(daily, dates["date"])

    m = market[["date", "market_close"]].dropna().sort_values("date").copy()
    m["market_return"] = m["market_close"].pct_change(fill_method=None)
    m["imoex_realized_vol_20d"] = m["market_return"].rolling(20, min_periods=10).std()

    macro = dates.copy().sort_values("date")
    for frame in [fx, key, ofz, m[["date", "imoex_realized_vol_20d"]], corr]:
        macro = pd.merge_asof(
            macro.sort_values("date"),
            frame.sort_values("date"),
            on="date",
            direction="backward",
        )
    for col in ["usd_rub", "cbr_key_rate", "ofz_slope_10y_2y", "imoex_realized_vol_20d", "average_market_correlation_60d"]:
        macro[col] = pd.to_numeric(macro[col], errors="coerce").ffill().bfill()
    return macro[["date", "cbr_key_rate", "usd_rub", "ofz_slope_10y_2y", "imoex_realized_vol_20d", "average_market_correlation_60d"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MOEX/CBR market data for the risk pipeline")
    parser.add_argument("--universe-source", default="data/report_sources_ru_bluechips.csv")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-05-13")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--universe-output", default="data/universe.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.universe_output).parent.mkdir(parents=True, exist_ok=True)

    universe = load_universe(Path(args.universe_source))
    universe.to_csv(args.universe_output, index=False, encoding="utf-8")

    daily, audit = fetch_daily_prices(universe, args.start, args.end)
    if daily.empty:
        raise RuntimeError("No MOEX daily price rows downloaded")
    daily.to_csv(out_dir / "daily_prices.csv", index=False, encoding="utf-8")
    audit.to_csv(out_dir / "daily_prices_download_audit.csv", index=False, encoding="utf-8")

    market = fetch_market_index(args.start, args.end)
    if market.empty:
        raise RuntimeError("No MOEX market index rows downloaded")
    market.to_csv(out_dir / "market.csv", index=False, encoding="utf-8")

    macro = build_macro(daily, market, args.start, args.end)
    macro.to_csv(out_dir / "macro.csv", index=False, encoding="utf-8")

    print(
        {
            "daily_rows": int(len(daily)),
            "daily_tickers": int(daily["ticker"].nunique()),
            "market_rows": int(len(market)),
            "macro_rows": int(len(macro)),
            "daily_min": str(daily["date"].min().date()),
            "daily_max": str(daily["date"].max().date()),
            "files": {
                "universe": args.universe_output,
                "daily_prices": str(out_dir / "daily_prices.csv"),
                "market": str(out_dir / "market.csv"),
                "macro": str(out_dir / "macro.csv"),
                "audit": str(out_dir / "daily_prices_download_audit.csv"),
            },
        }
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

MOEX_BASE_URL = "https://iss.moex.com/iss"
CBR_DAILY_XML_URL = "https://www.cbr.ru/scripts/XML_daily.asp"


class DownloadError(RuntimeError):
    """Raised when an external data source cannot be loaded."""


def _request_json(url: str, params: dict[str, Any] | None = None, retries: int = 3, sleep: float = 0.8) -> dict[str, Any]:
    params = params or {}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network-dependent
            last_error = exc
            LOGGER.warning("request failed: attempt=%s url=%s error=%s", attempt, url, exc)
            time.sleep(sleep * attempt)
    raise DownloadError(f"Could not download {url}: {last_error}")


def _moex_table_to_frame(payload: dict[str, Any], table_name: str) -> pd.DataFrame:
    if table_name not in payload or "columns" not in payload[table_name] or "data" not in payload[table_name]:
        return pd.DataFrame()
    return pd.DataFrame(payload[table_name]["data"], columns=payload[table_name]["columns"])


def fetch_moex_history(
    ticker: str,
    start: str,
    end: str,
    *,
    board: str = "TQBR",
    engine: str = "stock",
    market: str = "shares",
    page_size: int = 100,
) -> pd.DataFrame:
    """Download daily historical rows from MOEX ISS history endpoint.

    The function uses pagination through the `start` parameter and returns a tidy
    dataframe with lowercase column names. It does not adjust prices for corporate
    events; such events should be handled in a separate quality-control layer.
    """
    rows: list[pd.DataFrame] = []
    offset = 0
    while True:  # pragma: no cover - network-dependent
        url = f"{MOEX_BASE_URL}/history/engines/{engine}/markets/{market}/boards/{board}/securities/{ticker}.json"
        params = {
            "from": start,
            "till": end,
            "start": offset,
            "iss.meta": "off",
            "history.columns": "TRADEDATE,SECID,OPEN,HIGH,LOW,CLOSE,VOLUME,VALUE,NUMTRADES,WAPRICE",
        }
        payload = _request_json(url, params=params)
        frame = _moex_table_to_frame(payload, "history")
        if frame.empty:
            break
        rows.append(frame)
        if len(frame) < page_size:
            break
        offset += page_size
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume", "value", "num_trades", "wap"])

    result = pd.concat(rows, ignore_index=True)
    result = result.rename(
        columns={
            "TRADEDATE": "date",
            "SECID": "ticker",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "VOLUME": "volume",
            "VALUE": "value",
            "NUMTRADES": "num_trades",
            "WAPRICE": "wap",
        }
    )
    result["date"] = pd.to_datetime(result["date"])
    for col in ["open", "high", "low", "close", "volume", "value", "num_trades", "wap"]:
        if col in result:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result.sort_values(["ticker", "date"]).reset_index(drop=True)


def fetch_moex_universe(board: str = "TQBR", engine: str = "stock", market: str = "shares") -> pd.DataFrame:
    """Load securities metadata for a MOEX board.

    This is a helper for building a universe; manual filtering remains necessary
    because research-grade selection must exclude depositary receipts, duplicate
    low-liquidity share classes and too-short histories.
    """
    url = f"{MOEX_BASE_URL}/engines/{engine}/markets/{market}/boards/{board}/securities.json"
    payload = _request_json(url, params={"iss.meta": "off"})  # pragma: no cover - network-dependent
    securities = _moex_table_to_frame(payload, "securities")
    if securities.empty:
        return securities
    securities.columns = [c.lower() for c in securities.columns]
    return securities


def fetch_cbr_fx_xml(day: str | date | datetime, char_code: str = "USD") -> float:
    """Fetch official Bank of Russia FX rate from XML_daily endpoint."""
    if isinstance(day, str):
        dt = pd.to_datetime(day).date()
    elif isinstance(day, datetime):
        dt = day.date()
    else:
        dt = day
    params = {"date_req": dt.strftime("%d/%m/%Y")}
    response = requests.get(CBR_DAILY_XML_URL, params=params, timeout=30)  # pragma: no cover - network-dependent
    response.raise_for_status()
    root = ET.fromstring(response.content)
    for valute in root.findall("Valute"):
        if valute.findtext("CharCode") == char_code:
            nominal = float((valute.findtext("Nominal") or "1").replace(",", "."))
            value = float((valute.findtext("Value") or "nan").replace(",", "."))
            return value / nominal
    raise DownloadError(f"Currency {char_code} not found for {day}")


def load_local_csv(path: str | Path, date_cols: tuple[str, ...] = ("date", "decision_date", "publish_date", "report_date")) -> pd.DataFrame:
    """Read CSV or parquet and parse standard date columns when present."""
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def save_frame(df: pd.DataFrame, path: str | Path) -> None:
    """Save a dataframe with parquet preferred and CSV fallback."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return
        except Exception as exc:
            LOGGER.warning("parquet save failed, falling back to CSV: %s", exc)
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)

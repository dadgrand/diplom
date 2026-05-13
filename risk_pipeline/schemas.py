from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class RequiredColumns:
    name: str
    columns: tuple[str, ...]

    def validate(self, df: pd.DataFrame) -> None:
        missing = sorted(set(self.columns) - set(df.columns))
        if missing:
            raise SchemaError(f"{self.name}: missing required columns: {missing}")


DAILY_PRICE_SCHEMA = RequiredColumns("daily_prices", ("date", "ticker", "close", "value"))
MONTHLY_MODEL_SCHEMA = RequiredColumns("monthly_model", ("decision_date", "ticker", "sector", "risk_class"))
MACRO_SCHEMA = RequiredColumns(
    "macro",
    (
        "date",
        "cbr_key_rate",
        "usd_rub",
        "ofz_slope_10y_2y",
        "imoex_realized_vol_20d",
        "average_market_correlation_60d",
    ),
)
FUNDAMENTAL_SCHEMA = RequiredColumns("fundamentals", ("ticker", "report_date", "publish_date"))


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str = "dataframe") -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise SchemaError(f"{name}: missing required columns: {missing}")

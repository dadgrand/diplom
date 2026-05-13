from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from .config import SplitConfig
from .schemas import require_columns


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_train_val_test(df: pd.DataFrame, split: SplitConfig, date_col: str = "decision_date") -> TemporalSplit:
    require_columns(df, [date_col], "df")
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    train_mask = (data[date_col] >= pd.Timestamp(split.train_start)) & (data[date_col] <= pd.Timestamp(split.train_end))
    val_mask = (data[date_col] >= pd.Timestamp(split.validation_start)) & (data[date_col] <= pd.Timestamp(split.validation_end))
    test_mask = (data[date_col] >= pd.Timestamp(split.test_start)) & (data[date_col] <= pd.Timestamp(split.test_end))
    return TemporalSplit(
        train=data.loc[train_mask].sort_values([date_col]).reset_index(drop=True),
        validation=data.loc[val_mask].sort_values([date_col]).reset_index(drop=True),
        test=data.loc[test_mask].sort_values([date_col]).reset_index(drop=True),
    )


def walk_forward_splits(
    df: pd.DataFrame,
    *,
    date_col: str = "decision_date",
    initial_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield expanding-window walk-forward splits by calendar month."""
    require_columns(df, [date_col], "df")
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data["_month"] = data[date_col].dt.to_period("M")
    months = sorted(data["_month"].unique())
    start = initial_months
    while start + test_months <= len(months):
        train_months = months[:start]
        test_window = months[start : start + test_months]
        train = data[data["_month"].isin(train_months)].drop(columns=["_month"]).reset_index(drop=True)
        test = data[data["_month"].isin(test_window)].drop(columns=["_month"]).reset_index(drop=True)
        if not train.empty and not test.empty:
            yield train, test
        start += step_months

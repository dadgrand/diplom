from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schemas import DAILY_PRICE_SCHEMA, require_columns

EPS = 1e-12
COMPONENTS = ["future_max_drawdown", "future_downside_volatility", "future_cvar_95", "future_illiquidity"]
DEFAULT_WEIGHTS = {
    "future_max_drawdown": 0.35,
    "future_downside_volatility": 0.30,
    "future_cvar_95": 0.20,
    "future_illiquidity": 0.15,
}


def max_drawdown(close: np.ndarray) -> float:
    """Return positive maximum drawdown from a price vector."""
    close = np.asarray(close, dtype=float)
    close = close[np.isfinite(close)]
    if close.size < 2:
        return np.nan
    running_max = np.maximum.accumulate(close)
    drawdown = close / (running_max + EPS) - 1.0
    return float(abs(np.nanmin(drawdown)))


def downside_volatility(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    values = returns[np.isfinite(returns) & (returns < 0)]
    if values.size <= 1:
        return 0.0
    return float(np.nanstd(values, ddof=1))


def cvar_95(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    values = returns[np.isfinite(returns)]
    if values.size == 0:
        return np.nan
    cutoff = np.nanquantile(values, 0.05)
    tail = values[values <= cutoff]
    if tail.size == 0:
        return np.nan
    return float(abs(np.nanmean(tail)))


def future_illiquidity(returns: np.ndarray, value: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    value = np.asarray(value, dtype=float)
    metric = np.abs(returns) / (np.abs(value) + EPS)
    if metric.size == 0:
        return np.nan
    return float(np.nanmean(metric))


def compute_future_components(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    horizon_days: int = 126,
    *,
    date_col: str = "decision_date",
) -> pd.DataFrame:
    """Attach future realized risk components to monthly decisions."""
    DAILY_PRICE_SCHEMA.validate(daily)
    require_columns(monthly, [date_col, "ticker"], "monthly")

    d = daily.copy().sort_values(["ticker", "date"])
    d["date"] = pd.to_datetime(d["date"])
    if "return_1d" not in d.columns:
        d["return_1d"] = d.groupby("ticker")["close"].pct_change()
    m = monthly.copy()
    m[date_col] = pd.to_datetime(m[date_col])

    out_rows = []
    daily_by_ticker = {ticker: g.reset_index(drop=True) for ticker, g in d.groupby("ticker")}
    for row in m.itertuples(index=False):
        row_dict = row._asdict()
        ticker = row_dict["ticker"]
        decision_date = row_dict[date_col]
        g = daily_by_ticker.get(ticker)
        if g is None or g.empty:
            comps = {c: np.nan for c in COMPONENTS}
        else:
            pos = int(np.searchsorted(g["date"].to_numpy(dtype="datetime64[ns]"), np.datetime64(decision_date), side="right"))
            future = g.iloc[pos : pos + horizon_days].copy()
            if len(future) < max(20, horizon_days // 3):
                comps = {c: np.nan for c in COMPONENTS}
            else:
                comps = {
                    "future_max_drawdown": max_drawdown(future["close"].to_numpy()),
                    "future_downside_volatility": downside_volatility(future["return_1d"].to_numpy()),
                    "future_cvar_95": cvar_95(future["return_1d"].to_numpy()),
                    "future_illiquidity": future_illiquidity(future["return_1d"].to_numpy(), future["value"].to_numpy()),
                }
        row_dict.update(comps)
        out_rows.append(row_dict)
    return pd.DataFrame(out_rows)


@dataclass
class TargetRanker:
    """Train-only component ranker and class-threshold transformer."""

    weights: dict[str, float] | None = None
    train_values_: dict[str, np.ndarray] | None = None
    thresholds_: tuple[float, float] | None = None

    def fit(self, train_df: pd.DataFrame) -> "TargetRanker":
        weights = self.weights or DEFAULT_WEIGHTS
        require_columns(train_df, COMPONENTS, "train_df")
        self.weights = {k: float(weights[k]) for k in COMPONENTS}
        self.train_values_ = {}
        for component in COMPONENTS:
            values = pd.to_numeric(train_df[component], errors="coerce").dropna().sort_values().to_numpy(dtype=float)
            if len(values) == 0:
                raise ValueError(f"No valid train values for {component}")
            self.train_values_[component] = values
        scores = self._score_frame(train_df)
        self.thresholds_ = (float(np.nanquantile(scores, 1 / 3)), float(np.nanquantile(scores, 2 / 3)))
        return self

    def _percentile_against_train(self, component: str, x: pd.Series) -> np.ndarray:
        if self.train_values_ is None:
            raise RuntimeError("TargetRanker is not fitted")
        train = self.train_values_[component]
        arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        ranks = np.searchsorted(train, arr, side="right") / max(len(train), 1)
        ranks[~np.isfinite(arr)] = np.nan
        return ranks

    def _score_frame(self, df: pd.DataFrame) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("TargetRanker is not fitted")
        score = np.zeros(len(df), dtype=float)
        valid = np.ones(len(df), dtype=bool)
        for component, weight in self.weights.items():
            p = self._percentile_against_train(component, df[component])
            valid &= np.isfinite(p)
            score += weight * np.nan_to_num(p, nan=0.0)
        score[~valid] = np.nan
        return score

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.thresholds_ is None:
            raise RuntimeError("TargetRanker is not fitted")
        result = df.copy()
        result["risk_score"] = self._score_frame(result)
        low_thr, high_thr = self.thresholds_
        labels = np.empty(len(result), dtype=object)
        labels[:] = np.nan
        scores = result["risk_score"].to_numpy(dtype=float)
        valid = np.isfinite(scores)
        labels[valid & (scores <= low_thr)] = "low"
        labels[valid & (scores > low_thr) & (scores <= high_thr)] = "medium"
        labels[valid & (scores > high_thr)] = "high"
        result["risk_class"] = labels
        return result

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

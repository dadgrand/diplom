import pandas as pd

from risk_pipeline.features import compute_daily_features, make_monthly_snapshots


def test_compute_daily_features_and_monthly_snapshot():
    dates = pd.bdate_range("2024-01-01", periods=40)
    rows = []
    for ticker in ["AAA", "BBB"]:
        for i, day in enumerate(dates):
            rows.append({"date": day, "ticker": ticker, "close": 100 + i, "high": 101 + i, "low": 99 + i, "value": 1_000_000 + i, "volume": 1000 + i, "shares_outstanding": 1_000_000})
    df = pd.DataFrame(rows)
    features = compute_daily_features(df)
    monthly = make_monthly_snapshots(features)
    assert "rolling_vol_20d" in monthly.columns
    assert monthly.groupby("ticker").size().sum() >= 2

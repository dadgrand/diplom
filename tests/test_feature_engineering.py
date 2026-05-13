import pandas as pd

from risk_pipeline.feature_engineering import augment_monthly_features


def test_augment_monthly_features_adds_ranks_without_touching_target():
    df = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2024-01-31", "2024-01-31", "2024-02-29"]),
            "ticker": ["A", "B", "A"],
            "sector": ["banks", "oil", "banks"],
            "rolling_vol_20d": [0.01, 0.03, 0.02],
            "downside_vol_60d": [0.008, 0.02, 0.01],
            "beta_60d": [0.8, 1.2, 1.0],
            "amihud_20d": [1e-9, 3e-9, 2e-9],
            "avg_daily_traded_value_20d": [10_000_000, 5_000_000, 7_000_000],
            "spread_proxy": [0.002, 0.006, 0.004],
            "turnover_ratio": [0.001, 0.002, 0.003],
            "log_market_cap": [10.0, 9.0, 11.0],
            "momentum_6m": [0.1, -0.2, 0.0],
            "cbr_key_rate": [10.0, 10.0, 11.0],
            "usd_rub": [90.0, 90.0, 92.0],
            "ofz_slope_10y_2y": [0.1, 0.1, -0.2],
            "imoex_realized_vol_20d": [0.02, 0.02, 0.03],
            "average_market_correlation_60d": [0.3, 0.3, 0.5],
            "future_max_drawdown": [0.1, 0.2, 0.15],
        }
    )
    out, added = augment_monthly_features(df, rank_features=["rolling_vol_20d", "amihud_20d"])
    assert "liquidity_stress" in out.columns
    assert "rolling_vol_20d_cs_rank" in out.columns
    assert "future_max_drawdown" not in added
    assert out.loc[out["ticker"] == "B", "rolling_vol_20d_cs_rank"].iloc[0] == 1.0

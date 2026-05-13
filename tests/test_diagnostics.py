import pandas as pd

from risk_pipeline.diagnostics import data_quality_report, population_stability_index


def test_population_stability_index_detects_shift():
    train = pd.Series(range(100))
    shifted = pd.Series(range(100, 200))
    assert population_stability_index(train, shifted, bins=5) > 0.5


def test_data_quality_report_flags_duplicates():
    df = pd.DataFrame(
        {
            "decision_date": ["2024-01-31", "2024-01-31"],
            "ticker": ["AAA", "AAA"],
            "future_max_drawdown": [0.1, 0.2],
            "future_downside_volatility": [0.01, 0.02],
            "future_cvar_95": [0.03, 0.04],
            "future_illiquidity": [1e-10, 2e-10],
        }
    )
    report = data_quality_report(df)
    duplicate_row = report[report["check"] == "no_duplicate_keys"].iloc[0]
    assert duplicate_row["passed"] == False

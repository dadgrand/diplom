import pandas as pd

from risk_pipeline.config import SplitConfig
from risk_pipeline.splits import temporal_train_val_test, walk_forward_splits


def test_temporal_split_counts():
    df = pd.DataFrame({"decision_date": pd.date_range("2024-01-31", periods=12, freq="ME"), "x": range(12)})
    split = temporal_train_val_test(
        df,
        SplitConfig(
            train_start="2024-01-01",
            train_end="2024-06-30",
            validation_start="2024-07-01",
            validation_end="2024-09-30",
            test_start="2024-10-01",
            test_end="2024-12-31",
        ),
    )
    assert len(split.train) == 6
    assert len(split.validation) == 3
    assert len(split.test) == 3


def test_walk_forward_non_empty():
    df = pd.DataFrame({"decision_date": pd.date_range("2023-01-31", periods=18, freq="ME"), "x": range(18)})
    splits = list(walk_forward_splits(df, initial_months=6, test_months=3, step_months=3))
    assert len(splits) >= 3

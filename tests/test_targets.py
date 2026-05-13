import numpy as np
import pandas as pd

from risk_pipeline.targets import TargetRanker, cvar_95, downside_volatility, max_drawdown


def test_max_drawdown_positive_value():
    prices = np.array([100, 120, 90, 95, 80, 130], dtype=float)
    assert round(max_drawdown(prices), 4) == 0.3333


def test_downside_volatility_uses_only_negative_returns():
    returns = np.array([0.02, -0.01, -0.03, 0.04, -0.02])
    assert downside_volatility(returns) > 0


def test_cvar_is_positive_tail_loss():
    returns = np.array([0.01, -0.02, -0.10, 0.03, -0.01])
    assert cvar_95(returns) > 0


def test_target_ranker_train_only_thresholds():
    train = pd.DataFrame(
        {
            "future_max_drawdown": [0.01, 0.02, 0.20, 0.30, 0.40, 0.50],
            "future_downside_volatility": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "future_cvar_95": [0.01, 0.02, 0.04, 0.08, 0.10, 0.12],
            "future_illiquidity": [1, 2, 3, 4, 5, 6],
        }
    )
    ranker = TargetRanker().fit(train)
    labeled = ranker.transform(train)
    assert set(labeled["risk_class"]) == {"low", "medium", "high"}

import numpy as np

from risk_pipeline.models import ProbabilityPolicy


def test_probability_policy_prioritizes_high_threshold():
    policy = ProbabilityPolicy(classes=["low", "medium", "high"])
    proba = np.array([[0.60, 0.30, 0.10], [0.20, 0.35, 0.45], [0.30, 0.40, 0.30]])
    pred = policy.predict_from_proba(proba, low_threshold=0.55, high_threshold=0.40)
    assert pred.tolist() == ["low", "high", "medium"]

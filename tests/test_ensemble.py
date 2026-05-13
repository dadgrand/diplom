import numpy as np

from risk_pipeline.ensemble import WeightedProbabilityEnsemble, align_proba


def test_align_proba_canonical_order():
    proba = np.array([[0.2, 0.7, 0.1]])
    aligned = align_proba(["high", "low", "medium"], proba, ["low", "medium", "high"])
    assert np.allclose(aligned, [[0.7, 0.1, 0.2]])


def test_weighted_probability_ensemble_weights_sum_to_one():
    y = np.array(["low", "medium", "high", "high"])
    a = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
    b = np.ones_like(a) / 3
    ens = WeightedProbabilityEnsemble(weight_step=0.5).fit(y, {"a": a, "b": b})
    assert abs(sum(ens.weights_.values()) - 1.0) < 1e-12
    assert ens.weights_["a"] >= ens.weights_["b"]

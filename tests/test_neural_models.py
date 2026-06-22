import numpy as np
import pandas as pd
import pytest

from risk_pipeline.models import AutoencoderFactors, TorchMLPClassifier, torch


pytestmark = pytest.mark.skipif(torch is None, reason="torch is not installed")


def test_torch_mlp_classifier_predicts_probabilities():
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(90, 6)), columns=[f"f{i}" for i in range(6)])
    score = x["f0"] + 0.7 * x["f1"] - 0.4 * x["f2"]
    y = np.where(score > 0.65, "high", np.where(score < -0.65, "low", "medium"))

    model = TorchMLPClassifier(hidden_dims=(16,), epochs=18, patience=5, random_state=11).fit(x, y)
    proba = model.predict_proba(x.iloc[:7])

    assert model.fitted_with_ == "torch_mlp"
    assert proba.shape == (7, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert model.feature_importances_.shape == (6,)


def test_autoencoder_uses_torch_backend_when_available():
    rng = np.random.default_rng(13)
    df = pd.DataFrame(rng.normal(size=(40, 5)), columns=[f"f{i}" for i in range(5)])

    ae = AutoencoderFactors(numeric_features=list(df.columns), latent_dim=2, hidden_dim=4, epochs=6, random_state=13).fit(df)
    latent = ae.transform(df.iloc[:4])

    assert ae.fitted_with_ == "torch_autoencoder"
    assert latent.shape == (4, 2)


def test_autoencoder_linear_pca_init_is_real_torch_backend():
    rng = np.random.default_rng(19)
    df = pd.DataFrame(rng.normal(size=(44, 6)), columns=[f"f{i}" for i in range(6)])

    ae = AutoencoderFactors(numeric_features=list(df.columns), latent_dim=3, hidden_dim=0, epochs=6, random_state=19).fit(df)
    latent = ae.transform(df.iloc[:5])

    assert ae.fitted_with_ == "torch_linear_autoencoder_pca_init"
    assert type(ae.net_).__name__ == "_AutoencoderNet"
    assert latent.shape == (5, 3)
    assert np.isfinite(ae.reconstruction_loss_)

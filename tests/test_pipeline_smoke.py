from dataclasses import replace

from risk_pipeline.config import load_config
from risk_pipeline.pipeline import run_modeling_pipeline
from risk_pipeline.synthetic import make_synthetic_monthly_panel


def test_run_modeling_pipeline_smoke(tmp_path):
    cfg = load_config(None)
    cfg = replace(
        cfg,
        model=replace(cfg.model, n_estimators=8, autoencoder_epochs=3, candidates=["rf", "extra"], n_jobs=1),
        diagnostics=replace(cfg.diagnostics, save_model_package=False),
    )
    df = make_synthetic_monthly_panel(n_tickers=8, random_state=cfg.random_state)
    result = run_modeling_pipeline(df, cfg, artifact_dir=tmp_path)
    assert result.predictions.shape[0] > 0
    assert "sector_overlay" in result.metrics["test"]
    assert (tmp_path / "feature_drift_report.csv").exists()

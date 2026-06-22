from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .data_sources import load_local_csv, save_frame
from .financial_reports import build_financial_report_features, discover_reports_for_sources, download_report_registry
from .logging_utils import setup_logging
from .pipeline import run_modeling_pipeline
from .predict import load_model_package, predict_model_ready
from .real_data_pipeline import build_model_ready_panel_from_paths
from .synthetic import make_synthetic_monthly_panel
from .report_ablation import run_report_layer_ablation


def run_demo(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cfg = replace(cfg, model=replace(cfg.model, n_estimators=args.n_estimators, autoencoder_epochs=args.autoencoder_epochs, n_jobs=1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = make_synthetic_monthly_panel(n_tickers=args.n_tickers, random_state=cfg.random_state)
    save_frame(df, out / "synthetic_monthly_modeling.parquet")
    result = run_modeling_pipeline(df, cfg, artifact_dir=out)
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2, default=str))


def run_model_ready(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_local_csv(args.input)
    result = run_modeling_pipeline(df, cfg, artifact_dir=out)
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2, default=str))


def build_panel(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    out_path = Path(args.output)
    panel = build_model_ready_panel_from_paths(
        daily_prices_path=args.daily_prices,
        macro_path=args.macro,
        fundamentals_path=args.fundamentals,
        universe_path=args.universe,
        market_path=args.market,
        report_features_path=args.report_features,
        reports_registry_path=args.reports_registry,
        reports_dir=args.reports_dir,
        report_features_output_path=args.report_features_output,
        report_coverage_output_path=args.report_coverage_output,
        output_path=out_path,
        horizon_days=cfg.target.horizon_days,
    )
    print(json.dumps({"rows": len(panel), "columns": list(panel.columns), "output": str(out_path)}, ensure_ascii=False, indent=2, default=str))


def collect_reports(args: argparse.Namespace) -> None:
    discovered = discover_reports_for_sources(
        args.sources,
        start=args.start,
        end=args.end,
        document_types=args.document_types,
    )
    if args.download:
        discovered = download_report_registry(discovered, out_dir=args.reports_dir, registry_output=args.output)
    else:
        save_frame(discovered, args.output)
    print(json.dumps({"rows": len(discovered), "output": str(args.output), "download": bool(args.download)}, ensure_ascii=False, indent=2, default=str))


def build_report_features(args: argparse.Namespace) -> None:
    features = build_financial_report_features(args.registry, reports_dir=args.reports_dir, registry_path=args.registry, include_evidence=not args.no_evidence)
    save_frame(features, args.output)
    print(json.dumps({"rows": len(features), "columns": list(features.columns), "output": str(args.output)}, ensure_ascii=False, indent=2, default=str))


def run_report_demo(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cfg = replace(cfg, model=replace(cfg.model, n_estimators=args.n_estimators, autoencoder_epochs=args.autoencoder_epochs, n_jobs=1))
    summary = run_report_layer_ablation(cfg, out_dir=args.out, n_tickers=args.n_tickers, random_state=cfg.random_state)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def predict(args: argparse.Namespace) -> None:
    model_package = load_model_package(args.model_package)
    df = load_local_csv(args.input)
    pred = predict_model_ready(df, model_package)
    save_frame(pred, args.output)
    print(json.dumps({"rows": len(pred), "output": str(args.output)}, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Risk classification pipeline")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("run-demo", help="Run end-to-end demo on synthetic model-ready data")
    demo.add_argument("--out", default="results/demo_run", help="Output directory")
    demo.add_argument("--n-tickers", type=int, default=30, help="Number of synthetic tickers")
    demo.add_argument("--n-estimators", type=int, default=25, help="Tree count for quick demo")
    demo.add_argument("--autoencoder-epochs", type=int, default=8, help="Autoencoder epochs for quick demo")
    demo.set_defaults(func=run_demo)

    model_ready = sub.add_parser("run-model-ready", help="Train/evaluate on a prepared monthly model-ready CSV/parquet")
    model_ready.add_argument("--input", required=True, help="Path to model-ready monthly panel")
    model_ready.add_argument("--out", default="results/run_model_ready", help="Output directory")
    model_ready.set_defaults(func=run_model_ready)

    build = sub.add_parser("build-panel", help="Build model-ready monthly panel from raw daily/macro/fundamental CSVs")
    build.add_argument("--daily-prices", required=True, help="daily prices CSV/parquet")
    build.add_argument("--macro", required=True, help="macro CSV/parquet")
    build.add_argument("--fundamentals", default=None, help="fundamentals CSV/parquet")
    build.add_argument("--universe", default=None, help="universe CSV/parquet")
    build.add_argument("--market", default=None, help="market benchmark CSV/parquet")
    build.add_argument("--report-features", default=None, help="precomputed financial report features CSV/parquet")
    build.add_argument("--reports-registry", default=None, help="registry of local financial reports to parse and merge point-in-time")
    build.add_argument("--reports-dir", default=None, help="base directory for local report files")
    build.add_argument("--report-features-output", default=None, help="optional output path for extracted report features")
    build.add_argument("--report-coverage-output", default=None, help="optional CSV diagnostics for report coverage by ticker")
    build.add_argument("--output", required=True, help="Output model-ready parquet/csv")
    build.set_defaults(func=build_panel)


    collect = sub.add_parser("collect-reports", help="Discover and optionally download issuer reports from source registry")
    collect.add_argument("--sources", required=True, help="CSV/parquet with ticker, company_name, e_disclosure_id, issuer_url")
    collect.add_argument("--start", default=None, help="min publish date, e.g. 2022-01-01")
    collect.add_argument("--end", default=None, help="max publish date, e.g. 2025-08-31")
    collect.add_argument("--document-types", nargs="+", type=int, default=[4, 2], help="e-disclosure document types: 4=IFRS consolidated, 2=annual reports")
    collect.add_argument("--reports-dir", default="data/raw/reports", help="where downloaded report files are stored")
    collect.add_argument("--output", default="data/raw/report_registry.csv", help="output report registry")
    collect.add_argument("--download", action="store_true", help="download direct file URLs discovered by the crawler")
    collect.set_defaults(func=collect_reports)

    rep = sub.add_parser("build-report-features", help="Extract numeric and narrative features from financial reports")
    rep.add_argument("--registry", required=True, help="report registry with local_path or report_text")
    rep.add_argument("--reports-dir", default=None, help="base directory for local report files")
    rep.add_argument("--output", required=True, help="output financial_report_features CSV/parquet")
    rep.add_argument("--no-evidence", action="store_true", help="drop metric evidence snippets from output")
    rep.set_defaults(func=build_report_features)

    rep_demo = sub.add_parser("run-report-demo", help="Run synthetic ablation: market/fundamental baseline vs report-enhanced pipeline")
    rep_demo.add_argument("--out", default="results/report_layer_demo", help="Output directory")
    rep_demo.add_argument("--n-tickers", type=int, default=24, help="Number of synthetic tickers")
    rep_demo.add_argument("--n-estimators", type=int, default=30, help="Tree count for quick demo")
    rep_demo.add_argument("--autoencoder-epochs", type=int, default=8, help="Autoencoder epochs for quick demo")
    rep_demo.set_defaults(func=run_report_demo)

    pred = sub.add_parser("predict", help="Predict risk classes using a saved model_package.joblib")
    pred.add_argument("--model-package", required=True, help="Path to model_package.joblib")
    pred.add_argument("--input", required=True, help="Model-ready current observations")
    pred.add_argument("--output", required=True, help="Output predictions CSV/parquet")
    pred.set_defaults(func=predict)

    args = parser.parse_args(argv)
    setup_logging()
    args.func(args)


if __name__ == "__main__":
    main()

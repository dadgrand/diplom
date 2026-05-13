# Financial report layer: Aladdin-style issuer intelligence

This module adds a deterministic financial-report layer to the point-in-time risk pipeline. The goal is to use issuer reporting as an additional source of information without turning the model into an unverifiable LLM system.

## What the layer does

1. **Discovers report sources** for a ticker universe from `data/report_sources_ru_bluechips.csv`.
2. **Downloads or accepts local reports** through a report registry.
3. **Extracts report text** from `.txt`, `.md`, `.html`, `.pdf`, `.xlsx` and `.zip` containers.
   Direct e-disclosure downloads are audited with `download_status`, `content_type`,
   `detected_extension` and `file_sha256`; `FileLoad.ashx` links are saved with the
   actual detected extension instead of the handler extension.
4. **Extracts numeric metrics** using bilingual Russian/English dictionaries:
   - revenue / net interest income;
   - EBITDA / OIBDA;
   - net profit;
   - operating cash flow;
   - free cash flow;
   - capex;
   - total, short-term and long-term debt;
   - cash and equivalents;
   - interest expense / finance costs;
   - assets, equity, dividends and selected bank metrics.
5. **Extracts narrative risk signals** as deterministic counts and flags:
   - sanctions / market-access restrictions;
   - FX and currency risk;
   - liquidity and refinancing risk;
   - covenant risk;
   - impairment;
   - litigation;
   - going-concern language;
   - dividend pressure;
   - tax pressure;
   - capex pressure;
   - demand pressure;
   - auditor emphasis / qualified opinion;
   - related-party risk.
6. **Builds derived financial-report features**:
   - `report_net_debt_to_ebitda`;
   - `report_interest_coverage`;
   - `report_ebitda_margin`;
   - `report_free_cf_margin`;
   - `report_debt_to_assets`;
   - `report_financial_pressure`;
   - `report_integrated_stress`;
   - `fundamental_report_gap`.
7. **Merges reports point-in-time** by `publish_date <= decision_date`. A report never influences observations before it was published.
8. **Selects the report layer by validation**, not by assumption. The training pipeline fits `enriched_v2_without_reports` and `enriched_v2_with_reports`; report features are used only when validation score improves by the configured minimum gain.

## Source registry

The starter registry covers a liquid Russian equity universe:

```text
data/report_sources_ru_bluechips.csv
```

The important columns are:

| column | meaning |
| --- | --- |
| `ticker` | MOEX ticker used by the model |
| `company_name` | issuer name |
| `sector` | broad sector |
| `e_disclosure_id` | Interfax e-disclosure issuer id |
| `e_disclosure_ifrs_url` | consolidated financial statements page (`type=4`) |
| `e_disclosure_annual_url` | annual reports page (`type=2`) |
| `issuer_url` | issuer investor-relations fallback |

For a production run, verify ticker mappings, share-class selection and issuer reorganizations before downloading. This is especially important for legacy tickers, delistings and redomiciled companies.

## Build coverage for the full modeling sample

The default model config uses decisions from 2023-07-01 to 2025-08-31. For safe point-in-time coverage, collect reports starting earlier than the first decision date, for example from 2022-01-01:

```bash
python -m risk_pipeline.cli collect-reports \
  --sources data/report_sources_ru_bluechips.csv \
  --start 2022-01-01 \
  --end 2025-08-31 \
  --document-types 4 2 \
  --download \
  --reports-dir data/raw/reports \
  --output data/raw/report_registry.csv
```

If a website blocks automated downloading, manually place the downloaded PDF/ZIP/XLSX files in `data/raw/reports/<TICKER>/` and fill `local_path` in `data/raw/report_registry.csv`. The extraction step does not require internet access once documents are local.

The registry is intentionally failure-tolerant. If e-disclosure blocks or times
out, `collect-reports` keeps a `discovery_status=failed` row with
`discovery_error` and also keeps the issuer investor-relations page as
`discovery_status=manual_fallback` when `issuer_url` is available. This prevents
one blocked issuer from stopping the full collection run.

## Extract features

```bash
python -m risk_pipeline.cli build-report-features \
  --registry data/raw/report_registry.csv \
  --reports-dir data/raw/reports \
  --output data/processed/financial_report_features.csv
```

The output contains numeric features, narrative counts, missing indicators and optional evidence snippets. Evidence snippets are included to audit why a number was extracted.
It also includes `parse_status` and `parse_error`, so failed/empty documents stay
visible in coverage diagnostics instead of disappearing from the sample.

## Merge into the model-ready point-in-time panel

Using already extracted report features:

```bash
python -m risk_pipeline.cli build-panel \
  --daily-prices data/raw/daily_prices.csv \
  --macro data/raw/macro.csv \
  --fundamentals data/raw/fundamentals.csv \
  --universe data/universe.csv \
  --market data/raw/imoex.csv \
  --report-features data/processed/financial_report_features.csv \
  --report-coverage-output data/processed/report_coverage.csv \
  --output data/processed/monthly_model_ready.parquet
```

Or parse reports directly during panel assembly:

```bash
python -m risk_pipeline.cli build-panel \
  --daily-prices data/raw/daily_prices.csv \
  --macro data/raw/macro.csv \
  --fundamentals data/raw/fundamentals.csv \
  --universe data/universe.csv \
  --reports-registry data/raw/report_registry.csv \
  --reports-dir data/raw/reports \
  --report-features-output data/processed/financial_report_features.csv \
  --report-coverage-output data/processed/report_coverage.csv \
  --output data/processed/monthly_model_ready.parquet
```

## Report-layer ablation

A compact synthetic ablation is included:

```bash
python -m risk_pipeline.cli run-report-demo \
  --out results/report_layer_demo \
  --n-tickers 12 \
  --n-estimators 12 \
  --autoencoder-epochs 3
```

The demo saves:

```text
results/report_layer_demo/report_layer_ablation.csv
results/report_layer_demo/report_coverage.csv
results/report_layer_demo/financial_report_features.csv
results/report_layer_demo/baseline_run/metrics.json
results/report_layer_demo/report_enhanced_run/metrics.json
```

In the included run, the report layer was selected on validation and improved synthetic test macro-F1 from `0.5103` to `0.5250`, while high-risk precision rose from `0.8140` to `0.8462`. The gain is intentionally treated as an ablation result, not as a guarantee for real data.

## Leakage-control checklist

- Use `publish_date`, not `report_period_end`, for model availability.
- Keep `report_period_end` only for economic interpretation and year-over-year deltas.
- If `publish_date` is missing, the report is not merged into model observations.
- Stale reports are blanked when `report_lag_days` exceeds the configured maximum.
- Extracted metrics are accompanied by missing flags and evidence snippets.
- The model compares report vs no-report variants on validation and rejects the report layer if it does not help.

## Why this is close to the useful part of an Aladdin-like concept

The useful idea is not a black-box advisor. It is an auditable risk platform that separates:

- data ingestion;
- point-in-time feature generation;
- deterministic risk metrics;
- scenario/diagnostic thinking;
- validation-based model selection;
- human-readable reporting.

The financial-report layer follows the same principle: the model receives structured numbers and explicit text-risk counts, while narrative interpretation can be done later from already computed artifacts.

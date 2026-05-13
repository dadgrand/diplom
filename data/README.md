# Data contract

## `universe.csv`

| column | required | meaning |
| --- | --- | --- |
| `ticker` | yes | торговый код акции |
| `sector` | recommended | укрупнённая секторная группа |
| `include_flag` | optional | 1, если бумага включается в universe |

## `daily_prices.csv`

Минимальные колонки: `date`, `ticker`, `close`, `value`.

Желательные колонки: `open`, `high`, `low`, `volume`, `num_trades`, `wap`, `shares_outstanding`, `market_return`.

Если отдельный benchmark не передан через `--market`, рыночная доходность оценивается как средняя доходность бумаг в universe на дату.

## `market.csv`

Минимальные колонки: `date`, `market_close` или `close`.

Используется для расчёта `market_return` и `beta_60d`.

## `macro.csv`

Обязательные колонки:

- `date`
- `cbr_key_rate`
- `usd_rub`
- `ofz_slope_10y_2y`
- `imoex_realized_vol_20d`
- `average_market_correlation_60d`

## `fundamentals.csv`

Обязательные колонки: `ticker`, `report_date`, `publish_date`.

Рекомендуемые колонки:

- `revenue`
- `ebitda`
- `net_debt`
- `interest_expense`
- `operating_cash_flow`
- `free_cash_flow`
- `book_equity`
- `shares_outstanding`

Важно: `publish_date` используется для point-in-time присоединения. Если вместо него использовать `report_date`, модель получит информацию, которая на дату решения ещё не была раскрыта.

## `monthly_model_ready.csv/parquet`

После `build-panel` получается месячная таблица с:

- `decision_date`, `ticker`, `sector`;
- текущими признаками риска;
- target components: `future_max_drawdown`, `future_downside_volatility`, `future_cvar_95`, `future_illiquidity`.

Эта таблица подаётся в `run-model-ready`.

## Financial reports

### `report_sources_ru_bluechips.csv`

Starter source registry for issuer-report collection. Required columns for the collector:

| column | required | meaning |
| --- | --- | --- |
| `ticker` | yes | ticker used in the model panel |
| `company_name` | recommended | issuer name |
| `e_disclosure_id` | recommended | Interfax e-disclosure issuer identifier |
| `issuer_url` | optional | investor-relations fallback page |

### `report_registry.csv`

Created by `collect-reports` or filled manually. Required for feature extraction:

| column | required | meaning |
| --- | --- | --- |
| `ticker` | yes | ticker used in the model panel |
| `report_period_end` | recommended | economic period end of the report |
| `publish_date` | yes for PIT merge | date when the report became available |
| `report_type` | recommended | annual report, IFRS statement, interim statement, MD&A, etc. |
| `source_url` | optional | original URL |
| `local_path` | optional | local PDF/XLSX/TXT/ZIP path relative to `--reports-dir` |
| `report_text` | optional | pre-extracted text, useful for tests/manual ingestion |
| `discovery_status` | optional | `ok`, `failed` or `manual_fallback` after source discovery |
| `discovery_error` | optional | crawler error text when a source blocks or times out |
| `download_status` | optional | downloader result such as `downloaded:200`, `existing`, `failed:...` or `no_direct_file_url` |
| `content_type` | optional | HTTP `Content-Type` observed during download |
| `detected_extension` | optional | extension inferred from `Content-Disposition`, `Content-Type`, magic bytes or URL |
| `file_sha256` | optional | SHA-256 of the downloaded/local document |

`publish_date`, not `report_period_end`, is used for point-in-time merging.

### `financial_report_features.csv`

Created by `build-report-features`. In addition to extracted numeric/text features,
it carries audit columns:

| column | meaning |
| --- | --- |
| `parse_status` | `parsed_file`, `parsed_embedded_text`, `empty_text`, `extract_failed`, `parse_failed` or `missing_document` |
| `parse_error` | compact extraction/parsing error text |
| `report_extraction_quality` | heuristic quality score from available metrics, ratios and text length |

идея

Модель принимает решение на месячном срезе по бумаге. На дату решения ей доступны только текущие и уже раскрытые данные: рыночные, ликвидностные, макроэкономические, секторные и фундаментальные признаки. Цель строится по будущему окну 126 торговых дней через четыре компоненты риска:

```text
RiskScore = 0.35 * future_max_drawdown
          + 0.30 * future_downside_volatility
          + 0.20 * future_cvar_95
          + 0.15 * future_illiquidity
```

Пороги классов и процентильные ранги компонент вычисляются только на train-периоде.

Основные команды:

```bash
python -m risk_pipeline.cli collect-reports \
  --sources data/report_sources_ru_bluechips.csv \
  --start 2022-01-01 --end 2025-08-31 \
  --document-types 4 2 \
  --download \
  --reports-dir data/raw/reports \
  --output data/raw/report_registry.csv

python -m risk_pipeline.cli build-report-features \
  --registry data/raw/report_registry.csv \
  --reports-dir data/raw/reports \
  --output data/processed/financial_report_features.csv
```

Запуск на готовой model-ready панели

Панель должна содержать минимум:

- `decision_date`, `ticker`, `sector`;
- признаки из `configs/config.example.yaml`;
- target components: `future_max_drawdown`, `future_downside_volatility`, `future_cvar_95`, `future_illiquidity`.

```bash
python -m risk_pipeline.cli --config configs/config.example.yaml \
  run-model-ready \
  --input data/processed/monthly_model_ready.parquet \
  --out results/real_run
```

Сборка панели из raw CSV/parquet

```bash
python -m risk_pipeline.cli --config configs/config.example.yaml \
  build-panel \
  --daily-prices data/raw/daily_prices.csv \
  --macro data/raw/macro.csv \
  --fundamentals data/raw/fundamentals.csv \
  --universe data/universe.csv \
  --market data/raw/imoex.csv \
  --output data/processed/monthly_model_ready.parquet
```

`fundamentals.csv` обязан содержать `ticker`, `report_date`, `publish_date`. Присоединение выполняется по `publish_date`, а не по `report_date`.

Инференс после обучения

```bash
python -m risk_pipeline.cli predict \
  --model-package results/real_run/model_package.joblib \
  --input data/processed/current_month_observations.csv \
  --output results/real_run/current_predictions.csv
```

Для инференса target components не нужны.

# risk_pipeline: production-grade research pipeline for equity investment-risk classification

Пакет реализует воспроизводимый point-in-time пайплайн автоматической классификации инвестиционного риска акций по трём классам: `low`, `medium`, `high`. Архитектура ориентирована на исследовательскую проверку и дальнейшее прикладное использование: от сборки месячной панели до обучения ансамблей, walk-forward проверки, диагностики дрейфа, сохранения model package и инференса.

## Ключевая идея

Модель принимает решение на месячном срезе по бумаге. На дату решения ей доступны только текущие и уже раскрытые данные: рыночные, ликвидностные, макроэкономические, секторные и фундаментальные признаки. Цель строится по будущему окну 126 торговых дней через четыре компоненты риска:

```text
RiskScore = 0.35 * future_max_drawdown
          + 0.30 * future_downside_volatility
          + 0.20 * future_cvar_95
          + 0.15 * future_illiquidity
```

Пороги классов и процентильные ранги компонент вычисляются только на train-периоде.

## Что усилено в этой версии

- Добавлен real-data контур `build-panel`: дневные цены → rolling features → месячные срезы → macro as-of merge → fundamentals point-in-time merge → target components.
- Добавлен safe feature engineering: экономически интерпретируемые взаимодействия, cross-sectional ranks и sector z-scores без использования будущих target-полей.
- Исправлена train-only логика в regime clustering и autoencoder factors: медианы/скейлеры теперь обучаются только на train и не пересчитываются на validation/test/inference.
- Вместо одной модели используется candidate ensemble: Random Forest, Extra Trees и Gradient Boosting с validation-optimized soft voting.
- Добавлена probability calibration через power transform и probability-policy thresholds для консервативного управления классами `low` и `high`.
- Сохранён гибридный слой: regime clustering + enriched tree ensemble + autoencoder latent factors + sector expert overlay.
- Добавлены drift diagnostics: PSI, missing-rate сдвиги, train/test mean shift.
- Добавлены data quality checks: дубликаты ключей, наличие target components, корректность дат, контроль порядка split.
- Добавлены walk-forward отчёты с train/validation/test окнами внутри каждого фолда.
- Добавлен `model_package.joblib` и команда `predict` для инференса на новых наблюдениях.
- Расширен набор unit-тестов.


## Aladdin-style financial-report layer

В этой версии добавлен отдельный слой анализа финансовых отчетов эмитентов. Он не использует LLM для расчётов: отчёты превращаются в воспроизводимые признаки через словари, регулярные правила, evidence-snippets и point-in-time merge по `publish_date`.

Что добавлено:

- `data/report_sources_ru_bluechips.csv` — стартовый реестр источников отчетности по ликвидным российским эмитентам;
- `collect-reports` — поиск раскрытий на e-disclosure и подготовка report registry;
- `build-report-features` — извлечение показателей и смысловых risk-signals из PDF/XLSX/TXT/ZIP;
- `report_available`, `report_lag_days`, `report_financial_pressure`, `report_integrated_stress`, `fundamental_report_gap` и десятки дополнительных report-признаков;
- validation gate: модель сравнивает `enriched_v2_without_reports` и `enriched_v2_with_reports` и включает report-слой только при улучшении на validation;
- `results/report_layer_demo` — компактная ablation-проверка report-слоя.

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

Подробности: `docs/FINANCIAL_REPORT_LAYER.md` и `docs/ALADDIN_CONCEPT_INTEGRATION.md`.

## Быстрый запуск демо

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m risk_pipeline.cli run-demo --out results/demo_run
pytest -q
```

Демо использует синтетическую месячную панель и запускает ту же архитектуру, что и реальный pipeline, но с уменьшенным количеством деревьев.

## Запуск на готовой model-ready панели

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

## Сборка панели из raw CSV/parquet

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

## Инференс после обучения

```bash
python -m risk_pipeline.cli predict \
  --model-package results/real_run/model_package.joblib \
  --input data/processed/current_month_observations.csv \
  --output results/real_run/current_predictions.csv
```

Для инференса target components не нужны.

## Основные артефакты запуска

| Файл | Назначение |
| --- | --- |
| `metrics.json` | основные метрики, thresholds, calibration, walk-forward, drift summary |
| `predictions.csv` | test-предсказания, вероятности классов, target components |
| `feature_importance.csv` | важности признаков enriched/regime/ann branches |
| `model_leaderboard.csv` | качество кандидатов и ensemble на validation |
| `sector_overlay_report.csv` | решение о выборе секторных экспертов |
| `feature_drift_report.csv` | PSI и missing-rate diagnostics |
| `walk_forward_report.csv` | устойчивость на expanding-window splits |
| `data_quality_report.csv` | базовые проверки качества входной панели |
| `run_manifest.json` | fingerprint данных, config и контрольные суммы артефактов |
| `model_package.joblib` | обученный пакет для инференса |

## Структура проекта

```text
risk_pipeline/
  cli.py                    # CLI: demo, run-model-ready, build-panel, predict
  config.py                 # typed configuration
  data_sources.py           # local IO + MOEX/CBR adapters
  diagnostics.py            # PSI, manifest, data quality checks
  ensemble.py               # candidate models, soft voting, calibration
  evaluation.py             # metrics, confusion, monotonicity
  feature_engineering.py    # safe domain features and cross-sectional features
  financial_reports.py      # issuer-report discovery, extraction and point-in-time merge
  features.py               # daily and monthly feature layer
  models.py                 # RF/Extra/GB, regime clustering, autoencoder, overlay
  pipeline.py               # end-to-end training/evaluation orchestration
  predict.py                # saved-package inference
  preprocessing.py          # train-only imputation, clipping, signed-log, one-hot
  real_data_pipeline.py     # raw data -> model-ready panel
  report_ablation.py        # controlled ablation for the financial-report layer
  targets.py                # future risk components and train-only target ranker
  splits.py                 # temporal and walk-forward splits
  synthetic.py              # synthetic panel for smoke tests and demonstration
```

## Tests

```bash
pytest -q
```

Покрыты ключевые элементы: future target components, train-only target ranker, temporal splits, feature engineering, probability policy, weighted ensemble, drift diagnostics и end-to-end smoke test.

## Design notes

1. **No random split for finance.** Все проверки построены на temporal split и walk-forward windows.
2. **No future leakage.** Target строится по будущему окну, но target-поля не используются в feature engineering.
3. **No blind complexity.** ANN branch используется как источник латентных факторов и секторный эксперт, а не как безусловная замена интерпретируемого ядра.
4. **No silent drift.** PSI и missing-rate сдвиги сохраняются как отдельный отчёт.
5. **No untraceable run.** Каждый запуск получает `run_manifest.json` с fingerprint входной панели и checksums артефактов.

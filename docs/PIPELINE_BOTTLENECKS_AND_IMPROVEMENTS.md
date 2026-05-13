# Bottleneck analysis and implemented improvements

## 1. Leakage risk in financial data

**Bottleneck:** при финансовом моделировании даже небольшая утечка будущего через фундаментальные данные, target thresholds, imputers или scaling делает метрики нереалистичными.

**Решение:**

- `merge_fundamentals_pit` присоединяет фундаментальные данные только по `publish_date`.
- `TargetRanker` обучает percentile ranks и thresholds только на train.
- `FinancialPreprocessor`, `RegimeClusterer`, `AutoencoderFactors` хранят train-only медианы/скейлеры.
- `data_quality_report` и `validate_temporal_split` фиксируют базовые нарушения.

## 2. Underpowered single-model baseline

**Bottleneck:** один Random Forest может быть устойчивым, но не всегда достаточно гибким для разных рыночных режимов и секторных профилей.

**Решение:**

- `fit_candidate_bundle` обучает Random Forest, Extra Trees и Gradient Boosting.
- `WeightedProbabilityEnsemble` подбирает soft-voting веса на validation.
- `PowerProbabilityCalibrator` корректирует уверенность вероятностей.
- `ProbabilityPolicy` подбирает thresholds для `low` и `high`, учитывая macro-F1 и bonus к high-recall.

## 3. Regime mixing

**Bottleneck:** одна и та же бумага может иметь разный риск в спокойном и стрессовом рынке; без режима модель смешивает разные условные распределения.

**Решение:**

- `RegimeClusterer` строит train-only макрорыночные кластеры по ставке, USD/RUB, ОФЗ, волатильности IMOEX и средней корреляции.
- Regime labels добавляются как категориальный признак в regime/enriched/ANN ветви.

## 4. Heavy-tailed financial features

**Bottleneck:** обороты, Amihud, долговые метрики и ликвидность имеют тяжёлые хвосты, что ухудшает стабильность моделей и autoencoder.

**Решение:**

- `FinancialPreprocessor` применяет median imputation, train-quantile clipping, signed-log и scaling.
- `feature_engineering.py` создаёт устойчивые interaction features: `liquidity_stress`, `debt_service_stress`, `macro_pressure`, `fundamental_fragility`.

## 5. Cross-sectional market drift

**Bottleneck:** абсолютные уровни признаков дрейфуют со временем; модель может переобучаться на уровень периода, а не на относительное положение бумаги.

**Решение:**

- Добавлены monthly cross-sectional ranks: `<feature>_cs_rank`.
- Добавлены sector-normalized z-scores: `<feature>_sector_z`.

## 6. Neural branch interpretability and usefulness

**Bottleneck:** end-to-end нейросеть на малой финансовой панели легко переобучается и плохо интерпретируется.

**Решение:**

- `AutoencoderFactors` используется не как финальный классификатор, а как компактный extractor латентных факторов.
- ANN branch сравнивается с enriched tree branch; `SectorExpertOverlay` включает ANN-эксперта только при validation gain.

## 7. Model selection instability

**Bottleneck:** выбор модели по одному validation/test периоду может быть случайным.

**Решение:**

- `walk_forward_report.csv` строит expanding-window проверки.
- `model_leaderboard.csv` показывает качество кандидатов и ensemble на validation.
- `feature_drift_report.csv` показывает, какие признаки потенциально нестабильны.

## 8. Weak deployment readiness

**Bottleneck:** исследовательский notebook-код трудно повторно использовать и невозможно безопасно применить на новых данных.

**Решение:**

- CLI команды: `build-panel`, `run-model-ready`, `run-demo`, `predict`.
- `model_package.joblib` сохраняет preprocessing, regime clusterer, autoencoder, bundles и overlay.
- `predict_model_ready` применяет те же transformations к новым наблюдениям.

## 9. Traceability and reproducibility

**Bottleneck:** без manifest невозможно понять, какие данные и параметры дали конкретный результат.

**Решение:**

- `run_manifest.json` содержит config, input fingerprint, список колонок и checksums артефактов.
- `data_quality_report.csv` и `feature_drift_report.csv` сохраняются автоматически.

## 10. Performance bottlenecks

**Bottleneck:** слишком тяжёлые модели могут зависать на обычных ноутбуках, особенно при thread oversubscription.

**Решение:**

- Для demo используется лёгкая конфигурация.
- `n_jobs` вынесен в config.
- KMeans реализован на NumPy для стабильного поведения.
- Walk-forward использует облегчённый candidate set.

## 11. Missing issuer-report intelligence

**Bottleneck:** рыночные и фундаментальные CSV дают числовой срез, но пропускают смысловую информацию из годовых отчетов, МСФО-отчетности, MD&A и примечаний: санкционные ограничения, валютные риски, ковенанты, обесценения, судебные споры, аудиторские оговорки, инвестиционные программы и риски рефинансирования.

**Решение:**

- Добавлен модуль `financial_reports.py`.
- Источники отчетов отделены от model-ready панели через `report_sources_ru_bluechips.csv` и `report_registry.csv`.
- Извлечение поддерживает PDF/XLSX/TXT/MD/HTML/ZIP.
- Для чисел используются bilingual dictionaries и robust number parser для русских/английских форматов.
- Для смыслов используются deterministic counts/flags, а не генеративные ответы.
- Присоединение к панели выполняется через `publish_date <= decision_date`.
- Добавлены coverage diagnostics и evidence snippets для аудита.

## 12. Report features can be noisy

**Bottleneck:** автоматическое извлечение из отчетов может ошибаться: PDF-таблицы плохо парсятся, разные эмитенты используют разные названия строк, текстовые предупреждения могут быть юридическими шаблонами, а не настоящим сигналом риска.

**Решение:**

- Для каждого ключевого показателя добавляется missing flag.
- В feature set входят `report_extraction_quality`, `report_available`, `report_lag_days` и `report_stale_flag`.
- Report-derived features проходят cross-sectional ranks и sector z-scores.
- В `pipeline.py` добавлен validation gate: отдельно обучаются `enriched_v2_without_reports` и `enriched_v2_with_reports`. Report layer включается только при validation gain.
- В `results/report_layer_demo` сохранена ablation-проверка, чтобы показать, как именно оценивается полезность слоя.

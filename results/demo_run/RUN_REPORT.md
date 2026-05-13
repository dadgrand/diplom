# Risk pipeline run report

## Sample sizes

- Train: 379
- Validation: 161
- Test: 169

## Test metrics

| architecture | macro-F1 | balanced accuracy | high recall | high precision |
| --- | ---: | ---: | ---: | ---: |
| baseline_rf | 0.3648 | 0.4212 | 0.7899 | 0.7344 |
| regime_only | 0.3526 | 0.4367 | 0.7311 | 0.7565 |
| ann_plus_regime | 0.4490 | 0.5080 | 0.7311 | 0.8286 |
| enriched_reference | 0.4554 | 0.5503 | 0.7647 | 0.8667 |
| sector_overlay | 0.4554 | 0.5503 | 0.7647 | 0.8667 |

## Stability

- Walk-forward folds: 4
- Walk-forward macro-F1 mean: 0.4425
- Walk-forward high-recall mean: 0.8064
- Drift warning count: 24

## Hybrid components

- Autoencoder backend: torch_autoencoder
- Selected sector experts: none
- Added features: 34

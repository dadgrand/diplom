# Risk pipeline run report

## Sample sizes

- Train: 153
- Validation: 66
- Test: 69

## Test metrics

| architecture | macro-F1 | balanced accuracy | high recall | high precision |
| --- | ---: | ---: | ---: | ---: |
| baseline_rf | 0.5027 | 0.4986 | 0.9512 | 0.7222 |
| regime_only | 0.4583 | 0.4837 | 0.9512 | 0.7091 |
| ann_plus_regime | 0.4925 | 0.5772 | 0.7317 | 0.9091 |
| enriched_reference | 0.5250 | 0.6016 | 0.8049 | 0.8462 |
| sector_overlay | 0.5250 | 0.6016 | 0.8049 | 0.8462 |

## Stability

- Walk-forward folds: 0
- Walk-forward macro-F1 mean: None
- Walk-forward high-recall mean: None
- Drift warning count: 102

## Hybrid components

- Autoencoder backend: torch_autoencoder
- Selected sector experts: none
- Added features: 53

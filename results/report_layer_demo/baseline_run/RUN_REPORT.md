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
| ann_plus_regime | 0.4395 | 0.4928 | 0.7561 | 0.7750 |
| enriched_reference | 0.5103 | 0.5623 | 0.8537 | 0.8140 |
| sector_overlay | 0.5103 | 0.5623 | 0.8537 | 0.8140 |

## Stability

- Walk-forward folds: 0
- Walk-forward macro-F1 mean: None
- Walk-forward high-recall mean: None
- Drift warning count: 38

## Hybrid components

- Autoencoder backend: torch_autoencoder
- Selected sector experts: none
- Added features: 34

# Risk pipeline run report

## Sample sizes

- Train: 248
- Validation: 102
- Test: 102

## Test metrics

| architecture | macro-F1 | balanced accuracy | high recall | high precision |
| --- | ---: | ---: | ---: | ---: |
| baseline_rf | 0.3385 | 0.3993 | 0.7812 | 0.3425 |
| regime_only | 0.2922 | 0.3808 | 0.9062 | 0.3452 |
| ann_plus_regime | 0.5410 | 0.5652 | 0.9062 | 0.4085 |
| enriched_reference | 0.3352 | 0.4093 | 0.9062 | 0.3816 |
| sector_overlay | 0.3352 | 0.4093 | 0.9062 | 0.3816 |

## Stability

- Walk-forward folds: 12
- Walk-forward macro-F1 mean: 0.2876
- Walk-forward high-recall mean: 0.6954
- Drift warning count: 129

## Hybrid components

- Autoencoder backend: pca_fallback
- Selected sector experts: none
- Added features: 53

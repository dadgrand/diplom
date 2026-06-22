# Risk pipeline run report

## Sample sizes

- Train: 248
- Validation: 102
- Test: 102

## Test metrics

| architecture | macro-F1 | balanced accuracy | high recall | high precision |
| --- | ---: | ---: | ---: | ---: |
| baseline_rf | 0.3416 | 0.4078 | 0.7812 | 0.3247 |
| regime_only | 0.2922 | 0.3808 | 0.9062 | 0.3452 |
| ann_plus_regime | 0.4117 | 0.4487 | 0.8438 | 0.3803 |
| enriched_reference | 0.4069 | 0.4468 | 0.8125 | 0.3562 |
| sector_overlay | 0.4069 | 0.4468 | 0.8125 | 0.3562 |
| final_selected | 0.4117 | 0.4487 | 0.8438 | 0.3803 |

## Final selection

- Selected architecture: ann_plus_regime
- Final mean confidence: 0.6477
- Final ECE, 10 bins: 0.2184
- Financial-report layer: with_reports
- Financial-report feature count: 60

## Stability

- Walk-forward folds: 12
- Walk-forward macro-F1 mean: 0.2960
- Walk-forward high-recall mean: 0.6506
- Drift warning count: 129

## Hybrid components

- Autoencoder backend: torch_autoencoder
- Selected sector experts: none
- Added features: 53

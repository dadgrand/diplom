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
| ann_plus_regime | 0.3024 | 0.3804 | 0.8750 | 0.3544 |
| enriched_reference | 0.3566 | 0.4244 | 0.8750 | 0.3544 |
| sector_overlay | 0.3342 | 0.4035 | 0.8750 | 0.3544 |
| final_selected | 0.3566 | 0.4244 | 0.8750 | 0.3544 |

## Final selection

- Selected architecture: enriched_reference
- Final mean confidence: 0.5945
- Final ECE, 10 bins: 0.2041
- Financial-report layer: without_reports
- Financial-report feature count: 60

## Stability

- Walk-forward folds: 12
- Walk-forward macro-F1 mean: 0.2569
- Walk-forward high-recall mean: 0.6314
- Drift warning count: 44

## Hybrid components

- Autoencoder backend: torch_autoencoder
- Selected sector experts: banks_financials
- Added features: 53

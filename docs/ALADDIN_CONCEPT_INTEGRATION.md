# Integrated ideas from the RU Aladdin concept

The diploma pipeline now integrates the strongest engineering ideas from the RU Aladdin MVP while keeping the final task focused on investment-risk classification.

## Ideas kept

1. **Deterministic model core.** All numbers are produced by Python code and saved as artifacts. No LLM is allowed to invent metrics.
2. **Source separation.** Market, macro, fundamentals and reports are separate layers that can be tested independently.
3. **Scenario mindset.** Report text is converted into explicit risk dimensions: liquidity, refinancing, sanctions, FX, capex, litigation and auditor warnings.
4. **Validation discipline.** Report features are not accepted automatically. The pipeline compares with-report and without-report architectures on chronological validation.
5. **Auditability.** Report extraction stores evidence snippets, document SHA-256, source URLs and coverage diagnostics.
6. **Reproducibility.** The model package, run manifest, metrics, predictions and drift reports remain part of every run.

## Ideas intentionally not copied as production dependencies

- Portfolio trading automation.
- LLM-driven numeric extraction without evidence.
- Investment recommendations.
- Unverified scraping assumptions that could break silently.

## Resulting architecture

```text
issuer report sources
        |
        v
report registry + local files + source URLs
        |
        v
text/table extraction -> numeric metrics + narrative risk counts + evidence
        |
        v
publish_date as-of merge into monthly model panel
        |
        v
feature engineering: report pressure, staleness, integrated stress
        |
        v
validation gate: without reports vs with reports
        |
        v
selected enriched model + regime layer + autoencoder factors + sector overlay
```

from __future__ import annotations

from pathlib import Path
from typing import Any


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def write_run_report(metrics: dict[str, Any], path: str | Path) -> None:
    """Create a compact markdown report for quick review of a run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Risk pipeline run report")
    lines.append("")
    lines.append("## Sample sizes")
    lines.append("")
    lines.append(f"- Train: {metrics.get('n_train')}")
    lines.append(f"- Validation: {metrics.get('n_validation')}")
    lines.append(f"- Test: {metrics.get('n_test')}")
    lines.append("")
    lines.append("## Test metrics")
    lines.append("")
    lines.append("| architecture | macro-F1 | balanced accuracy | high recall | high precision |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    test = metrics.get("test", {}) or {}
    for name, row in test.items():
        lines.append(
            f"| {name} | {_fmt(row.get('macro_f1'))} | {_fmt(row.get('balanced_accuracy'))} | {_fmt(row.get('high_recall'))} | {_fmt(row.get('high_precision'))} |"
        )
    lines.append("")
    lines.append("## Final selection")
    lines.append("")
    lines.append(f"- Selected architecture: {metrics.get('final_architecture') or 'n/a'}")
    final_probability = metrics.get("final_probability", {}) or {}
    if final_probability:
        lines.append(f"- Final mean confidence: {_fmt(final_probability.get('mean_confidence'))}")
        lines.append(f"- Final ECE, 10 bins: {_fmt(final_probability.get('ece_10_bins'))}")
    report_layer = metrics.get("report_layer_selection", {}) or {}
    if report_layer:
        lines.append(f"- Financial-report layer: {report_layer.get('selected')}")
        lines.append(f"- Financial-report feature count: {report_layer.get('report_feature_count')}")
    lines.append("")
    lines.append("## Stability")
    lines.append("")
    wf = metrics.get("walk_forward", {}) or {}
    lines.append(f"- Walk-forward folds: {wf.get('folds')}")
    lines.append(f"- Walk-forward macro-F1 mean: {_fmt(wf.get('macro_f1_mean'))}")
    lines.append(f"- Walk-forward high-recall mean: {_fmt(wf.get('high_recall_mean'))}")
    lines.append(f"- Drift warning count: {metrics.get('drift_warning_count')}")
    lines.append("")
    lines.append("## Hybrid components")
    lines.append("")
    lines.append(f"- Autoencoder backend: {metrics.get('autoencoder_backend')}")
    lines.append(f"- Selected sector experts: {', '.join(metrics.get('validation_selected_overlay_sectors') or []) or 'none'}")
    lines.append(f"- Added features: {metrics.get('added_features_count')}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASSES = ["low", "medium", "high"]
CLASS_LABELS = {"low": "Low", "medium": "Medium", "high": "High"}


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except TypeError:
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _metric_table(metrics: dict, names: list[str]) -> str:
    rows = ["| Architecture | Macro-F1 | Weighted-F1 | Balanced acc. | High recall | High precision | High FN rate | Adjacent acc. |"]
    rows.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in names:
        row = metrics["test"][name]
        rows.append(
            "| "
            + name
            + " | "
            + " | ".join(
                _fmt(row[key])
                for key in [
                    "macro_f1",
                    "weighted_f1",
                    "balanced_accuracy",
                    "high_recall",
                    "high_precision",
                    "high_false_negative_rate",
                    "ordinal_adjacent_accuracy",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def plot_architecture_metrics(metrics: dict, out: Path) -> None:
    names = ["baseline_rf", "regime_only", "enriched_reference", "ann_plus_regime", "sector_overlay", "final_selected"]
    labels = ["Baseline", "Regime", "Reports", "ANN+regime", "Overlay", "Final"]
    metric_keys = ["macro_f1", "balanced_accuracy", "high_recall"]
    metric_labels = ["Macro-F1", "Balanced acc.", "High recall"]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    colors = ["#2F6F9F", "#C46A2B", "#3C8D5A"]
    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        values = [metrics["test"][name][key] for name in names]
        ax.bar(x + (i - 1) * width, values, width, label=label, color=colors[i])
    ax.set_title("Test metrics by architecture")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(ncols=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out)


def plot_report_ablation(metrics: dict, out: Path) -> None:
    selection = metrics["report_layer_selection"]
    names = ["without_reports", "with_reports"]
    labels = ["Without reports", "With reports"]
    metric_keys = ["selection_score", "macro_f1", "weighted_f1", "high_recall"]
    metric_labels = ["Selection score", "Macro-F1", "Weighted-F1", "High recall"]
    x = np.arange(len(metric_labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for offset, name, color in [(-width / 2, names[0], "#7A8793"), (width / 2, names[1], "#2F8F6B")]:
        values = [selection[name][key] for key in metric_keys]
        ax.bar(x + offset, values, width, label=labels[names.index(name)], color=color)
        for xi, value in zip(x + offset, values):
            ax.text(xi, value + 0.012, _fmt(value, 3), ha="center", va="bottom", fontsize=8)
    ax.set_title("Financial-report layer validation gate")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Validation score")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=10, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out)


def plot_confusion(metrics: dict, out: Path) -> None:
    cm = pd.DataFrame(metrics["confusion_final"]).T
    cm = cm.reindex(index=[f"pred_{c}" for c in CLASSES], columns=[f"actual_{c}" for c in CLASSES]).fillna(0)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(cm.to_numpy(dtype=float), cmap="YlGnBu")
    ax.set_title("Final model confusion matrix")
    ax.set_xlabel("Actual class")
    ax.set_ylabel("Predicted class")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels([CLASS_LABELS[c] for c in CLASSES])
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels([CLASS_LABELS[c] for c in CLASSES])
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            value = int(cm.iloc[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="black", fontsize=12, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out)


def plot_classwise(metrics: dict, out: Path) -> None:
    frame = pd.DataFrame(metrics["classwise_final"])
    x = np.arange(len(frame))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = ["#2F6F9F", "#C46A2B", "#3C8D5A"]
    for i, key in enumerate(["precision", "recall", "f1"]):
        ax.bar(x + (i - 1) * width, frame[key], width, label=key, color=colors[i])
    ax.set_title("Final model classwise quality")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in frame["class"]])
    ax.legend(frameon=False, ncols=3)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out)


def plot_feature_importance(feature_importance: pd.DataFrame, final_architecture: str, out: Path) -> None:
    importance_col = "ann_branch_importance" if final_architecture == "ann_plus_regime" else "enriched_importance"
    top = feature_importance.sort_values(importance_col, ascending=False).head(20).iloc[::-1]
    is_report = top["feature"].str.startswith("report_") | top["feature"].eq("fundamental_report_gap")
    colors = np.where(is_report, "#2F8F6B", "#2F6F9F")
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    ax.barh(top["feature"], top[importance_col], color=colors)
    ax.set_title(f"Top-20 feature importance, {final_architecture}")
    ax.set_xlabel("Weighted importance")
    ax.grid(axis="x", alpha=0.25)
    report_patch = plt.Line2D([0], [0], color="#2F8F6B", lw=6, label="Report feature")
    market_patch = plt.Line2D([0], [0], color="#2F6F9F", lw=6, label="Market/macro/fundamental")
    ax.legend(handles=[market_patch, report_patch], frameon=False, loc="lower right")
    _save(fig, out)


def plot_walk_forward(walk_forward: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.9))
    ax.plot(walk_forward["fold"], walk_forward["macro_f1"], marker="o", color="#2F6F9F", label="Macro-F1")
    ax.plot(walk_forward["fold"], walk_forward["high_recall"], marker="o", color="#C46A2B", label="High recall")
    ax.axhline(walk_forward["macro_f1"].mean(), color="#2F6F9F", linestyle="--", alpha=0.45)
    ax.set_title("Walk-forward stability")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(walk_forward["fold"])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out)


def plot_drift(drift: pd.DataFrame, out: Path) -> None:
    top = drift.sort_values("psi", ascending=False).head(20).iloc[::-1]
    colors = np.where(top["period"].eq("test"), "#C46A2B", "#7A8793")
    fig, ax = plt.subplots(figsize=(9.4, 7.0))
    labels = top["feature"] + " (" + top["period"].astype(str) + ")"
    ax.barh(labels, top["psi"], color=colors)
    ax.axvline(0.2, color="#922B21", linestyle="--", linewidth=1.2, label="PSI warning threshold")
    ax.set_title("Top PSI drift diagnostics")
    ax.set_xlabel("PSI")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, out)


def build_report(run_dir: Path, assets: dict[str, Path]) -> None:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    names = ["baseline_rf", "regime_only", "enriched_reference", "ann_plus_regime", "sector_overlay", "final_selected"]
    final = metrics["test"]["final_selected"]
    report = metrics["report_layer_selection"]
    wf = metrics["walk_forward"]
    prob = metrics["final_probability"]
    classwise = pd.DataFrame(metrics["classwise_final"])

    lines: list[str] = []
    lines.append("# Материалы для защиты диплома")
    lines.append("")
    lines.append("## 1. Короткий вывод")
    lines.append("")
    lines.append(
        "В работе реализован воспроизводимый point-in-time pipeline классификации инвестиционного риска российских акций "
        "на три класса: `low`, `medium`, `high`. Финальная версия использует рыночные, ликвидностные, макроэкономические, "
        "фундаментальные признаки и признаки, извлеченные из финансовой отчетности эмитентов."
    )
    lines.append("")
    lines.append(f"- Финальная выбранная архитектура: `{metrics['final_architecture']}`.")
    lines.append(f"- Test macro-F1: **{_fmt(final['macro_f1'])}**.")
    lines.append(f"- Test weighted-F1: **{_fmt(final['weighted_f1'])}**.")
    lines.append(f"- Test balanced accuracy: **{_fmt(final['balanced_accuracy'])}**.")
    lines.append(f"- High-risk recall: **{_fmt(final['high_recall'])}**; high-risk false negative rate: **{_fmt(final['high_false_negative_rate'])}**.")
    lines.append(f"- Walk-forward: {wf['folds']} folds, mean macro-F1 **{_fmt(wf['macro_f1_mean'])}**, mean high recall **{_fmt(wf['high_recall_mean'])}**.")
    lines.append(f"- Drift warnings: **{metrics['drift_warning_count']}**, что показывает существенный сдвиг рынка между train и future-периодами.")
    lines.append("")
    lines.append(f"![Architecture metrics]({assets['architecture'].relative_to(run_dir)})")
    lines.append("")
    lines.append("## 2. Что было доработано перед защитой")
    lines.append("")
    lines.append(
        "1. Добавлен явный выбор финальной архитектуры по validation objective `macro_f1 + 0.03 * high_recall`. "
        "Это убирает ситуацию, когда сохраненный пакет использует `sector_overlay`, хотя на validation и test лучше работает `ann_plus_regime`."
    )
    lines.append("2. Выбранная архитектура сохраняется в `model_package.joblib`, а команда `predict` использует ее при инференсе.")
    lines.append("3. В зависимости добавлен `pyarrow`, чтобы parquet-входы и выходы из README работали без CSV fallback.")
    lines.append("4. Подготовлены графики для комиссии: сравнение архитектур, ablation финотчетности, confusion matrix, classwise-метрики, feature importance, walk-forward и drift.")
    lines.append("")
    lines.append("## 3. Данные и постановка задачи")
    lines.append("")
    lines.append(f"- Train / validation / test: **{metrics['n_train']} / {metrics['n_validation']} / {metrics['n_test']}** наблюдений.")
    lines.append(f"- Классы во всей размеченной выборке: low={metrics['class_distribution']['low']}, medium={metrics['class_distribution']['medium']}, high={metrics['class_distribution']['high']}.")
    lines.append(f"- Пороги RiskScore: low <= **{_fmt(metrics['target_thresholds']['low_upper'])}**, medium <= **{_fmt(metrics['target_thresholds']['medium_upper'])}**, выше - high.")
    lines.append(f"- Добавлено engineered features: **{metrics['added_features_count']}**.")
    lines.append("- Split строго временной: train заканчивается 2024-08-30, validation начинается 2024-09-30, test начинается 2025-03-31.")
    lines.append("")
    lines.append("Целевая переменная строится по будущему окну риска, но признаки берутся только на дату принятия решения или на дату уже опубликованной отчетности.")
    lines.append("")
    lines.append("## 4. Финансовая отчетность эмитентов")
    lines.append("")
    lines.append(f"- Report features present: **{report['report_features_present']}**.")
    lines.append(f"- Выбранный слой: **`{report['selected']}`**.")
    lines.append(f"- Количество report-признаков в модели: **{report['report_feature_count']}**.")
    lines.append(f"- Validation selection score без отчетности: **{_fmt(report['without_reports']['selection_score'])}**.")
    lines.append(f"- Validation selection score с отчетностью: **{_fmt(report['with_reports']['selection_score'])}**.")
    lines.append(f"- Macro-F1 без отчетности: **{_fmt(report['without_reports']['macro_f1'])}**; с отчетностью: **{_fmt(report['with_reports']['macro_f1'])}**.")
    lines.append("")
    lines.append(f"![Report layer ablation]({assets['report_ablation'].relative_to(run_dir)})")
    lines.append("")
    lines.append("Интерпретация для защиты: отчетность не используется как свободный текст для LLM-вывода. Она превращается в воспроизводимые числовые признаки: долговая нагрузка, покрытие процентов, cash-flow pressure, доля краткосрочного долга, признаки санкционного/валютного/ковенантного риска и stale/missing indicators. Присоединение выполняется point-in-time по `publish_date`, поэтому будущая отчетность не попадает в прошлые решения.")
    lines.append("")
    lines.append("## 5. Сравнение архитектур")
    lines.append("")
    lines.append(_metric_table(metrics, names))
    lines.append("")
    lines.append(
        "Финальная архитектура `ann_plus_regime` выбрана по validation, а не по test: ее validation selection score "
        f"равен **{_fmt(metrics['final_architecture_selection']['ann_plus_regime']['selection_score'])}**, "
        f"против **{_fmt(metrics['final_architecture_selection']['enriched_reference']['selection_score'])}** у enriched-reference."
    )
    lines.append("")
    lines.append("## 6. Ошибки классификации")
    lines.append("")
    lines.append(f"![Final confusion matrix]({assets['confusion'].relative_to(run_dir)})")
    lines.append("")
    lines.append(f"![Classwise quality]({assets['classwise'].relative_to(run_dir)})")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, row in classwise.iterrows():
        lines.append(f"| {row['class']} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['f1'])} | {int(row['support'])} |")
    lines.append("")
    lines.append("Главный акцент: модель консервативна в отношении высокого риска. Для класса `high` recall равен "
                 f"**{_fmt(classwise[classwise['class'].eq('high')]['recall'].iloc[0])}**, то есть модель редко пропускает высокий риск, "
                 "но часть среднерисковых бумаг переводит в high.")
    lines.append("")
    lines.append("## 7. Важность признаков")
    lines.append("")
    lines.append(f"![Feature importance]({assets['features'].relative_to(run_dir)})")
    lines.append("")
    lines.append("Среди важных факторов есть макро-рыночные признаки (`rate_fx_pressure`, `cbr_key_rate`, `average_market_correlation_60d`) и признаки отчетности, например `report_capex`, `report_staleness_weight` и report-derived stress features. Это хорошо защищается как гибридный подход: рыночный риск + финансовое состояние эмитента + режим рынка.")
    lines.append("")
    lines.append("## 8. Устойчивость и drift")
    lines.append("")
    lines.append(f"![Walk-forward stability]({assets['walk_forward'].relative_to(run_dir)})")
    lines.append("")
    lines.append(f"![Drift diagnostics]({assets['drift'].relative_to(run_dir)})")
    lines.append("")
    lines.append(
        "Наличие drift warnings не надо скрывать: это важный результат. Российский рынок в test-периоде отличается от train-периода, "
        "поэтому качество на test ниже validation. В работе это не замалчивается, а фиксируется через PSI, walk-forward и отдельные drift-отчеты."
    )
    lines.append("")
    lines.append("## 9. Примерный текст защиты")
    lines.append("")
    lines.append("Здравствуйте. В дипломной работе я разработал систему классификации инвестиционного риска российских акций. Задача формулируется не как прогноз доходности, а как отнесение бумаги на месячном срезе к одному из трех классов риска: low, medium или high.")
    lines.append("")
    lines.append("Целевая переменная построена через будущий риск-скор, который учитывает максимальную просадку, downside volatility, CVaR 95% и неликвидность. При этом все признаки формируются point-in-time: на дату решения доступны только текущие рыночные данные, макроэкономика и уже опубликованная отчетность.")
    lines.append("")
    lines.append(f"В выборке после разметки получилось {sum(metrics['class_distribution'].values())} наблюдений: {metrics['class_distribution']['low']} low, {metrics['class_distribution']['medium']} medium и {metrics['class_distribution']['high']} high. Разделение train-validation-test сделано строго по времени: {metrics['n_train']} наблюдений в train, {metrics['n_validation']} в validation и {metrics['n_test']} в test.")
    lines.append("")
    lines.append("Отдельная часть работы - слой финансовой отчетности. Я извлекаю из отчетов не произвольные LLM-выводы, а воспроизводимые признаки: долговую нагрузку, покрытие процентов, денежные потоки, маржинальность, признаки санкционного, валютного, ковенантного и ликвидностного риска. Эти признаки присоединяются по дате публикации отчета, чтобы исключить заглядывание в будущее.")
    lines.append("")
    lines.append(f"На validation я сравнил модель без отчетности и с отчетностью. Selection score вырос с {_fmt(report['without_reports']['selection_score'])} до {_fmt(report['with_reports']['selection_score'])}, macro-F1 - с {_fmt(report['without_reports']['macro_f1'])} до {_fmt(report['with_reports']['macro_f1'])}. Поэтому validation-gate выбрал вариант with_reports.")
    lines.append("")
    lines.append(f"Финальная архитектура выбиралась по validation-метрике, где macro-F1 дополняется небольшим бонусом за recall класса high. Победила архитектура `{metrics['final_architecture']}`. На test она дала macro-F1 {_fmt(final['macro_f1'])}, balanced accuracy {_fmt(final['balanced_accuracy'])}, high-risk recall {_fmt(final['high_recall'])} и high-risk false negative rate {_fmt(final['high_false_negative_rate'])}.")
    lines.append("")
    lines.append("Для задачи риск-менеджмента я считаю особенно важным recall класса high: ошибка пропуска высокого риска дороже, чем ложное завышение риска. Поэтому модель сознательно консервативна: она лучше ловит high-risk бумаги, но иногда относит medium к high.")
    lines.append("")
    lines.append(f"Устойчивость проверялась через walk-forward: {wf['folds']} фолдов, средний macro-F1 {_fmt(wf['macro_f1_mean'])}, средний high recall {_fmt(wf['high_recall_mean'])}. Также рассчитан drift diagnostics: найдено {metrics['drift_warning_count']} предупреждений PSI, что показывает сильное изменение распределений макро- и report-признаков между периодами.")
    lines.append("")
    lines.append("Главный результат работы - не один классификатор, а воспроизводимый исследовательский pipeline: сбор и подготовка панели, point-in-time признаки, финансовая отчетность, temporal validation, walk-forward, drift diagnostics, сохранение model package и отдельная команда predict для инференса.")
    lines.append("")
    lines.append("Ограничения работы я также фиксирую: выборка по российским акциям сравнительно мала, качество финансовой отчетности неоднородно, а рынок имеет сильный regime shift. Поэтому дальнейшие улучшения - расширение universe, улучшение парсинга отчетов и регулярное переобучение модели.")
    lines.append("")
    lines.append("## 10. Что открыть на защите")
    lines.append("")
    lines.append("1. Этот файл: `results/defense_run/DEFENSE_MATERIALS.md`.")
    lines.append("2. Графики из папки `results/defense_run/defense_assets/`.")
    lines.append("3. Полные метрики: `results/defense_run/metrics.json`.")
    lines.append("4. Предсказания test-периода: `results/defense_run/predictions.csv`.")
    lines.append("5. Сохраненный пакет модели: `results/defense_run/model_package.joblib`.")
    lines.append("")
    lines.append("Команда воспроизведения:")
    lines.append("")
    lines.append("```bash")
    lines.append("source .venv/bin/activate")
    lines.append("risk-pipeline --config configs/config.example.yaml \\")
    lines.append("  run-model-ready \\")
    lines.append("  --input data/processed/monthly_model_ready.csv \\")
    lines.append("  --out results/defense_run")
    lines.append(".venv/bin/python scripts/make_defense_materials.py results/defense_run")
    lines.append("```")
    lines.append("")
    (run_dir / "DEFENSE_MATERIALS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build defense charts and markdown report for a completed run")
    parser.add_argument("run_dir", nargs="?", default="results/defense_run")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    assets_dir = run_dir / "defense_assets"
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    feature_importance = pd.read_csv(run_dir / "feature_importance.csv")
    walk_forward = pd.read_csv(run_dir / "walk_forward_report.csv")
    drift = pd.read_csv(run_dir / "feature_drift_report.csv")

    assets = {
        "architecture": assets_dir / "architecture_test_metrics.png",
        "report_ablation": assets_dir / "report_layer_ablation.png",
        "confusion": assets_dir / "confusion_matrix_final.png",
        "classwise": assets_dir / "classwise_final_metrics.png",
        "features": assets_dir / "feature_importance_top20.png",
        "walk_forward": assets_dir / "walk_forward_stability.png",
        "drift": assets_dir / "drift_top20_psi.png",
    }

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    plot_architecture_metrics(metrics, assets["architecture"])
    plot_report_ablation(metrics, assets["report_ablation"])
    plot_confusion(metrics, assets["confusion"])
    plot_classwise(metrics, assets["classwise"])
    plot_feature_importance(feature_importance, metrics["final_architecture"], assets["features"])
    plot_walk_forward(walk_forward, assets["walk_forward"])
    plot_drift(drift, assets["drift"])
    build_report(run_dir, assets)


if __name__ == "__main__":
    main()

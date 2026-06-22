from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from .models import CLASSES, ProbabilityPolicy, make_tree_model
from .preprocessing import FinancialPreprocessor

EPS = 1e-12

def align_proba(model_classes: list[str] | np.ndarray, proba: np.ndarray, target_classes: list[str] = CLASSES) -> np.ndarray:
    """Align probability columns to low/medium/high."""
    model_classes = [str(c) for c in model_classes]
    out = np.zeros((proba.shape[0], len(target_classes)), dtype=float)
    for j, cls in enumerate(target_classes):
        if cls in model_classes:
            out[:, j] = proba[:, model_classes.index(cls)]
    row_sum = out.sum(axis=1, keepdims=True)
    missing = row_sum[:, 0] <= EPS
    if missing.any():
        out[missing, :] = 1.0 / len(target_classes)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


@dataclass
class PowerProbabilityCalibrator:
    """Small validation-only probability calibrator."""

    classes: list[str] = field(default_factory=lambda: CLASSES.copy())
    gamma_grid: list[float] = field(default_factory=lambda: [0.65, 0.80, 1.00, 1.25, 1.50, 1.80])
    gamma_: float = 1.0
    validation_log_loss_: float | None = None

    def transform(self, proba: np.ndarray, gamma: float | None = None) -> np.ndarray:
        gamma = self.gamma_ if gamma is None else gamma
        p = np.asarray(proba, dtype=float).clip(EPS, 1.0)
        p = p**gamma
        return p / p.sum(axis=1, keepdims=True)

    def fit(self, y_true: np.ndarray | pd.Series, proba: np.ndarray) -> "PowerProbabilityCalibrator":
        y_true = np.asarray(y_true).astype(str)
        best_loss = np.inf
        best_gamma = 1.0
        for gamma in self.gamma_grid:
            calibrated = self.transform(proba, gamma=gamma)
            idx = np.array([self.classes.index(y) if y in self.classes else -1 for y in y_true], dtype=int)
            valid = idx >= 0
            if not valid.any():
                continue
            loss = -np.mean(np.log(calibrated[np.arange(len(idx))[valid], idx[valid]].clip(EPS, 1.0)))
            if loss < best_loss:
                best_loss = float(loss)
                best_gamma = float(gamma)
        self.gamma_ = best_gamma
        self.validation_log_loss_ = best_loss if np.isfinite(best_loss) else None
        return self


@dataclass
class WeightedProbabilityEnsemble:
    """Soft voting with weights chosen on validation."""

    classes: list[str] = field(default_factory=lambda: CLASSES.copy())
    weight_step: float = 0.10
    high_recall_bonus: float = 0.0
    weights_: dict[str, float] = field(default_factory=dict)
    validation_score_: float | None = None
    validation_objective_: float | None = None

    def _weight_grid(self, names: list[str]) -> list[np.ndarray]:
        n = len(names)
        if n == 1:
            return [np.ones(1)]
        grid = np.arange(0, 1 + self.weight_step / 2, self.weight_step)
        candidates: list[np.ndarray] = []
        for weights in product(grid, repeat=n):
            w = np.asarray(weights, dtype=float)
            if np.isclose(w.sum(), 1.0):
                candidates.append(w)
        for i in range(n):
            w = np.zeros(n); w[i] = 1.0; candidates.append(w)
        return candidates

    def fit(self, y_true: np.ndarray | pd.Series, proba_by_name: dict[str, np.ndarray]) -> "WeightedProbabilityEnsemble":
        names = list(proba_by_name)
        if not names:
            raise ValueError("No candidate probabilities supplied")
        y_true = np.asarray(y_true).astype(str)
        best = (-np.inf, -np.inf, np.ones(len(names)) / len(names))
        for w in self._weight_grid(names):
            p = sum(w[i] * proba_by_name[names[i]] for i in range(len(names)))
            pred = np.array([self.classes[i] for i in np.argmax(p, axis=1)], dtype=object)
            score = f1_score(y_true, pred, labels=self.classes, average="macro", zero_division=0)
            high_recall = ((pred == "high") & (y_true == "high")).sum() / max((y_true == "high").sum(), 1)
            objective = score + self.high_recall_bonus * high_recall
            if objective > best[0]:
                best = (float(objective), float(score), w.copy())
        self.validation_objective_ = best[0]
        self.validation_score_ = best[1]
        self.weights_ = {name: float(best[2][i]) for i, name in enumerate(names)}
        return self

    def predict_proba(self, proba_by_name: dict[str, np.ndarray]) -> np.ndarray:
        if not self.weights_:
            raise RuntimeError("WeightedProbabilityEnsemble is not fitted")
        out = None
        for name, weight in self.weights_.items():
            if name not in proba_by_name:
                raise KeyError(f"Missing probabilities for candidate {name}")
            piece = weight * proba_by_name[name]
            out = piece if out is None else out + piece
        assert out is not None
        return out / out.sum(axis=1, keepdims=True)


@dataclass
class CandidateBundle:
    """Everything needed to score one architecture."""

    name: str
    features: list[str]
    categorical: list[str]
    preprocessor: FinancialPreprocessor
    models: dict[str, Any]
    ensemble: WeightedProbabilityEnsemble
    calibrator: PowerProbabilityCalibrator
    policy: ProbabilityPolicy
    classes: list[str] = field(default_factory=lambda: CLASSES.copy())
    validation_leaderboard: pd.DataFrame | None = None
    feature_importance: pd.DataFrame | None = None

    def _candidate_probas(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        x = self.preprocessor.transform(df)
        out: dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            out[name] = align_proba(model.classes_, model.predict_proba(x), self.classes)
        return out

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = self.ensemble.predict_proba(self._candidate_probas(df))
        return self.calibrator.transform(p)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.policy.predict_from_proba(self.predict_proba(df))


def _fit_single_candidate(name: str, x_train: pd.DataFrame, y_train: pd.Series, *, n_estimators: int, min_samples_leaf: int, class_weight: str | None, random_state: int, n_jobs: int):
    model = make_tree_model(
        name,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    if name in {"hgb", "gb"}:
        sample_weight = compute_sample_weight("balanced", y_train)
        model.fit(x_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(x_train, y_train)
    return model


def _aggregate_feature_importance(models: dict[str, Any], preprocessor: FinancialPreprocessor, weights: dict[str, float]) -> pd.DataFrame:
    frames = []
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
        else:
            imp = np.zeros(len(preprocessor.feature_names_), dtype=float)
        frames.append(pd.DataFrame({"feature": preprocessor.feature_names_, f"{name}_importance": imp}))
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="feature", how="outer")
    result = result.fillna(0.0)
    result["weighted_importance"] = 0.0
    for name, weight in weights.items():
        col = f"{name}_importance"
        if col in result:
            result["weighted_importance"] += weight * result[col]
    return result.sort_values("weighted_importance", ascending=False)


def fit_candidate_bundle(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    features: list[str],
    categorical: list[str],
    candidates: list[str],
    n_estimators: int,
    min_samples_leaf: int,
    class_weight: str | None,
    random_state: int,
    n_jobs: int,
    ensemble_weight_step: float,
    calibration_grid: list[float],
    high_recall_bonus: float,
) -> CandidateBundle:
    numeric = [c for c in features if c not in categorical]
    cat = [c for c in categorical if c in features]
    pre = FinancialPreprocessor(numeric_features=numeric, categorical_features=cat)
    x_train = pre.fit_transform(train)
    x_val = pre.transform(validation)
    y_train = train["risk_class"].astype(str)
    y_val = validation["risk_class"].astype(str)

    fitted_models: dict[str, Any] = {}
    val_probas: dict[str, np.ndarray] = {}
    rows = []
    for i, candidate in enumerate(candidates):
        model = _fit_single_candidate(
            candidate,
            x_train,
            y_train,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state + 101 * i,
            n_jobs=n_jobs,
        )
        fitted_models[candidate] = model
        proba = align_proba(model.classes_, model.predict_proba(x_val), CLASSES)
        val_probas[candidate] = proba
        pred = np.array([CLASSES[j] for j in np.argmax(proba, axis=1)], dtype=object)
        macro = float(f1_score(y_val, pred, labels=CLASSES, average="macro", zero_division=0))
        high_recall = float(((pred == "high") & (y_val.to_numpy() == "high")).sum() / max((y_val.to_numpy() == "high").sum(), 1))
        rows.append(
            {
                "candidate": candidate,
                "validation_macro_f1_argmax": macro,
                "validation_high_recall_argmax": high_recall,
                "validation_selection_objective": macro + high_recall_bonus * high_recall,
            }
        )

    ensemble = WeightedProbabilityEnsemble(classes=CLASSES, weight_step=ensemble_weight_step, high_recall_bonus=0.0).fit(y_val, val_probas)
    ensemble_val = ensemble.predict_proba(val_probas)
    calibrator = PowerProbabilityCalibrator(classes=CLASSES, gamma_grid=calibration_grid).fit(y_val, ensemble_val)
    calibrated_val = calibrator.transform(ensemble_val)
    policy = ProbabilityPolicy(classes=CLASSES, high_recall_bonus=high_recall_bonus).fit(y_val, calibrated_val)
    pred_policy = policy.predict_from_proba(calibrated_val)

    leaderboard = pd.DataFrame(rows)
    ensemble_argmax_pred = np.array([CLASSES[j] for j in np.argmax(ensemble_val, axis=1)], dtype=object)
    ensemble_macro = float(f1_score(y_val, ensemble_argmax_pred, labels=CLASSES, average="macro", zero_division=0))
    ensemble_high_recall = float(((ensemble_argmax_pred == "high") & (y_val.to_numpy() == "high")).sum() / max((y_val.to_numpy() == "high").sum(), 1))
    leaderboard.loc[len(leaderboard)] = {
        "candidate": "weighted_ensemble_argmax",
        "validation_macro_f1_argmax": ensemble_macro,
        "validation_high_recall_argmax": ensemble_high_recall,
        "validation_selection_objective": float(ensemble.validation_objective_ or 0.0),
    }
    policy_macro = float(f1_score(y_val, pred_policy, labels=CLASSES, average="macro", zero_division=0))
    policy_high_recall = float(((pred_policy == "high") & (y_val.to_numpy() == "high")).sum() / max((y_val.to_numpy() == "high").sum(), 1))
    leaderboard.loc[len(leaderboard)] = {
        "candidate": "weighted_ensemble_calibrated_policy",
        "validation_macro_f1_argmax": policy_macro,
        "validation_high_recall_argmax": policy_high_recall,
        "validation_selection_objective": policy_macro + high_recall_bonus * policy_high_recall,
    }
    fi = _aggregate_feature_importance(fitted_models, pre, ensemble.weights_)
    return CandidateBundle(
        name=name,
        features=features,
        categorical=cat,
        preprocessor=pre,
        models=fitted_models,
        ensemble=ensemble,
        calibrator=calibrator,
        policy=policy,
        classes=CLASSES,
        validation_leaderboard=leaderboard.sort_values("validation_macro_f1_argmax", ascending=False).reset_index(drop=True),
        feature_importance=fi,
    )

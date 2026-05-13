from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Torch is optional. If it is unavailable, AutoencoderFactors automatically falls
# back to PCA while preserving the same downstream interface.
try:  # pragma: no cover - import path depends on environment
    import torch
    from torch import nn

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
except Exception:  # pragma: no cover - exercised in environments without torch
    torch = None
    nn = None


if nn is not None:  # pragma: no cover - class is exercised only when torch is available
    class _AutoencoderNet(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            out = self.decoder(z)
            return out, z
else:
    _AutoencoderNet = object


def _load_torch():  # pragma: no cover - environment-dependent
    return torch, nn


CLASSES = ["low", "medium", "high"]


def make_tree_model(
    kind: str = "rf",
    *,
    n_estimators: int = 500,
    min_samples_leaf: int = 4,
    class_weight: str | None = "balanced_subsample",
    random_state: int = 42,
    n_jobs: int = -1,
):
    """Create a tree-like model used by candidate ensembles."""
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if kind == "gb":
        return GradientBoostingClassifier(
            n_estimators=max(50, min(n_estimators, 180)),
            learning_rate=0.045,
            max_depth=3,
            min_samples_leaf=max(min_samples_leaf, 5),
            subsample=0.85,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model kind: {kind}")


@dataclass
class ProbabilityPolicy:
    """Validation-fitted probability thresholds for risk-oriented decisions."""

    classes: list[str] = field(default_factory=lambda: CLASSES.copy())
    low_threshold_: float = 0.50
    high_threshold_: float = 0.45
    validation_score_: float | None = None
    high_recall_bonus: float = 0.02

    def fit(self, y_true: np.ndarray | pd.Series, proba: np.ndarray, grid: np.ndarray | None = None) -> "ProbabilityPolicy":
        grid = grid if grid is not None else np.arange(0.25, 0.76, 0.025)
        y_true = np.asarray(y_true)
        best = (-np.inf, self.low_threshold_, self.high_threshold_)
        for low_thr in grid:
            for high_thr in grid:
                pred = self.predict_from_proba(proba, low_threshold=low_thr, high_threshold=high_thr)
                score = f1_score(y_true, pred, labels=self.classes, average="macro", zero_division=0)
                high_recall = ((pred == "high") & (y_true == "high")).sum() / max((y_true == "high").sum(), 1)
                objective = score + self.high_recall_bonus * high_recall
                if objective > best[0]:
                    best = (objective, float(low_thr), float(high_thr))
        self.validation_score_ = float(best[0])
        self.low_threshold_ = best[1]
        self.high_threshold_ = best[2]
        return self

    def predict_from_proba(
        self,
        proba: np.ndarray,
        *,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
    ) -> np.ndarray:
        low_threshold = self.low_threshold_ if low_threshold is None else low_threshold
        high_threshold = self.high_threshold_ if high_threshold is None else high_threshold
        proba = np.asarray(proba)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        base = np.array([self.classes[i] for i in np.argmax(proba, axis=1)], dtype=object)
        if "high" in class_to_idx:
            high_mask = proba[:, class_to_idx["high"]] >= high_threshold
            base[high_mask] = "high"
        if "low" in class_to_idx:
            low_mask = (proba[:, class_to_idx["low"]] >= low_threshold) & (base != "high")
            base[low_mask] = "low"
        return base


@dataclass
class RegimeClusterer:
    """KMeans-like regime labels fitted only on train-period macro features."""

    features: list[str]
    n_clusters: int = 2
    random_state: int = 42
    scaler_: StandardScaler | None = None
    centroids_: np.ndarray | None = None
    medians_: pd.Series | None = None
    inertia_: float | None = None

    @staticmethod
    def _fit_numpy_kmeans(x: np.ndarray, n_clusters: int, random_state: int, n_init: int = 16, max_iter: int = 120) -> tuple[np.ndarray, float]:
        """Deterministic NumPy KMeans implementation.

        It avoids environment-specific threaded BLAS issues and keeps the thesis
        pipeline reproducible on ordinary laptops.
        """
        rng = np.random.default_rng(random_state)
        n = x.shape[0]
        if n == 0:
            raise ValueError("Cannot fit regime clusters on an empty matrix")
        k = min(n_clusters, n)
        best_centroids = None
        best_inertia = np.inf
        for _ in range(n_init):
            init_idx = rng.choice(n, size=k, replace=False)
            centroids = x[init_idx].copy()
            labels = np.zeros(n, dtype=int)
            for iter_idx in range(max_iter):
                distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
                new_labels = distances.argmin(axis=1)
                if np.array_equal(labels, new_labels) and iter_idx > 0:
                    break
                labels = new_labels
                for j in range(k):
                    mask = labels == j
                    centroids[j] = x[mask].mean(axis=0) if mask.any() else x[rng.integers(0, n)]
            inertia = float(((x - centroids[labels]) ** 2).sum())
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.copy()
        assert best_centroids is not None
        return best_centroids, best_inertia

    def _matrix(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        x = df[self.features].apply(pd.to_numeric, errors="coerce")
        if fit:
            self.medians_ = x.median(axis=0).fillna(0.0)
        if self.medians_ is None:
            raise RuntimeError("RegimeClusterer is not fitted")
        return x.fillna(self.medians_).fillna(0.0)

    def fit(self, df: pd.DataFrame) -> "RegimeClusterer":
        x = self._matrix(df, fit=True)
        self.scaler_ = StandardScaler()
        xs = self.scaler_.fit_transform(x)
        self.centroids_, self.inertia_ = self._fit_numpy_kmeans(xs, self.n_clusters, self.random_state)
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        if self.scaler_ is None or self.centroids_ is None:
            raise RuntimeError("RegimeClusterer is not fitted")
        x = self._matrix(df, fit=False)
        xs = self.scaler_.transform(x)
        distances = ((xs[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        return pd.Series([f"regime_{int(v)}" for v in labels], index=df.index, name="regime_cluster")

    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        return self.fit(df).transform(df)


@dataclass
class AutoencoderFactors:
    """Compact nonlinear risk-factor extractor with train-only preprocessing."""

    numeric_features: list[str]
    latent_dim: int = 3
    hidden_dim: int = 8
    epochs: int = 250
    lr: float = 1e-3
    weight_decay: float = 1e-4
    random_state: int = 42
    scaler_: StandardScaler | None = None
    medians_: pd.Series | None = None
    net_: Any | None = None
    fallback_: PCA | None = None
    fitted_with_: str | None = None
    reconstruction_loss_: float | None = None

    def _matrix(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if not self.numeric_features:
            raise ValueError("AutoencoderFactors requires at least one numeric feature")
        x = df[self.numeric_features].apply(pd.to_numeric, errors="coerce")
        if fit:
            self.medians_ = x.median(axis=0).fillna(0.0)
        if self.medians_ is None:
            raise RuntimeError("AutoencoderFactors is not fitted")
        return x.fillna(self.medians_).fillna(0.0)

    def fit(self, df: pd.DataFrame) -> "AutoencoderFactors":
        x = self._matrix(df, fit=True)
        self.scaler_ = StandardScaler()
        xs = self.scaler_.fit_transform(x).astype("float32")
        latent_dim = min(self.latent_dim, xs.shape[1])
        torch_mod, nn_mod = _load_torch()
        if torch_mod is None or nn_mod is None or xs.shape[0] < 20:
            self.fallback_ = PCA(n_components=latent_dim, random_state=self.random_state)
            self.fallback_.fit(xs)
            reconstructed = self.fallback_.inverse_transform(self.fallback_.transform(xs))
            self.reconstruction_loss_ = float(np.mean((xs - reconstructed) ** 2))
            self.fitted_with_ = "pca_fallback"
            return self
        torch_mod.manual_seed(self.random_state)
        net = _AutoencoderNet(xs.shape[1], self.hidden_dim, latent_dim)
        optimizer = torch_mod.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn_mod.MSELoss()
        tensor = torch_mod.tensor(xs, dtype=torch_mod.float32)
        net.train()
        best_loss = np.inf
        patience = 20
        stale = 0
        best_state = None
        for _ in range(self.epochs):
            optimizer.zero_grad()
            reconstructed, _ = net(tensor)
            loss = loss_fn(reconstructed, tensor)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            if loss_value + 1e-7 < best_loss:
                best_loss = loss_value
                stale = 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net_ = net.eval()
        self.reconstruction_loss_ = float(best_loss)
        self.fitted_with_ = "torch_autoencoder"
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_ is None:
            raise RuntimeError("AutoencoderFactors is not fitted")
        x = self._matrix(df, fit=False)
        xs = self.scaler_.transform(x).astype("float32")
        if self.fallback_ is not None:
            z = self.fallback_.transform(xs)
        else:
            torch_mod, _ = _load_torch()
            with torch_mod.no_grad():
                _, latent = self.net_(torch_mod.tensor(xs, dtype=torch_mod.float32))
                z = latent.detach().cpu().numpy()
        return pd.DataFrame(z, index=df.index, columns=[f"latent_factor_{i+1}" for i in range(z.shape[1])])

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


@dataclass
class SectorExpertOverlay:
    """Select a specialized expert only in sectors where validation gain is robust."""

    min_sector_rows: int = 18
    min_gain: float = 0.10
    selected_sectors_: set[str] = field(default_factory=set)
    sector_report_: pd.DataFrame | None = None

    def fit(self, validation: pd.DataFrame, *, sector_col: str, y_col: str, global_pred_col: str, expert_pred_col: str) -> "SectorExpertOverlay":
        rows = []
        self.selected_sectors_ = set()
        for sector, group in validation.groupby(sector_col):
            global_score = f1_score(group[y_col], group[global_pred_col], labels=CLASSES, average="macro", zero_division=0)
            expert_score = f1_score(group[y_col], group[expert_pred_col], labels=CLASSES, average="macro", zero_division=0)
            gain = float(expert_score - global_score)
            selected = len(group) >= self.min_sector_rows and gain >= self.min_gain
            if selected:
                self.selected_sectors_.add(str(sector))
            rows.append({"sector": sector, "rows": int(len(group)), "global_macro_f1": float(global_score), "expert_macro_f1": float(expert_score), "gain": gain, "selected": bool(selected)})
        self.sector_report_ = pd.DataFrame(rows).sort_values(["selected", "gain"], ascending=[False, False]) if rows else pd.DataFrame()
        return self

    def predict(self, df: pd.DataFrame, *, sector_col: str, global_pred_col: str, expert_pred_col: str) -> np.ndarray:
        use_expert = df[sector_col].astype(str).isin(self.selected_sectors_)
        return np.where(use_expert, df[expert_pred_col], df[global_pred_col])

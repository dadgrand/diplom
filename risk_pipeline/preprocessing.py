from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schemas import require_columns


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class FinancialPreprocessor:
    """Train-only imputation, winsorization, signed-log transform and one-hot encoding."""

    numeric_features: list[str]
    categorical_features: list[str] = field(default_factory=list)
    clip_quantiles: tuple[float, float] = (0.01, 0.99)
    medians_: pd.Series | None = None
    lower_: pd.Series | None = None
    upper_: pd.Series | None = None
    mean_: pd.Series | None = None
    std_: pd.Series | None = None
    categories_: dict[str, list[str]] | None = None
    feature_names_: list[str] | None = None

    def fit(self, df: pd.DataFrame) -> "FinancialPreprocessor":
        require_columns(df, self.numeric_features + self.categorical_features, "preprocessor.fit")
        num = df[self.numeric_features].apply(pd.to_numeric, errors="coerce")
        self.medians_ = num.median(axis=0).fillna(0.0)
        filled = num.fillna(self.medians_)
        self.lower_ = filled.quantile(self.clip_quantiles[0])
        self.upper_ = filled.quantile(self.clip_quantiles[1])
        clipped = filled.clip(self.lower_, self.upper_, axis=1)
        transformed = pd.DataFrame(signed_log1p(clipped.to_numpy(dtype=float)), columns=self.numeric_features, index=df.index)
        self.mean_ = transformed.mean(axis=0)
        self.std_ = transformed.std(axis=0).replace(0.0, 1.0).fillna(1.0)
        self.categories_ = {}
        for col in self.categorical_features:
            values = df[col].astype("object").where(df[col].notna(), "__MISSING__").astype(str)
            self.categories_[col] = sorted(values.unique().tolist())
        cat_names = [f"{col}={value}" for col, cats in self.categories_.items() for value in cats]
        self.feature_names_ = list(self.numeric_features) + cat_names
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if any(x is None for x in [self.medians_, self.lower_, self.upper_, self.mean_, self.std_, self.categories_, self.feature_names_]):
            raise RuntimeError("FinancialPreprocessor is not fitted")
        require_columns(df, self.numeric_features + self.categorical_features, "preprocessor.transform")
        num = df[self.numeric_features].apply(pd.to_numeric, errors="coerce")
        filled = num.fillna(self.medians_)
        clipped = filled.clip(self.lower_, self.upper_, axis=1)
        transformed = pd.DataFrame(signed_log1p(clipped.to_numpy(dtype=float)), columns=self.numeric_features, index=df.index)
        scaled = (transformed - self.mean_) / self.std_
        pieces = [scaled.reset_index(drop=True)]
        for col, cats in self.categories_.items():
            values = df[col].astype("object").where(df[col].notna(), "__MISSING__").astype(str)
            one_hot = pd.DataFrame({f"{col}={cat}": (values == cat).astype(int).to_numpy() for cat in cats})
            pieces.append(one_hot)
        result = pd.concat(pieces, axis=1)
        return result[self.feature_names_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

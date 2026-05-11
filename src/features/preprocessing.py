from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RegressionSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_features_target(dataset: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if target not in dataset.columns:
        msg = f"Target column '{target}' is missing from dataset."
        raise KeyError(msg)

    features = dataset.drop(columns=[target])
    y = dataset[target]
    return features, y


def select_numeric_features(features: pd.DataFrame) -> pd.DataFrame:
    numeric_features = features.select_dtypes(include=[np.number]).copy()
    if numeric_features.shape[1] == 0:
        msg = "No numeric features found in the dataset."
        raise ValueError(msg)
    return numeric_features


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    random_state: int,
) -> RegressionSplit:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return RegressionSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def make_standardized_pipeline(regressor: RegressorMixin) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )


def with_log_target_transform(regressor: RegressorMixin) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )

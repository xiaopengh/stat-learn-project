from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

from src.features import make_standardized_pipeline, with_log_target_transform


def _as_float_list(values: Any, *, key: str) -> list[float]:
    if not isinstance(values, list):
        msg = f"Expected a list for config key '{key}'."
        raise TypeError(msg)
    return [float(value) for value in values]


def _as_int_list(values: Any, *, key: str) -> list[int]:
    if not isinstance(values, list):
        msg = f"Expected a list for config key '{key}'."
        raise TypeError(msg)
    return [int(value) for value in values]


def _maybe_log_target(regressor: RegressorMixin, *, apply_log_target: bool) -> RegressorMixin:
    if not apply_log_target:
        return regressor
    return with_log_target_transform(regressor)


def build_model_registry(
    config: Mapping[str, Any],
    *,
    apply_log_target: bool = True,
) -> dict[str, RegressorMixin]:
    project_config = config.get("project", {})
    cv_config = config.get("cross_validation", {})
    models_config = config.get("models", {})
    hgb_config = models_config.get("hist_gradient_boosting", {})

    random_seed = int(project_config.get("random_seed", 42))
    folds = int(cv_config.get("folds", 5))
    n_jobs = int(cv_config.get("n_jobs", -1))

    ridge_alphas = _as_float_list(
        models_config.get("ridge_alphas", [0.1, 1.0, 10.0]),
        key="models.ridge_alphas",
    )
    lasso_alphas = _as_float_list(
        models_config.get("lasso_alphas", [0.001, 0.01, 0.1]),
        key="models.lasso_alphas",
    )
    knn_neighbors = _as_int_list(
        models_config.get("knn_neighbors", [3, 5, 7]),
        key="models.knn_neighbors",
    )

    linear_regression = make_standardized_pipeline(LinearRegression())
    ridge_regression = make_standardized_pipeline(RidgeCV(alphas=ridge_alphas, cv=folds))
    lasso_regression = make_standardized_pipeline(
        LassoCV(
            alphas=lasso_alphas,
            cv=folds,
            random_state=random_seed,
            n_jobs=n_jobs,
            max_iter=10_000,
        )
    )
    knn_search = GridSearchCV(
        estimator=make_standardized_pipeline(KNeighborsRegressor()),
        param_grid={"regressor__n_neighbors": knn_neighbors},
        cv=folds,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
        refit=True,
    )
    hist_gradient_boosting = HistGradientBoostingRegressor(
        learning_rate=float(hgb_config.get("learning_rate", 0.05)),
        max_iter=int(hgb_config.get("max_iter", 300)),
        max_leaf_nodes=int(hgb_config.get("max_leaf_nodes", 31)),
        l2_regularization=float(hgb_config.get("l2_regularization", 0.0)),
        validation_fraction=float(hgb_config.get("validation_fraction", 0.1)),
        n_iter_no_change=int(hgb_config.get("n_iter_no_change", 20)),
        random_state=random_seed,
    )

    return {
        "dummy_regressor": _maybe_log_target(
            DummyRegressor(strategy="mean"),
            apply_log_target=apply_log_target,
        ),
        "linear_regression": _maybe_log_target(
            linear_regression,
            apply_log_target=apply_log_target,
        ),
        "ridge_cv": _maybe_log_target(ridge_regression, apply_log_target=apply_log_target),
        "lasso_cv": _maybe_log_target(lasso_regression, apply_log_target=apply_log_target),
        "knn_gridsearch": _maybe_log_target(knn_search, apply_log_target=apply_log_target),
        "hist_gradient_boosting": _maybe_log_target(
            hist_gradient_boosting,
            apply_log_target=apply_log_target,
        ),
    }

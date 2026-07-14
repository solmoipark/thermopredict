from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, LeaveOneOut


@dataclass(frozen=True)
class ModelResult:
    name: str
    score: float
    mae: dict[str, float]
    rmse: dict[str, float]
    normalized_rmse: dict[str, float]
    r2: dict[str, float]
    predictions: np.ndarray | None = None  # cross-validated predictions, row-aligned


def compare_models(
    models: dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    target_columns: list[str],
    validation: str = "loo",
    groups: np.ndarray | None = None,
) -> list[ModelResult]:
    splitter = _make_splitter(len(X), validation)
    if validation == "logo" and groups is None:
        raise ValueError("validation='logo' requires group labels.")
    results: list[ModelResult] = []
    scales = _target_scales(y)

    for name, estimator in models.items():
        try:
            predicted = np.zeros_like(y, dtype=float)
            for train_index, test_index in splitter.split(X, groups=groups):
                model = clone(estimator)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    model.fit(X[train_index], y[train_index])
                predicted[test_index] = model.predict(X[test_index])
        except Exception as error:
            print(f"Skipping model '{name}': {error}")
            continue

        mae_values = mean_absolute_error(y, predicted, multioutput="raw_values")
        rmse_values = np.sqrt(mean_squared_error(y, predicted, multioutput="raw_values"))
        norm_values = rmse_values / scales
        r2_values = r2_score(y, predicted, multioutput="raw_values")
        results.append(ModelResult(
            name=name,
            score=float(np.mean(norm_values)),
            mae=dict(zip(target_columns, map(float, mae_values))),
            rmse=dict(zip(target_columns, map(float, rmse_values))),
            normalized_rmse=dict(zip(target_columns, map(float, norm_values))),
            r2=dict(zip(target_columns, map(float, r2_values))),
            predictions=predicted,
        ))

    if not results:
        raise RuntimeError("No candidate model could be trained successfully.")
    return sorted(results, key=lambda result: (result.score, _complexity_rank(result.name)))


def _make_splitter(n_rows: int, validation: str):
    if validation == "loo":
        return _GroupsIgnoringSplitter(LeaveOneOut())
    if validation == "logo":
        return LeaveOneGroupOut()
    n_splits = min(5, n_rows)
    return _GroupsIgnoringSplitter(KFold(n_splits=n_splits, shuffle=True, random_state=42))


class _GroupsIgnoringSplitter:
    def __init__(self, splitter):
        self._splitter = splitter

    def split(self, X, groups=None):
        return self._splitter.split(X)


def _target_scales(y: np.ndarray) -> np.ndarray:
    ranges = np.ptp(y, axis=0)
    stds = np.std(y, axis=0)
    scales = np.where(ranges > 0, ranges, stds)
    return np.where(scales > 0, scales, 1.0)


def _complexity_rank(name: str) -> int:
    order = ["kNN", "IDW", "Ridge", "Quadratic", "Random", "Extra", "SVR", "Kernel"]
    for idx, marker in enumerate(order):
        if marker in name:
            return idx
    return 99

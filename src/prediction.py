from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from .constants import TARGET_COLUMNS
from .data_loader import Dataset
from .evaluation import ModelResult, compare_models
from .grouping import GROUP_FEATURE_PREFIX, infer_subgroup
from .models import candidate_models


MIN_GROUP_ROWS_FOR_UNCERTAINTY = 5


@dataclass
class TrainedPredictor:
    dataset: Dataset
    model_name: str
    model: object
    results: list[ModelResult]
    scaler: StandardScaler
    # {subgroup or "__global__": {target: (median |error|, 95th percentile |error|, n)}}
    uncertainty: dict[str, dict[str, tuple[float, float, int]]] | None = None


def train_predictor(
    dataset: Dataset,
    validation: str = "loo",
    model_name: str | None = None,
    model_set: str = "expanded",
) -> TrainedPredictor:
    n_group_cols = sum(1 for c in dataset.feature_columns if c.startswith(GROUP_FEATURE_PREFIX))
    models = candidate_models(len(dataset.frame), model_set=model_set, n_group_cols=n_group_cols)
    groups = dataset.frame["Subgroup"].to_numpy() if "Subgroup" in dataset.frame.columns else None
    results = compare_models(
        models, dataset.X, dataset.y, dataset.target_columns, validation=validation, groups=groups
    )
    selected_name = model_name or results[0].name
    if selected_name not in models:
        available = ", ".join(models)
        raise ValueError(f"Unknown model '{selected_name}'. Available models: {available}")

    model = clone(models[selected_name])
    model.fit(dataset.X, dataset.y)
    scaler = StandardScaler().fit(dataset.X)
    selected_result = next((r for r in results if r.name == selected_name), None)
    return TrainedPredictor(
        dataset=dataset,
        model_name=selected_name,
        model=model,
        results=results,
        scaler=scaler,
        uncertainty=_residual_quantiles(dataset, selected_result),
    )


def _residual_quantiles(dataset: Dataset, result: ModelResult | None):
    """Per-subgroup |error| quantiles of the selected model's cross-validation."""
    if result is None or result.predictions is None:
        return None
    errors = np.abs(result.predictions - dataset.y)
    groups = dataset.frame["Subgroup"].to_numpy()
    quantiles: dict[str, dict[str, tuple[float, float, int]]] = {}

    def stats(rows: np.ndarray) -> dict[str, tuple[float, float, int]]:
        n = len(rows)
        # small-sample correction: target the ceil(0.95*(n+1))-th order statistic
        # so nominal 95% coverage holds for small subgroups (else ~89-91%)
        level = min(100.0, 95.0 * (n + 1) / n)
        return {
            target: (
                float(np.median(rows[:, j])),
                float(np.percentile(rows[:, j], level, method="higher")),
                n,
            )
            for j, target in enumerate(dataset.target_columns)
        }

    quantiles["__global__"] = stats(errors)
    for group in set(groups):
        mask = groups == group
        if mask.sum() >= MIN_GROUP_ROWS_FOR_UNCERTAINTY:
            quantiles[group] = stats(errors[mask])
    return quantiles


def predict_properties(predictor: TrainedPredictor, vector: list[float], k_nearest: int = 5) -> dict:
    x = np.asarray(vector, dtype=float).reshape(1, -1)
    exact = _find_exact_match(predictor.dataset, vector)
    if exact is not None:
        prediction = exact[TARGET_COLUMNS].to_dict()
        source = "exact-match"
    else:
        raw = predictor.model.predict(x)[0]
        prediction = dict(zip(TARGET_COLUMNS, map(float, raw)))
        source = "model"

    nearest = nearest_rows(predictor, vector, k=k_nearest)
    distance = nearest[0]["distance"] if nearest else float("inf")
    element_columns = [
        c for c in predictor.dataset.feature_columns if not c.startswith(GROUP_FEATURE_PREFIX)
    ]
    subgroup = infer_subgroup(dict(zip(element_columns, vector)))
    return {
        "source": source,
        "model": predictor.model_name,
        "training_rows": len(predictor.dataset.frame),
        "inferred_subgroup": subgroup,
        "domain_status": classify_domain(distance),
        "nearest_distance": distance,
        "prediction": prediction,
        "uncertainty": _lookup_uncertainty(predictor, subgroup, classify_domain(distance)),
        "nearest": nearest,
    }


def _lookup_uncertainty(predictor: TrainedPredictor, subgroup: str, domain: str) -> dict:
    """Empirical |error| quantiles for the phase's subgroup (global fallback)."""
    if not predictor.uncertainty:
        return {}
    basis = subgroup if subgroup in predictor.uncertainty else "__global__"
    entry = predictor.uncertainty[basis]
    result = {
        "basis": "subgroup" if basis == subgroup else "global",
        "n_reference_rows": next(iter(entry.values()))[2] if entry else 0,
        "median_abs_error": {t: v[0] for t, v in entry.items()},
        "p95_abs_error": {t: v[1] for t, v in entry.items()},
    }
    if domain == "out-of-domain":
        result["caveat"] = (
            "out-of-domain input: intervals are based on in-domain cross-validation "
            "residuals and likely underestimate the true error"
        )
    return result


def nearest_rows(predictor: TrainedPredictor, vector: list[float], k: int = 5) -> list[dict]:
    x_scaled = predictor.scaler.transform(np.asarray(vector, dtype=float).reshape(1, -1))
    X_scaled = predictor.scaler.transform(predictor.dataset.X)
    distances = np.linalg.norm(X_scaled - x_scaled, axis=1)
    order = np.argsort(distances)[:k]
    rows: list[dict] = []
    for idx in order:
        row = predictor.dataset.frame.iloc[int(idx)]
        item = {
            "distance": float(distances[idx]),
            "Group": str(row.get("Group", "")),
            "Subgroup": str(row.get("Subgroup", "")),
            "Name": str(row.get("Name", "")),
            "Chemical_formula": str(row.get("Chemical_formula", "")),
        }
        for target in TARGET_COLUMNS:
            item[target] = float(row[target])
        rows.append(item)
    return rows


def classify_domain(distance: float) -> str:
    if distance <= 0.15:
        return "in-domain"
    if distance <= 1.5:
        return "near-domain"
    return "out-of-domain"


def _find_exact_match(dataset: Dataset, vector: list[float], tolerance: float = 1e-9) -> pd.Series | None:
    x = np.asarray(vector, dtype=float)
    matches = np.all(np.isclose(dataset.X, x, atol=tolerance, rtol=0.0), axis=1)
    indexes = np.flatnonzero(matches)
    if len(indexes) == 0:
        return None
    return dataset.frame.iloc[int(indexes[0])]

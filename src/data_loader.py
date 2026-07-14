from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DEFAULT_FEATURE_COLUMNS, DERIVED_COLUMNS, META_COLUMNS, TARGET_COLUMNS
from .grouping import SUBGROUPS, infer_subgroup, subgroup_feature_columns, subgroup_to_feature_column
from .io_utils import read_csv_with_fallback


@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_columns: list[str]

    @property
    def X(self) -> np.ndarray:
        return self.frame[self.feature_columns].to_numpy(dtype=float)

    @property
    def y(self) -> np.ndarray:
        return self.frame[self.target_columns].to_numpy(dtype=float)


def load_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = read_csv_with_fallback(source)
    elif suffix in {".xlsx", ".xlsm", ".xls"}:
        frame = pd.read_excel(source, sheet_name=0)
    else:
        raise ValueError(f"Unsupported data file extension: {source.suffix}")

    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].replace("-", np.nan)
    return frame


def detect_feature_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [c for c in DEFAULT_FEATURE_COLUMNS if c in frame.columns]
    if preferred:
        return preferred

    excluded = META_COLUMNS | DERIVED_COLUMNS | set(TARGET_COLUMNS)
    candidates: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            candidates.append(column)
    return candidates


def prepare_dataset(
    path: str | Path,
    feature_columns: list[str] | None = None,
    group_features: bool = True,
) -> Dataset:
    frame = load_table(path)
    features = feature_columns or detect_feature_columns(frame)

    missing_features = [c for c in features if c not in frame.columns]
    missing_targets = [c for c in TARGET_COLUMNS if c not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {', '.join(missing_features)}")
    if missing_targets:
        raise ValueError(f"Missing target columns: {', '.join(missing_targets)}")

    for column in features + TARGET_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    valid_mask = frame[features + TARGET_COLUMNS].notna().all(axis=1)
    valid = frame.loc[valid_mask].reset_index(drop=True)
    if len(valid) < 3:
        raise ValueError("At least 3 valid rows are required for training.")

    for column in ["Group", "Subgroup", "Name", "Chemical_formula"]:
        if column not in valid.columns:
            valid[column] = ""
        valid[column] = valid[column].fillna("").astype(str)

    if group_features:
        valid = _add_group_features(valid, features)
        features = features + subgroup_feature_columns()

    return Dataset(frame=valid, feature_columns=features, target_columns=TARGET_COLUMNS)


def _add_group_features(frame: pd.DataFrame, element_columns: list[str]) -> pd.DataFrame:
    """Attach one-hot subgroup columns, inferring the subgroup where missing."""
    frame = frame.copy()
    subgroups: list[str] = []
    for _, row in frame.iterrows():
        stored = str(row.get("Subgroup", "")).strip()
        if stored in SUBGROUPS:
            subgroups.append(stored)
        else:
            composition = {column: float(row[column]) for column in element_columns}
            subgroups.append(infer_subgroup(composition))
    frame["Subgroup"] = subgroups
    for name in SUBGROUPS:
        column = subgroup_to_feature_column(name)
        frame[column] = [1.0 if subgroup == name else 0.0 for subgroup in subgroups]
    return frame

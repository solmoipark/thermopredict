from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import TARGET_COLUMNS
from .features import parse_composition, parse_formula, vectorize_composition
from .io_utils import read_csv_with_fallback
from .prediction import predict_properties

# common header variants (including mojibake of Fe³⁺/Fe²⁺/S²⁻) mapped to feature names
COLUMN_ALIASES = {
    "Fe3+": "Fe3", "Fe³⁺": "Fe3", "Fe(III)": "Fe3", "FeIII": "Fe3", "Fe???": "Fe3",
    "Fe2+": "Fe2", "Fe²⁺": "Fe2", "Fe(II)": "Fe2", "FeII": "Fe2", "Fe???.1": "Fe2",
    "S2-": "S2", "S²⁻": "S2", "S???": "S2",
}

SHORT_TARGET_NAMES = {
    "V0d_cm3_per_mol": "V0d",
    "G0d_J_per_mol": "G0d",
    "H0d_J_per_mol": "H0d",
    "S0d_J_per_K_mol": "S0d",
    "Cp0d_J_per_K_mol": "Cp0d",
}


def load_input_table(input_file: str | Path) -> pd.DataFrame:
    source = Path(input_file)
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")

    if source.suffix.lower() == ".csv":
        frame = read_csv_with_fallback(source)
    elif source.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        frame = pd.read_excel(source)
    else:
        raise ValueError(f"Unsupported input file extension: {source.suffix}")
    return normalize_input_columns(frame)


def normalize_input_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renames = {c: COLUMN_ALIASES[c] for c in frame.columns if c in COLUMN_ALIASES}
    if renames:
        frame = frame.rename(columns=renames)
    return frame


def predict_input_frame(predictor, inputs: pd.DataFrame) -> pd.DataFrame:
    records = []
    for index, row in inputs.iterrows():
        if row.isna().all() or all(str(value).strip() == "" for value in row.dropna().tolist()):
            continue
        input_record = row.to_dict()
        output_record = dict(input_record)
        output_record["input_row"] = int(index)
        try:
            composition = composition_from_input_row(row, predictor.dataset.feature_columns)
            vector, unknown = vectorize_composition(composition, predictor.dataset.feature_columns)
            result = predict_properties(predictor, vector)
        except Exception as error:
            output_record["prediction_error"] = str(error)
            records.append(output_record)
            continue

        output_record["parsed_composition"] = format_composition(composition)
        output_record["unknown_elements"] = ", ".join(unknown)
        output_record["prediction_error"] = ""
        for column, value in zip(predictor.dataset.feature_columns, vector):
            output_record[f"input_{column}"] = value
        output_record["prediction_source"] = result["source"]
        output_record["inferred_subgroup"] = result.get("inferred_subgroup", "")
        output_record["selected_model"] = result["model"]
        output_record["training_rows"] = result["training_rows"]
        output_record["domain_status"] = result["domain_status"]
        output_record["nearest_distance"] = result["nearest_distance"]
        uncertainty = result.get("uncertainty") or {}
        p95 = uncertainty.get("p95_abs_error", {})
        for target, value in result["prediction"].items():
            output_record[f"predicted_{target}"] = value
            output_record[SHORT_TARGET_NAMES.get(target, target)] = value
            if target in p95:
                output_record[f"p95_{SHORT_TARGET_NAMES.get(target, target)}"] = p95[target]
        if uncertainty:
            output_record["uncertainty_basis"] = uncertainty["basis"]

        nearest = result["nearest"][0] if result["nearest"] else {}
        output_record["nearest_Group"] = nearest.get("Group", "")
        output_record["nearest_Subgroup"] = nearest.get("Subgroup", "")
        output_record["nearest_Name"] = nearest.get("Name", "")
        output_record["nearest_Chemical_formula"] = nearest.get("Chemical_formula", "")
        records.append(output_record)

    return pd.DataFrame(records)


def predict_input_file(predictor, input_file: str | Path, output_path: str | Path) -> pd.DataFrame:
    inputs = load_input_table(input_file)
    output = predict_input_frame(predictor, inputs)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output


def composition_from_input_row(row: pd.Series, feature_columns: list[str]) -> dict[str, float]:
    formula_errors: list[str] = []
    for column in ["Formula", "formula", "Chemical_formula", "chemical_formula"]:
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            try:
                return parse_formula(str(value))
            except ValueError as error:
                formula_errors.append(f"{column}: {error}")
                break

    for column in ["Composition", "composition"]:
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            try:
                return parse_composition(str(value))
            except ValueError as error:
                formula_errors.append(f"{column}: {error}")
                break

    composition = {}
    for column in feature_columns:
        value = row.get(column)
        if pd.notna(value) and str(value).strip() != "":
            composition[column] = float(value)
    if composition:
        return composition
    if formula_errors:
        raise ValueError("; ".join(formula_errors))
    raise ValueError("Input row must contain Formula, Composition, or element columns.")


def format_composition(composition: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:g}" for key, value in sorted(composition.items()))

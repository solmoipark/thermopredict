"""Rebuild the master training database from Cemdata18 (ThermoFun export).

Reads data/sources/cemdata18-thermofun.json (retrieved 2026-07-12 from
https://raw.githubusercontent.com/thermohub/thermofun-jupyter/master/applications/cemdata18-thermofun.json),
extracts all crystalline phases, and merges legacy rows that are not present
in Cemdata18. Run from the repository root:

    py scripts/build_database.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.constants import DEFAULT_FEATURE_COLUMNS, TARGET_COLUMNS
from src.features import parse_formula
from src.grouping import infer_subgroup
from src.io_utils import read_csv_with_fallback

SOURCE_JSON = ROOT / "data" / "sources" / "cemdata18-thermofun.json"
OUTPUT_CSV = ROOT / "data" / "selected_thermodynamic_properties.csv"
# immutable input snapshot: the pre-rebuild master (with 2026-07-12 corrections)
LEGACY_SNAPSHOT = ROOT / "data" / "sources" / "legacy_master_2026-07-12.csv"
CONFLICT_LOG = ROOT / "data" / "sources" / "value_conflicts.csv"
# curated literature data typed from published tables; overrides Cemdata18
ZEOLITE_CSV = ROOT / "data" / "sources" / "ma_lothenbach_zeolites.csv"
LITERATURE_CSV = ROOT / "data" / "sources" / "curated_ldh_nash.csv"
ADDITIONS_CSV = ROOT / "data" / "sources" / "curated_additions.csv"
HIRSCH_CSV = ROOT / "data" / "sources" / "hirsch2024_csh_crystalline.csv"

ZEOLITE_DOI = {
    "106111": "https://doi.org/10.1016/j.cemconres.2020.106111",
    "106537": "https://doi.org/10.1016/j.cemconres.2021.106537",
    "106149": "https://doi.org/10.1016/j.cemconres.2020.106149",
}

ATOMIC_MASS = {
    "Ca": 40.078, "Si": 28.0855, "Al": 26.9815, "O": 15.999, "H": 1.008,
    "Na": 22.9898, "Fe": 55.845, "Fe3": 55.845, "Fe2": 55.845, "S": 32.06,
    "S2": 32.06, "C": 12.011, "Cl": 35.453, "K": 39.0983, "Mg": 24.305, "N": 14.007,
    "Sr": 87.62,
}

CEMDATA_CITATION = "Cemdata18 (Lothenbach et al. 2019, CCR 115, 472-506) via ThermoHub"

ELEMENT_TOLERANCE = 0.02


def parse_thermofun_formula(formula: str) -> dict[str, float]:
    """Parse a ThermoFun formula; oxidation states like Fe|3| map to Fe3/Fe2."""
    text = formula.strip()
    # tag oxidation-state-specific columns before stripping the |n| markers
    # (placeholders must look like valid two-letter element symbols)
    text = text.replace("Fe|3|", "Xq").replace("Fe|2|", "Xr").replace("S|-2|", "Xs")
    text = re.sub(r"\|-?\d+\|", "", text)  # N|5|, S|6|, ... -> plain element
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("[", "(").replace("]", ")")
    text = text.replace(":", "+")  # e.g. (SO4)3:Ca6Al2(OH)12(H2O)26
    composition = parse_formula(text)
    placeholder_map = {"Xq": "Fe3", "Xr": "Fe2", "Xs": "S2"}
    result: dict[str, float] = {}
    for key, value in composition.items():
        key = placeholder_map.get(key, key)
        result[key] = result.get(key, 0.0) + value
    return result


def scalar(record: dict, field: str) -> float | None:
    entry = record.get(field)
    if not entry:
        return None
    values = entry.get("values")
    if not values:
        return None
    return float(values[0])


def cemdata_rows() -> list[dict]:
    db = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    rows: list[dict] = []
    skipped: list[str] = []
    for record in db["substances"]:
        aggregate = json.dumps(record.get("aggregate_state", {}))
        if "AS_CRYSTAL" not in aggregate:
            continue
        try:
            composition = parse_thermofun_formula(record["formula"])
        except ValueError as error:
            skipped.append(f"{record['symbol']}: {error}")
            continue
        unknown = [e for e in composition if e not in DEFAULT_FEATURE_COLUMNS]
        if unknown:
            skipped.append(f"{record['symbol']}: unsupported elements {unknown}")
            continue

        volume_j_bar = scalar(record, "sm_volume")
        datasources = ", ".join(
            d.strip() for d in record.get("datasources", []) if d.strip() and d.strip() != "`"
        )
        row = {
            "Group": "Cemdata18",
            "Name": str(record.get("name", record["symbol"])).strip(),
            "Symbol": record["symbol"],
            "Chemical_formula": record["formula"],
            **{e: composition.get(e, 0.0) for e in DEFAULT_FEATURE_COLUMNS},
            "M0_g_per_mol": scalar(record, "mass_per_mole"),
            "V0d_cm3_per_mol": volume_j_bar * 10.0 if volume_j_bar is not None else None,
            "G0d_J_per_mol": scalar(record, "sm_gibbs_energy"),
            "H0d_J_per_mol": scalar(record, "sm_enthalpy"),
            "S0d_J_per_K_mol": scalar(record, "sm_entropy_abs"),
            "Cp0d_J_per_K_mol": scalar(record, "sm_heat_capacity_p"),
            "Remark": "",
            "Ref": f"{CEMDATA_CITATION}; sources: {datasources}" if datasources else CEMDATA_CITATION,
        }
        row["Subgroup"] = infer_subgroup(composition)
        rows.append(row)

    if skipped:
        print("Skipped substances:")
        for item in skipped:
            print(f"  {item}")
    return rows


def curated_rows(csv_path: Path, group_label: str) -> list[dict]:
    frame = read_csv_with_fallback(csv_path)
    rows: list[dict] = []
    for _, record in frame.iterrows():
        composition = parse_thermofun_formula(str(record["Chemical_formula"]))
        ref = str(record["Ref"])
        doi = next((doi for key, doi in ZEOLITE_DOI.items() if key in ref), "")
        rows.append({
            "Group": group_label,
            "Name": record["Name"],
            "Symbol": "",
            "Chemical_formula": record["Chemical_formula"],
            **{e: composition.get(e, 0.0) for e in DEFAULT_FEATURE_COLUMNS},
            "M0_g_per_mol": sum(ATOMIC_MASS[e] * n for e, n in composition.items()),
            "V0d_cm3_per_mol": float(record["V0d_cm3_per_mol"]),
            "G0d_J_per_mol": float(record["G0d_J_per_mol"]),
            "H0d_J_per_mol": float(record["H0d_J_per_mol"]),
            "S0d_J_per_K_mol": float(record["S0d_J_per_K_mol"]),
            "Cp0d_J_per_K_mol": float(record["Cp0d_J_per_K_mol"]),
            "Remark": "",
            "Ref": f"{ref} ({doi})" if doi else ref,
            "Subgroup": infer_subgroup(composition),
        })
    return rows


def apply_curated_overrides(new_rows: list[dict], csv_path: Path, group_label: str) -> list[dict]:
    """Replace Cemdata18 estimates with curated experimental literature data."""
    matrix = np.array([matching_vector(r) for r in new_rows])
    replaced = 0
    added = 0
    for row in curated_rows(csv_path, group_label):
        vector = matching_vector(row)
        distances = np.abs(matrix - vector).max(axis=1)
        match = int(np.argmin(distances))
        if distances[match] <= ELEMENT_TOLERANCE and new_rows[match]["Group"] == "Cemdata18":
            new_rows[match] = row
            replaced += 1
        else:
            new_rows.append(row)
            added += 1
    print(f"{group_label}: replaced {replaced} Cemdata18 records, added {added}")
    return new_rows


def element_vector(row: dict | pd.Series) -> np.ndarray:
    values = []
    for column in DEFAULT_FEATURE_COLUMNS:
        value = row.get(column)
        try:
            values.append(0.0 if value is None or pd.isna(value) else float(value))
        except (TypeError, ValueError):
            values.append(0.0)
    return np.array(values)


_FE_INDEXES = [DEFAULT_FEATURE_COLUMNS.index(c) for c in ("Fe", "Fe3", "Fe2")]
_NON_FE_INDEXES = [i for i in range(len(DEFAULT_FEATURE_COLUMNS)) if i not in _FE_INDEXES]


def matching_vector(row: dict | pd.Series) -> np.ndarray:
    """Composition vector with iron oxidation states collapsed, for phase matching.

    Sources disagree on whether ferric iron is stored in Fe or Fe3, so matching
    identical phases across sources must compare total iron.
    """
    vector = element_vector(row)
    fe_total = vector[_FE_INDEXES].sum()
    return np.append(vector[_NON_FE_INDEXES], fe_total)


def apply_quality_rules(rows: list[dict]) -> list[dict]:
    """Fix known legacy data-entry errors and null out placeholder values.

    - Legacy hydrotalcite rows carry 10x magnitude typos in G0/H0 (no phase in
      this chemical space has |G0| or |H0| above 5e7 J/mol; correct values of
      the sibling hydration states are ~9e6 J/mol).
    - The legacy Water row carried corrupted properties; standard-state liquid
      water values from the Cemdata18 ThermoFun H2O@ record (SUPCRT92) are used.
    - Cemdata18 reaction-derived precipitate records (_rdc_) use V0=0, Cp0=0 or
      negative S0 as placeholders; these are set to NaN so the affected rows are
      excluded from training instead of poisoning it.
    """
    water_targets = {
        "V0d_cm3_per_mol": 18.068,
        "G0d_J_per_mol": -237183.0,
        "H0d_J_per_mol": -285881.0,
        "S0d_J_per_K_mol": 69.923,
        "Cp0d_J_per_K_mol": 75.361,
    }
    fixed_scale = 0
    for row in rows:
        for target in ("G0d_J_per_mol", "H0d_J_per_mol"):
            value = pd.to_numeric(row.get(target), errors="coerce")
            if pd.notna(value) and value < -5e7:
                row[target] = float(value) / 10.0
                row["Remark"] = (str(row.get("Remark", "") or "") + " 10x magnitude typo corrected").strip()
                fixed_scale += 1
        if str(row.get("Name", "")).strip() == "Water":
            row.update(water_targets)
            row["Ref"] = "SUPCRT92 liquid water (via Cemdata18 ThermoFun H2O@ record)"
        for target, bad in (
            ("V0d_cm3_per_mol", lambda v: v == 0),
            ("Cp0d_J_per_K_mol", lambda v: v == 0),
            ("S0d_J_per_K_mol", lambda v: v <= 0),
        ):
            value = pd.to_numeric(row.get(target), errors="coerce")
            if pd.notna(value) and bad(float(value)):
                row[target] = np.nan
    print(f"Scale corrections applied: {fixed_scale}")
    return rows


def main() -> None:
    legacy = read_csv_with_fallback(LEGACY_SNAPSHOT)
    new_rows = cemdata_rows()
    print(f"Cemdata18 crystalline phases parsed: {len(new_rows)}")
    new_rows = apply_curated_overrides(new_rows, ZEOLITE_CSV, "Ma-Lothenbach-zeolites")
    new_rows = apply_curated_overrides(new_rows, LITERATURE_CSV, "Curated-literature")
    new_rows = apply_curated_overrides(new_rows, ADDITIONS_CSV, "Curated-additions")
    new_rows = apply_curated_overrides(new_rows, HIRSCH_CSV, "Hirsch-Lothenbach-2024")

    new_matrix = np.array([matching_vector(r) for r in new_rows])
    kept_legacy = 0
    conflicts: list[dict] = []
    for _, row in legacy.iterrows():
        vector = matching_vector(row)
        distances = np.abs(new_matrix - vector).max(axis=1)
        match = int(np.argmin(distances))
        if distances[match] <= ELEMENT_TOLERANCE:
            # same composition: keep the Cemdata18 record, log value disagreements
            # (composition-identical polymorphs are covered by Cemdata18's own records)
            for target in TARGET_COLUMNS:
                legacy_value = pd.to_numeric(row.get(target), errors="coerce")
                new_value = new_rows[match][target]
                if pd.notna(legacy_value) and new_value is not None:
                    scale = max(abs(new_value), 1.0)
                    if abs(legacy_value - new_value) / scale > 0.02:
                        conflicts.append({
                            "legacy_Name": row["Name"],
                            "cemdata_Symbol": new_rows[match]["Symbol"],
                            "target": target,
                            "legacy_value": legacy_value,
                            "cemdata18_value": new_value,
                        })
            continue
        entry = row.to_dict()
        existing_ref = str(entry.get("Ref", "") or "").strip()
        if str(entry.get("Ref", "")) in ("nan", ""):
            entry["Ref"] = "legacy dataset (unverified origin; predates 2026-07 rebuild)"
        else:
            entry["Ref"] = existing_ref
        entry["Symbol"] = ""
        composition = dict(zip(DEFAULT_FEATURE_COLUMNS, vector))
        entry["Subgroup"] = infer_subgroup(composition)
        new_rows.append(entry)
        kept_legacy += 1

    print(f"Legacy rows kept (not in Cemdata18): {kept_legacy}")
    if conflicts:
        pd.DataFrame(conflicts).to_csv(CONFLICT_LOG, index=False, encoding="utf-8-sig")
        print(f"Value conflicts (>2%): {len(conflicts)} -> {CONFLICT_LOG}")

    new_rows = apply_quality_rules(new_rows)

    frame = pd.DataFrame(new_rows)
    ordered = [
        "Group", "Subgroup", "Name", "Symbol", "Chemical_formula",
        *DEFAULT_FEATURE_COLUMNS,
        "M0_g_per_mol", *TARGET_COLUMNS, "Remark", "Ref",
    ]
    frame = frame[[c for c in ordered if c in frame.columns]]
    frame = frame.sort_values(["Subgroup", "Name"]).reset_index(drop=True)
    frame.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(frame)} rows -> {OUTPUT_CSV}")
    print(frame["Subgroup"].value_counts().to_string())


if __name__ == "__main__":
    main()

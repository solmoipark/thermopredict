from __future__ import annotations

import re
from collections import defaultdict

from .grouping import GROUP_FEATURE_PREFIX, infer_subgroup, subgroup_to_feature_column


SUBSCRIPT_MAP = str.maketrans({
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "₊": "+", "₋": "-", "₍": "(", "₎": ")",
})


def parse_composition(text: str) -> dict[str, float]:
    """Parse direct input such as 'Ca=1.5, Si=1, O=6, H=5'."""
    result: dict[str, float] = {}
    chunks = re.split(r"[,;\s]+", text.strip())
    for chunk in chunks:
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Expected key=value near '{chunk}'")
        key, raw = chunk.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Z][a-z]?[0-9+-]*", key):
            raise ValueError(f"Invalid element key: {key}")
        result[key] = float(raw)
    return result


def parse_formula(text: str) -> dict[str, float]:
    """Parse formulas like '(CaO)1.5(SiO2)(H2O)2.5' or '2.22SiO2·5.51H2O'."""
    formula = _normalize_formula(text)
    if not formula:
        raise ValueError("Formula is empty.")
    values, pos = _parse_group(formula, 0)
    if pos != len(formula):
        raise ValueError(f"Could not parse formula near '{formula[pos:]}'")
    return dict(values)


def vectorize_composition(composition: dict[str, float], feature_columns: list[str]) -> tuple[list[float], list[str]]:
    element_columns = [c for c in feature_columns if not c.startswith(GROUP_FEATURE_PREFIX)]
    unknown = sorted(k for k in composition if k not in element_columns)

    group_column = None
    if len(element_columns) < len(feature_columns):
        group_column = subgroup_to_feature_column(infer_subgroup(composition))

    vector: list[float] = []
    for column in feature_columns:
        if column.startswith(GROUP_FEATURE_PREFIX):
            vector.append(1.0 if column == group_column else 0.0)
        else:
            vector.append(float(composition.get(column, 0.0)))
    return vector, unknown


def _parse_group(text: str, pos: int) -> tuple[defaultdict[str, float], int]:
    totals: defaultdict[str, float] = defaultdict(float)
    while pos < len(text):
        char = text[pos]
        if char in "+·∙•⋅":
            pos += 1
            continue
        if char == ")":
            return totals, pos + 1
        prefix = 1.0
        if char.isdigit() or char == ".":
            prefix, pos = _parse_number(text, pos)
            if pos >= len(text):
                raise ValueError("Formula ended after a coefficient.")
            char = text[pos]
        if char == "(":
            nested, pos = _parse_group(text, pos + 1)
            multiplier, pos = _parse_number(text, pos)
            for key, value in nested.items():
                totals[key] += value * prefix * multiplier
            continue

        match = re.match(r"[A-Z][a-z]?", text[pos:])
        if not match:
            raise ValueError(f"Expected element symbol near '{text[pos:]}'")
        element = match.group(0)
        pos += len(element)
        amount, pos = _parse_number(text, pos)
        totals[element] += prefix * amount
    return totals, pos


def _parse_number(text: str, pos: int) -> tuple[float, int]:
    match = re.match(r"(?:\d+(?:\.\d*)?|\.\d+)", text[pos:])
    if not match:
        return 1.0, pos
    raw = match.group(0)
    return float(raw), pos + len(raw)


def _normalize_formula(text: str) -> str:
    formula = text.translate(SUBSCRIPT_MAP)
    formula = formula.replace(" ", "")
    for separator in ["·", "∙", "•", "⋅", "＊", "*"]:
        formula = formula.replace(separator, "+")
    return formula

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .batch import predict_input_file
from .data_loader import prepare_dataset
from .features import parse_composition, parse_formula, vectorize_composition
from .prediction import predict_properties, train_predictor


DEFAULT_DATA = Path("data/selected_thermodynamic_properties.csv")


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(
        prog="thermo-predict",
        description="Train and run a Python thermodynamic property predictor for cement hydrates.",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="CSV or Excel training data path.")
    parser.add_argument(
        "--validation",
        choices=["loo", "kfold", "logo"],
        default="loo",
        help="Validation protocol. logo = leave-one-subgroup-out (extrapolation test).",
    )
    parser.add_argument(
        "--model-set",
        choices=["fast", "expanded", "full"],
        default="expanded",
        help="Candidate model set. full is slower, especially with leave-one-out.",
    )
    parser.add_argument("--model", default=None, help="Optional model name. Defaults to best validation score.")
    parser.add_argument("--formula", help="Formula input, e.g. '(CaO)1.5(SiO2)(H2O)2.5'.")
    parser.add_argument("--composition", help="Direct composition, e.g. 'Ca=1.5,Si=1,O=6,H=5'.")
    parser.add_argument("--input-file", help="CSV or Excel file containing Formula, Composition, or element columns.")
    parser.add_argument("--output", help="CSV path for combined input and prediction results.")
    parser.add_argument("--train-only", action="store_true", help="Only print model comparison results.")
    parser.add_argument(
        "--no-group-features",
        action="store_true",
        help="Disable subgroup one-hot features (composition columns only).",
    )
    parser.add_argument("--json", action="store_true", help="Print prediction as JSON.")
    args = parser.parse_args()

    dataset = prepare_dataset(args.data, group_features=not args.no_group_features)
    predictor = train_predictor(
        dataset,
        validation=args.validation,
        model_name=args.model,
        model_set=args.model_set,
    )

    _print_training_summary(predictor)

    if args.train_only:
        return

    if args.input_file:
        output_path = Path(args.output or "prediction_results.csv")
        output = predict_input_file(predictor, args.input_file, output_path)
        print(f"\nSaved prediction results: {output_path}")
        _print_output_preview(output)
        return

    if args.formula:
        composition = parse_formula(args.formula)
    elif args.composition:
        composition = parse_composition(args.composition)
    else:
        composition = {"Ca": 1.5, "Si": 1.0, "O": 4.0, "H": 5.0}
        print("\nNo input composition was provided; using example Ca=1.5, Si=1, O=4, H=5.")

    vector, unknown = vectorize_composition(composition, dataset.feature_columns)
    if unknown:
        print(f"\nWarning: ignored unknown elements not present in training features: {', '.join(unknown)}")

    result = predict_properties(predictor, vector)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_prediction(result, dataset.feature_columns, vector)


def _print_output_preview(output: pd.DataFrame) -> None:
    columns = [
        "Sample_ID",
        "Formula",
        "parsed_composition",
        "selected_model",
        "domain_status",
        "nearest_distance",
        "predicted_V0d_cm3_per_mol",
        "predicted_G0d_J_per_mol",
        "predicted_H0d_J_per_mol",
        "predicted_S0d_J_per_K_mol",
        "predicted_Cp0d_J_per_K_mol",
        "nearest_Name",
    ]
    existing = [column for column in columns if column in output.columns]
    print("\nOutput preview")
    print(output[existing].to_string(index=False))


def _print_training_summary(predictor) -> None:
    print(f"Loaded training rows: {len(predictor.dataset.frame)}")
    print(f"Feature columns: {', '.join(predictor.dataset.feature_columns)}")
    print(f"Selected model: {predictor.model_name}")
    print("\nModel comparison")
    print(f"{'Rank':>4}  {'Model':<28} {'Score':>10}")
    for idx, result in enumerate(predictor.results, start=1):
        print(f"{idx:>4}  {result.name:<28} {result.score:>10.5f}")


def _print_prediction(result: dict, feature_columns: list[str], vector: list[float]) -> None:
    print("\nInput vector")
    for column, value in zip(feature_columns, vector):
        if value:
            print(f"  {column}: {value:g}")

    print("\nPrediction")
    print(f"  source: {result['source']}")
    if result.get("inferred_subgroup"):
        print(f"  inferred subgroup: {result['inferred_subgroup']}")
    print(f"  model: {result['model']}")
    print(f"  domain: {result['domain_status']} (nearest distance {result['nearest_distance']:.4f})")
    uncertainty = result.get("uncertainty") or {}
    p95 = uncertainty.get("p95_abs_error", {})
    for key, value in result["prediction"].items():
        if key in p95:
            print(f"  {key}: {value:.6g}  (+/- {p95[key]:.3g}, 95% empirical)")
        else:
            print(f"  {key}: {value:.6g}")
    if uncertainty:
        print(
            f"  uncertainty basis: {uncertainty['basis']} residuals "
            f"(n={uncertainty['n_reference_rows']}, cross-validated)"
        )

    print("\nNearest training rows")
    for item in result["nearest"]:
        print(
            f"  d={item['distance']:.4f} | {item['Subgroup'] or item['Group']} | "
            f"{item['Name']} | {item['Chemical_formula']}"
        )


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()

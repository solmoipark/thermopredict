# thermoPredict

Composition-based prediction of standard thermodynamic properties of cement
hydrates. Given only a chemical composition (formula or element moles), the
tool predicts the five standard-state properties at 25 °C and 1 bar:

| Property | Symbol | Unit |
|---|---|---|
| Molar volume | V° | cm³/mol |
| Gibbs energy of formation | ΔfG° | J/mol |
| Enthalpy of formation | ΔfH° | J/mol |
| Absolute entropy | S° | J/(mol·K) |
| Heat capacity | Cp° | J/(mol·K) |

Each prediction is returned with an empirical uncertainty (median and 95th
percentile of cross-validation errors for the phase's chemical subgroup) and
an applicability-domain flag based on the distance to the nearest training
phase.

## How it works

1. The input composition is parsed into 17 element mole numbers
   (Ca, Si, Al, O, H, Na, Fe, Fe(III), Fe(II), S, S(-II), C, Cl, K, Mg, N, Sr).
2. A rule-based classifier assigns one of 21 cement-chemistry subgroups
   (C-S-H, C-A-S-H, C-A-H, AFt, AFm, hydrogarnet, LDH, M-S-H, zeolite/N-A-S-H,
   alkali silicate hydrate, carbonate, sulfate, clinker, ...) from element
   ratios alone, so the same procedure applies to structurally
   uncharacterized phases.
3. A two-stage model — ordinary least squares on the composition (a
   database-calibrated oxide-additivity model) plus a per-subgroup residual
   correction — predicts the five properties. The model is selected
   automatically among ~25 candidate regressors by leave-one-out
   cross-validation.
4. If the input composition exactly matches a database phase, the database
   values are returned instead of a model estimate.

## Database

`data/selected_thermodynamic_properties.csv` contains 222 cement-relevant
phases with per-record literature sources. It is rebuilt deterministically
from the immutable inputs in `data/sources/` by:

```powershell
py scripts/build_database.py
```

Sources: Cemdata18 (ThermoFun export via ThermoHub; 134 records), zeolites of
Ma & Lothenbach (2020, 2021), hydrotalcite/pyroaurite end-members of Bernard
et al. (2022, 2025), N-A-S-H gels of Chen et al. (2024), crystalline
calcium-silicate phases of Hirsch & Lothenbach (2024), Mg carbonates from
Thermoddem, and ye'elimite/ternesite derived from measured ΔfH° and S°.
Full citations: [data/REFERENCES.md](data/REFERENCES.md). The build script
enforces quality rules (formula/molar-mass cross-checks, closure screening,
placeholder removal).

## Installation

Python ≥ 3.10 with:

```powershell
py -m pip install -r requirements.txt
```

## Usage

Predict from a chemical formula:

```powershell
py -m src.cli --formula "(CaO)1.5(SiO2)1(H2O)2.5"
```

Predict from element moles:

```powershell
py -m src.cli --composition "Ca=1.5,Si=1,O=6,H=5"
```

Batch prediction from a CSV/Excel file (columns: `Formula`, `Composition`,
or per-element columns; see `data/example_input_compositions.csv`):

```powershell
py -m src.cli --input-file "data\example_input_compositions.csv" --output "results.csv"
```

Streamlit web interface:

```powershell
.\scripts\run_ui.ps1     # opens http://localhost:8501
```

Useful options:

- `--validation {loo,kfold,logo}` — model-selection protocol
  (`logo` = leave-one-subgroup-out, an extrapolation stress test)
- `--model-set {fast,expanded,full}` — size of the candidate model pool
- `--no-group-features` — disable the subgroup one-hot features
- `--json` — machine-readable output

### Example output

```
Prediction
  source: model
  inferred subgroup: C-A-S-H
  domain: in-domain (nearest distance 0.0506)
  V0d_cm3_per_mol:   61.7   (+/- 5.6, 95% empirical)
  G0d_J_per_mol:    -2.258e+06 (+/- 1.3e+04)
  ...
  uncertainty basis: subgroup residuals (n=5, cross-validated)

Nearest training rows
  d=0.0506 | C-A-S-H | CASH-5CA ... | (CaO)1.25(SiO2)1(Al2O3)0.125(H2O)1.625
```

Input headers such as `Fe3+`, `Fe³⁺`, `Fe(III)` are automatically mapped to
the internal `Fe3` column (`Fe2`, `S2` likewise).

## Intended use and limitations

At 25 °C, one logarithmic unit of a solubility product corresponds to
5.71 kJ/mol in ΔfG°. Percent-level accuracy in ΔfG° therefore supports
screening, sensitivity analysis, provisional solid-solution end-members and
starting values for thermodynamic refinement — not direct solubility
prediction. Out-of-domain predictions are flagged and their uncertainty
intervals may understate the true error.

## Code layout

```
src/features.py     formula/composition parsing
src/grouping.py     rule-based subgroup classifier
src/models.py       candidate regressors incl. the additivity models
src/prediction.py   training, prediction, uncertainty, domain flag
src/cli.py          command-line interface
app.py              Streamlit UI
scripts/build_database.py   reproducible database build
```

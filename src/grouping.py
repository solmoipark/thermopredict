from __future__ import annotations

"""Rule-based subgroup classification from elemental composition.

The classifier only uses element mole numbers, so the same rules apply to
training rows and to new user inputs where only the composition is known.
Iron handling: the Fe3/Fe2 columns are oxidation-state specific; the plain Fe
column is used by some records for ferrous or metallic iron. Only Al and Fe3
count as the trivalent (octahedral) cation of AFt/AFm/hydrogarnet/CAH phases,
while Mg-LDH accepts any iron.
"""

SUBGROUPS = [
    "Water",
    "Element",
    "Oxide",
    "Clinker",
    "Simple hydroxide",
    "Silica",
    "Ca sulfate",
    "Alkali sulfate",
    "Sulfide",
    "Carbonate",
    "C-S-H",
    "C-A-S-H",
    "C-A-H",
    "AFt",
    "AFm",
    "Hydrogarnet",
    "LDH",
    "M-S-H",
    "Zeolite",
    "Alkali silicate hydrate",
    "Other",
]

GROUP_FEATURE_PREFIX = "grp_"


def subgroup_feature_columns() -> list[str]:
    return [GROUP_FEATURE_PREFIX + name.replace(" ", "_") for name in SUBGROUPS]


def subgroup_to_feature_column(subgroup: str) -> str:
    return GROUP_FEATURE_PREFIX + subgroup.replace(" ", "_")


def infer_subgroup(composition: dict[str, float]) -> str:
    """Classify a phase into a cement-chemistry subgroup from element moles."""
    get = lambda key: float(composition.get(key, 0.0) or 0.0)
    # Sr substitutes for Ca in cement phases (ECSH, strontianite, celestite):
    # treat it as a Ca-analog for classification.
    ca = get("Ca") + get("Sr")
    si = get("Si")
    al = get("Al")
    fe_plain = get("Fe")
    fe3 = get("Fe3")
    fe2 = get("Fe2")
    fe_all = fe_plain + fe3 + fe2
    mg = get("Mg")
    na = get("Na")
    k = get("K")
    s = get("S")
    s2 = get("S2")
    c = get("C")
    cl = get("Cl")
    n = get("N")
    h = get("H")
    o = get("O")
    alkali = na + k
    trivalent = al + fe3  # octahedral M(III) of AFt/AFm/hydrogarnet/CAH

    # Elemental phases: sulfur, graphite, metallic iron.
    nonzero = [k for k, v in composition.items() if v and float(v) > 0]
    if h == 0 and o == 0 and len(nonzero) == 1:
        return "Element"

    # Sulfides (pyrite, troilite).
    if s2 > 0 and s == 0 and c == 0:
        return "Sulfide"

    # Mg-dominated phases (no mixed Ca-Mg hydrates in cement databases).
    if mg > 0:
        if c > 0 and al + fe_all == 0 and si == 0:
            return "Carbonate"  # magnesite, dolomite
        if si > 0 and (al + fe_all) / si < 0.5:
            return "M-S-H"  # incl. M-A-S-H gels with minor Al
        if al + fe_all > 0:
            return "LDH"
        if si > 0:
            return "M-S-H"
        return "Simple hydroxide"

    # Sulfate-bearing phases.
    if s > 0:
        if h == 0 and ca > 0 and (trivalent > 0 or si > 0):
            return "Clinker"  # ye'elimite, ternesite
        if ca > 0 and trivalent > 0:
            if s / trivalent >= 1.0:
                return "AFt"
            return "AFm"  # monosulfate, Kuzel's salt
        if ca > 0 and h > 0 and (si > 0 or c > 0):
            return "AFt"  # thaumasite
        if ca > 0:
            return "Ca sulfate"
        if alkali > 0:
            return "Alkali sulfate"
        return "Other"  # e.g. melanterite FeSO4·7H2O

    # Zeolites, N-A-S-H and their Fe(III) analogs (N-F-S-H): alkali
    # (alumino/ferri)silicate frameworks.
    if alkali > 0 and si > 0 and (al > 0 or fe3 > 0) and ca == 0:
        return "Zeolite"

    # Nitrate phases: alkali nitrates occur only as zeolitic cages here.
    if n > 0 and alkali > 0:
        return "Zeolite"

    # Chloride phases: Friedel's salt type.
    if cl > 0 and ca > 0 and trivalent > 0:
        return "AFm"

    # Nitrite/nitrate AFm (Balonis-type): interlayer N-anions.
    if n > 0 and ca > 0 and trivalent > 0:
        return "AFm"

    # Carbonate-bearing phases.
    if c > 0:
        if ca > 0 and trivalent > 0:
            if c / trivalent >= 1.0:
                return "AFt"  # tricarboaluminate (CO3-ettringite)
            return "AFm"  # mono-/hemicarbonate
        if (ca > 0 or fe_all > 0) and si == 0 and al == 0:
            return "Carbonate"  # calcite, aragonite, siderite

    # Alkali phases without an aluminosilicate framework.
    if alkali > 0 and ca == 0:
        if si > 0:
            return "Alkali silicate hydrate"
        return "Oxide"

    # Anhydrous Ca phases: clinker minerals (incl. C4AF).
    if h == 0 and ca > 0 and (si > 0 or trivalent > 0):
        return "Clinker"

    # Calcium silicate / aluminate hydrates.
    if ca > 0 and si > 0:
        if trivalent > 0:
            if ca / trivalent >= 1.4 and trivalent / si >= 1.0:
                return "Hydrogarnet"
            if ca / si <= 0.5:
                return "Zeolite"  # Ca-zeolites (chabazite, scolecite, ...)
            if trivalent / si >= 1.0:
                return "AFm"  # straetlingite
            return "C-A-S-H"
        if alkali > 0:
            return "C-A-S-H"  # alkali-substituted C-(N-)S-H
        return "C-S-H"

    if ca > 0 and trivalent > 0:
        return "C-A-H"  # calcium aluminate/ferrite hydrates and hydroxides

    if ca > 0:
        return "Simple hydroxide" if h > 0 else "Oxide"

    if si > 0:
        return "Silica" if h > 0 else "Oxide"

    if al + fe_all > 0:
        return "Simple hydroxide" if h > 0 else "Oxide"

    if h > 0:
        return "Water"

    return "Other"

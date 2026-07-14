# Database references

## Primary source

The master database (`selected_thermodynamic_properties.csv`) is rebuilt by
`scripts/build_database.py` from:

- **Cemdata18**: B. Lothenbach, D.A. Kulik, T. Matschei, M. Balonis, L. Baquerizo,
  B.Z. Dilnesa, G.D. Miron, R.J. Myers, *Cemdata18: A chemical thermodynamic database for
  hydrated Portland cements and alkali-activated materials*, Cement and Concrete Research
  115 (2019) 472–506. https://doi.org/10.1016/j.cemconres.2018.04.018
- Machine-readable ThermoFun export retrieved 2026-07-12 from the official ThermoHub
  repository: `data/sources/cemdata18-thermofun.json`
  (https://github.com/thermohub/thermofun-jupyter, `applications/cemdata18-thermofun.json`).
- **Zeolites (36 rows)**: typed directly from the published tables and stored in
  `data/sources/ma_lothenbach_zeolites.csv` (verified against the PDFs on 2026-07-12).
  These override the older Cemdata18 estimates of the same compositions:
  - Ma, Lothenbach, *Synthesis, characterization, and thermodynamic study of selected
    Na-based zeolites*, CCR 135 (2020) 106111, Table 6.
    https://doi.org/10.1016/j.cemconres.2020.106111
  - Ma, Lothenbach, *Synthesis, characterization, and thermodynamic study of selected
    K-based zeolites*, CCR 148 (2021) 106537, Table 5.
    https://doi.org/10.1016/j.cemconres.2021.106537
  - Ma, Lothenbach, *Thermodynamic study of cement/rock interactions using experimentally
    generated solubility data of zeolites*, CCR 135 (2020) 106149, Table 4.
    https://doi.org/10.1016/j.cemconres.2020.106149
- **Hydrotalcite anion variants, pyroaurite, N-A-S-H gels (21 rows)**: typed from the
  published tables into `data/sources/curated_ldh_nash.csv` (verified against the PDFs
  on 2026-07-12). These were the previously unidentified legacy rows (`[2]`, `[7,8]`):
  - Bernard, Zucha, Lothenbach, Mäder, *Stability of hydrotalcite (Mg-Al layered double
    hydroxide) in presence of different anions*, CCR 152 (2022) 106674, Table 8.
    (15 CO3/OH/Cl/SO4/NO3-hydrotalcite end-members, Mg/Al = 2, 3, 4.) Note: the
    ScienceDirect rendering garbles three values (e.g. "−85,196.8"); correct readings
    confirmed by the additivity relation given in the paper: OH-Ht Mg/Al=3
    G = −8519.68 kJ/mol, NO3-Ht Mg/Al=3 G = −8497.17 kJ/mol, SO4-Ht Mg/Al=3
    H = −9948.56 kJ/mol.
  - Bernard, Lothenbach, Myers, Yio, *Pyroaurite-like phases (Mg-Fe LDH) synthesis and
    solubility*, CCR 189 (2025) 107739, Table 7. (3 CO3-pyroaurite end-members.)
  - Chen, de Lima, Li, Ma, Lothenbach, Yin, Yu, Ye, *Synthesis, solubility and
    thermodynamic properties of N-A-S-H gels with various target Si/Al ratios*,
    CCR 180 (2024) 107484, Table 10. (3 N-A-S-H gels, Si/Al = 1, 2, 3.)
- **Mg carbonates and ye'elimite (4 rows)**: stored in `data/sources/curated_additions.csv`
  (retrieved/derived 2026-07-12):
  - Hydromagnesite, nesquehonite, artinite: Thermoddem database (BRGM,
    https://thermoddem.brgm.fr) — underlying calorimetry from Königsberger et al. (1999),
    Hemingway & Robie (1972, 1973), Cp/V from Helgeson et al. (1978).
  - Ye'elimite Ca₄Al₆O₁₂(SO₄): ΔfH° = −8382.5 kJ/mol, S° = 487 J/K/mol, Cp° = 480 J/K/mol
    from Lothenbach et al. (2022), *Thermodynamic properties and hydration behavior of
    ye'elimite*, Cem. Concr. Res. (PII S0008884622002873). ΔfG° = −7928.6 kJ/mol derived
    here from ΔfH° and S° using CODATA element entropies (within the literature range
    −7803 to −8060). V° = 234.5 cm³/mol computed from the orthorhombic cell
    (a=13.036, b=13.035, c=9.168 Å, Z=4; Cuesta et al. 2013, Chem. Mater.).
  - Ternesite Ca₅(SiO₄)₂(SO₄): ΔfH° = −5993 kJ/mol measured by Skalamprinos, Galan,
    Hanein, Glasser et al., *Enthalpy of formation of ye'elimite and ternesite*,
    J. Therm. Anal. Calorim. 131 (2018) 2345–2359. S° = 352.2 J/K/mol volume-based
    estimate from Ghazizadeh, Hanein, Provis et al. (2020), *Estimation of standard molar
    entropy of cement hydrates and clinker minerals*, Cem. Concr. Res. (PII
    S0008884620304701), their Table 1. V° = 161.86 cm³/mol from the Pnma cell of Irran,
    Tillmanns, Hentschel (1997), Mineral. Petrol. 60, 121–132 (V_cell = 1075.12 Å³, Z=4).
    Cp° = 356.9 J/K/mol additivity estimate (2·β-C₂S + anhydrite, Cemdata18 values).
    ΔfG° = −5648.2 kJ/mol derived from ΔfH° and S° with CODATA element entropies;
    consistency check: ternesite is +60 kJ/mol metastable vs. 2 C₂S + CaSO₄ at 25 °C,
    in line with Hanein et al. (2017), CCR 98, 91–100 (stable only above ~1050 °C).
- **Sr phases**: `Sr` was added to the feature columns (2026-07-12); the five Cemdata18
  Sr-bearing solids (3 ECSH-SrSH end-members, strontianite, celestite) are now included
  with their Cemdata18 values.
- Only 3 legacy rows remain unverified: Zeolite-Phillipsite (K/Na/Ca forms,
  M2Al2Si6O16·6H2O, refs `[7,8]`/`[78]` of an unrecorded list). The user could not
  identify the source (2026-07-12). Keep flagged; consider excluding these three rows
  from the training set for the publication if no source is found.
- Value disagreements (>2 %) between input sources of identical composition are logged in
  `data/sources/value_conflicts.csv`. Notable resolved cases: legacy
  "Zeolite-Mordenite" G0 had a missing digit (-616,540 -> -6,151,940 J/mol, actually a
  CLI-composition phase); polymorph pairs (calcite/aragonite, quartz/SiO2(am)) produce
  harmless cross-matches. Unresolved: legacy vs Cemdata18 S0 of siliceous iron
  hydrogarnet (414 vs 820 J/K/mol) and K-S-H S0/Cp0 — the Cemdata18 value is kept,
  check against Dilnesa et al. (2014) and the original K-S-H source.

## Per-record source keys (`Ref` column, from ThermoHub `datasources`)

Confident citations:

| Key | Citation |
|---|---|
| Lothenbach_ea:2018 | Lothenbach et al., CCR 115 (2019) 472–506 (the Cemdata18 paper) |
| Matschei_ea:2007 | Matschei, Lothenbach, Glasser, *Thermodynamic properties of Portland cement hydrates in the system CaO–Al₂O₃–SiO₂–CaSO₄–CaCO₃–H₂O*, CCR 37 (2007) 1379–1410 |
| Kulik:2011 | Kulik, *Improving the structural consistency of C-S-H solid solution thermodynamic models*, CCR 41 (2011) 477–495 (CSHQ model) |
| Myers_ea:2014 | Myers, Bernal, Provis, *A thermodynamic model for C-(N-)A-S-H gel: CNASH_ss*, CCR 66 (2014) 27–47 |
| Lothenbach_ea:2006 | Lothenbach, Winnefeld, *Thermodynamic modelling of the hydration of Portland cement*, CCR 36 (2006) 209–226 |
| Lothenbach_ea:2008 | Lothenbach, Matschei, Möschner, Glasser, *Thermodynamic modelling of the effect of temperature on the hydration and porosity of Portland cement*, CCR 38 (2008) 1–18 |
| Lothenbach_ea:2012 | Lothenbach, Pelletier-Chaignat, Winnefeld, *Stability in the system CaO–Al₂O₃–H₂O*, CCR 42 (2012) 1621–1634 |
| Moeschner_ea:2008 | Möschner, Lothenbach, Rose, Ulrich, Figi, Kretzschmar, *Solubility of Fe-ettringite*, Geochim. Cosmochim. Acta 72 (2008) 1–18 |
| Dilnesa_ea:2011 | Dilnesa, Lothenbach, Le Saout, Renaudin, Mesbah, Filinchuk, Wieland, *Iron in carbonate containing AFm phases*, CCR 41 (2011) 311–323 |
| Dilnesa_ea:2014 / Dilnesa:2014 | Dilnesa, Lothenbach, Renaudin, Mesbah, Wieland, *Synthesis and characterization of hydrogarnet Ca₃(AlₓFe₁₋ₓ)₂(SiO₄)ᵧ(OH)₄₍₃₋ᵧ₎*, CCR 59 (2014) 96–111 |
| Balonis_ea:2009 | Balonis, Glasser, *The density of cement phases*, CCR 39 (2009) 733–739 |
| Balonis_ea:2010 | Balonis, Lothenbach, Le Saout, Glasser, *Impact of chloride on the mineralogy of hydrated Portland cement systems*, CCR 40 (2010) 1009–1022 |
| Balonis:2010:PhD | Balonis, PhD thesis, University of Aberdeen (2010) |
| Baquerizo_ea:2015 | Baquerizo, Matschei, Scrivener, Saeidpour, Wadsö, *Hydration states of AFm cement phases*, CCR 73 (2015) 143–157 |
| Robie_Hem:1995 | Robie, Hemingway, *Thermodynamic properties of minerals and related substances at 298.15 K and 1 bar*, U.S. Geological Survey Bulletin 2131 (1995) |
| Wagman_ea:1982 | Wagman et al., *The NBS tables of chemical thermodynamic properties*, J. Phys. Chem. Ref. Data 11, Suppl. 2 (1982) |
| Kubaschewski_ea:1993 | Kubaschewski, Alcock, Spencer, *Materials Thermochemistry*, 6th ed., Pergamon (1993) |
| Babushkin_ea:1985 | Babushkin, Matveyev, Mchedlov-Petrossyan, *Thermodynamics of Silicates*, Springer (1985) |
| Taylor:1997 | H.F.W. Taylor, *Cement Chemistry*, 2nd ed., Thomas Telford (1997) |
| Kelley:1960 | Kelley, *Contributions to the data on theoretical metallurgy*, U.S. Bureau of Mines Bulletin 584 (1960) |
| Shock_ea:1997 | Shock, Sassani, Willis, Sverjensky, GCA 61 (1997) 907–950 (HKF aqueous species) |
| Sverjensky_ea:1997 | Sverjensky, Shock, Helgeson, GCA 61 (1997) 1359–1412 |
| SUPCRT:1992 | Johnson, Oelkers, Helgeson, *SUPCRT92*, Computers & Geosciences 18 (1992) 899–947 |
| Kulik_ea:2001 | Kulik, Kersten, *Aqueous solubility diagrams for cementitious waste stabilization systems: II*, J. Am. Ceram. Soc. 84 (2001) 3017–3026 |
| Hoerckner_ea:1976 | Hörkner, Müller-Buschbaum, *Zur Kristallstruktur von CaAl₂O₄*, J. Inorg. Nucl. Chem. 38 (1976) 983–984 |

Citations to verify before publication (key known, exact bibliographic details unconfirmed):

| Key | Likely citation |
|---|---|
| Balonis_ea:2011 / Balonis_ea_b:2011 | Balonis et al., nitrate/nitrite AFm phases (Adv. Cem. Res. or CCR, 2011) [verify] |
| Baquerizo_ea:2014 | Baquerizo et al., hydration states / vapour sorption methods (2014) [verify] |
| Baquerizo_ea:2016 | Baquerizo, Matschei, Scrivener, water activity vs. ettringite stability (2016) [verify] |
| Boysen_ea:2007 | Boysen et al., mayenite C12A7 (2007) [verify] |
| Corazza:1967 | Corazza (1967), crystal structure reference [verify] |
| Dilnesa_ea:2012 | Dilnesa et al., Fe-containing AFm/monosulfate phases, CCR 42 (2012) [verify] |
| Ederova_ea:1979 | Ederová, Šatava, heat capacities of hydrated aluminate phases, Thermochim. Acta 31 (1979) [verify] |
| Felmy_ea:2003 | Felmy et al. (2003), Fe carbonate/oxide solubility [verify] |
| Garvin_ea:1987 | Garvin, Parker, White, CODATA thermodynamic tables (1987) [verify] |
| Goodwin_ea:1970 | Goodwin et al. (1970), calcium aluminate data [verify] |
| Kulik_ea:2002 | Kulik et al. (2002) [verify] |
| Lothenbach_ea:2007 | Lothenbach et al. (2007) [verify] |
| Lothenbach_ea_b:2012 / Lothenbach_ea:2016 | Lothenbach et al., hydrotalcite / M-S-H related (2012/2016) [verify] |
| Majzlan_ea:2003 | Majzlan et al., thermodynamics of Fe oxides/oxyhydroxides (2003) [verify] |
| Mascolo_ea:1980 | Mascolo, Marino, hydrotalcite-type phases (1980) [verify] |
| Matschei_ea:2015 | Matschei, Glasser, thaumasite stability (2015) [verify] |
| Rozov_ea:2011 | Rozov et al., hydrotalcite–pyroaurite LDH solid solution (2011) [verify] |
| AUG20_GEMS / JULY_GEMS / Slop98 / PRONSPREP | GEMS / SUPCRT internal dataset provenance markers (not literature) |
| Change hyd state / struct analog / structural analogy | Cemdata18 internal derivation notes (not literature) |

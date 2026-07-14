# Database sources

The master database (`selected_thermodynamic_properties.csv`, 222 phases) is rebuilt by
`scripts/build_database.py` from the source files in `data/sources/`. All values are at
25 °C and 1 bar.

## Primary sources

- **Cemdata18** (134 records): Lothenbach, Kulik, Matschei, Balonis, Baquerizo, Dilnesa,
  Miron, Myers, *Cemdata18: A chemical thermodynamic database for hydrated Portland
  cements and alkali-activated materials*, Cem. Concr. Res. 115 (2019) 472–506.
  Machine-readable ThermoFun export retrieved from the ThermoHub repository
  (https://github.com/thermohub/thermofun-jupyter): `cemdata18-thermofun.json`.
  Per-record provenance (Matschei et al. 2007; Kulik 2011; Myers et al. 2014; Dilnesa
  et al. 2011–2014; Balonis et al. 2009–2011; Baquerizo et al. 2014–2016; Möschner et al.
  2008; Robie & Hemingway 1995; and others) is preserved in the `Ref` column.

- **Zeolites** (36 records, `ma_lothenbach_zeolites.csv`; supersede earlier Cemdata18
  estimates of identical compositions):
  - Ma, Lothenbach, Cem. Concr. Res. 135 (2020) 106111 (Na-based zeolites, Table 6)
  - Ma, Lothenbach, Cem. Concr. Res. 148 (2021) 106537 (K-based zeolites, Table 5)
  - Ma, Lothenbach, Cem. Concr. Res. 135 (2020) 106149 (Ca-based zeolites, Table 4)

- **Hydrotalcites, pyroaurites, N-A-S-H gels** (21 records, `curated_ldh_nash.csv`):
  - Bernard, Zucha, Lothenbach, Mäder, Cem. Concr. Res. 152 (2022) 106674, Table 8
    (CO3/OH/Cl/SO4/NO3-hydrotalcite end-members, Mg/Al = 2–4)
  - Bernard, Lothenbach, Myers, Yio, Cem. Concr. Res. 189 (2025) 107739, Table 7
    (CO3-pyroaurites, Mg/Fe = 2–4)
  - Chen, de Lima, Li, Ma, Lothenbach, Yin, Yu, Ye, Cem. Concr. Res. 180 (2024) 107484,
    Table 10 (N-A-S-H gels, Si/Al = 1–3)

- **Crystalline calcium-silicate phases** (18 records, `hirsch2024_csh_crystalline.csv`):
  Hirsch, Lothenbach, *Phase stabilities and thermodynamic properties of crystalline
  phases in CaO–SiO2–H2O above 100 °C*, Cem. Concr. Res. 177 (2024) 107412, Table 4.

- **Mg carbonates and clinker sulfoaluminates** (5 records, `curated_additions.csv`):
  - Hydromagnesite, nesquehonite, artinite: Thermoddem database (BRGM,
    https://thermoddem.brgm.fr); underlying calorimetry from Königsberger et al. (1999),
    Hemingway & Robie (1972, 1973), Helgeson et al. (1978).
  - Ye'elimite: ΔfH°, S°, Cp° measured by Lothenbach et al., Cem. Concr. Res. (2022,
    PII S0008884622002873); ΔfG° derived via ΔG = ΔH − TΔS with CODATA element
    entropies; V° from the crystal structure of Cuesta et al., Chem. Mater. 25 (2013)
    1680–1687.
  - Ternesite: ΔfH° from Skalamprinos et al., J. Therm. Anal. Calorim. 131 (2018)
    2345–2359; S° from the volume-based estimate of Ghazizadeh et al., Cem. Concr. Res.
    (2020, PII S0008884620304701); V° from Irran et al., Mineral. Petrol. 60 (1997)
    121–132; Cp° by additivity; ΔfG° derived as above.

- **Legacy snapshot** (`legacy_master_2026-07-12.csv`): pre-rebuild master used as an
  immutable input; three phillipsite records of unidentified origin are retained in the
  database but flagged in the `Ref` column.

## Quality control applied during the build

- Element columns cross-checked against parsed chemical formulas and stored molar
  masses; entry errors corrected (documented in the build script).
- Placeholder values in reaction-derived records (V° = 0, Cp° = 0, S° ≤ 0) set to
  missing and excluded from model training.
- Conflicts > 2 % between sources of identical composition resolved in favour of the
  newer experimental determination.

If you use this database, please cite the original data sources above alongside this
repository.

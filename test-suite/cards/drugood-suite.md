# drugood-suite

| Field | Value |
|-------|-------|
| Profile (suite-level) | (P=0, T=0, H=1, C=3, Ty=2) |
| Regime | event |
| Origin | natural |
| Stability | stable |
| Member of | [drugood-cross-axis](../pairs/drugood-cross-axis.json) |
| Source | Ji et al. 2022 NeurIPS Datasets |
| License | MIT |

> **Suite** — 96 ChEMBL configurations. [drugood-lbap-ic50-assay](./drugood-lbap-ic50-assay.md) broken out as a core member within the suite.

## Facts

- 96 ChEMBL configurations from ~2.1M activity records; 5 domain split axes (assay/scaffold/size/protein/protein-family); LBAP/SBAP variants.

## Hypotheses

1. The 5 axes are not statistically independent in ChEMBL but correlate (scaffold-shift ≈ size-shift, since molecule size correlates with scaffold complexity).
2. DrugOOD performance predicts only computational held-out-ChEMBL generalization, not *wet-lab* generalization (the actual drug-discovery objective).

## Intuitions

1. DrugOOD-2 updates from 2024–25 likely exist.
2. A cross-DrugOOD-TDC scaffold-split agreement check is feasible since both use ChEMBL, and the splits probably disagree in part.
3. Critiques of the pinned ChEMBL snapshot version likely exist.

## Unknowns

1. Whether assay-as-env captures the right granularity (ID vs family vs protocol) cannot be decided — assays drift in protocol within ChEMBL itself.
2. Whether the 96 configurations are representative or curated for clear shift cannot be determined; they cover only a small region of (5-axis × ChEMBL) space.

## Boundary

ChEMBL-bounded (no patents, no other DBs); the 5 axes are pre-specified, not discovered.

## Curation

Ji et al. (Tencent AI Lab), NeurIPS 2022 — 5 domain split axes × measurement variants; motivation: realistic OOD for drug-discovery ML.

# drugood-lbap-ic50-assay (within drugood-suite)

| Field | Value |
|-------|-------|
| Profile | (P=0, T=0, H=1, C=3, Ty=2) |
| Regime | event |
| Origin | natural |
| Stability | stable |
| Member of | drugood-lbap-ic50-pair · drugood-cross-axis |
| Source | Ji et al. 2022 NeurIPS Datasets |
| License | MIT |

## Facts

- ChEMBL IC50 measurements with assay-based env split; ligand-based affinity prediction.

## Hypotheses

1. Assay identity is primarily a *target-biology-difference* signal (mechanism) rather than a pure *protocol-difference* signal (covariate), because distinct assays target distinct biology.
2. Method ordering on the assay-split differs from ordering on the scaffold-split for the same ChEMBL slice, indicating the two axes stress different model capabilities.

## Intuitions

1. State-of-the-art results on DrugOOD-LBAP-IC50 from 2024–25 likely exist.
2. Papers using the same ChEMBL slice with different env definitions likely exist and would directly expose the curation effect.

## Unknowns

1. Whether an "assay" is treated as a distinct publication-instance or merged across recurring target/protocol — same target/protocol may recur under different ChEMBL assay IDs, and whether DrugOOD merges or separates these is buried in the curation script.

## Boundary

ChEMBL-bounded; IC50 only (no Kd, Ki); ligand-only (no protein structure).

## Representational commitments

- IC50 only (not Kd, Ki, etc.)
- Ligand-based (LBAP) — no protein structure
- Assay-ID as env (not assay-family or protocol)
- Binary classification framing (active vs inactive)

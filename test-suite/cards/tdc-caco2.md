# tdc-caco2 (within tdc-suite)

| Field | Value |
|-------|-------|
| Profile | (P=0, T=0, H=1, C=3, Ty=2) |
| Regime | event |
| Origin | natural |
| Stability | stable |
| Member of | tdc-caco2-scaffold-pair · tdc-cross-task-suite |
| Source | Huang et al. 2021 Nature Chemical Biology; Wang et al. (original Caco-2) |
| License | CC-BY-4.0 |

## Facts

- Caco-2 cell permeability (ADMET oral-absorption proxy); ~900 compounds; canonical scaffold split.

## Hypotheses

1. TDC-Caco2's scaffold-split and a DrugOOD-equivalent scaffold-split produce different shifts on overlapping data rather than agreeing.
2. Caco-2 is an outlier rather than a representative ADMET task (small-N, high literature variance).

## Intuitions

1. Caco-2 cross-pipeline benchmarks likely exist for 2024–25.
2. Foundation-molecular-model results on TDC ADMET probably already exist.

## Unknowns

1. N≈900 is borderline for statistical resolution of the scaffold shift — whether the observed method ranking is signal or noise cannot be determined at this scale; this is a limit of data size, not of the shift itself.

## Boundary

Very small; useful as a canonical-but-noisy ADMET probe, not for robust shift claims.

## Representational commitments

- Bemis-Murcko scaffold definition
- Log-permeability target (not raw cm/s)
- Caco-2 cell line as proxy for oral absorption
- ~900-compound scale

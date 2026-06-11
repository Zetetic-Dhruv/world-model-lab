# galaxy-zoo-kids-desi

| Field | Value |
|-------|-------|
| Profile | (P=1, T=0, H=2, C=2, Ty=2) |
| Regime | blob |
| Origin | natural |
| Stability | stable |
| Member of | [kids-desi-cross-instrument](../pairs/kids-desi-cross-instrument.json) |
| Source | Walmsley et al. 2024 (Zoobot); Galaxy Zoo DESI |
| License | CC-BY-4.0 |

## Facts

- Same galaxies imaged by KiDS (deeper, higher-res) and DESI (shallower, lower-res) on GAMA fields; Zoobot CNN baseline.

## Hypotheses

1. Morphology classification trained on KiDS does not fully transfer to DESI of the *same* galaxies, because the instrument gap produces a *physical* effect (different features visible at different depths) rather than a pure covariate shift.
2. The cross-instrument shift on the same galaxies acts as a "label-stability test", so method behavior here bounds generalization to *new* galaxies only weakly.

## Intuitions

1. Zoobot extensions from 2024–25, Galaxy Zoo DESI follow-ups, and foundation-astronomy-model results on cross-instrument galaxy data likely exist.
2. LSST early-release galaxy morphology classifications that relate to KiDS/DESI trained models probably exist.

## Unknowns

1. Cross-instrument shift on the *same* galaxies decouples population shift from image-formation shift more cleanly than any other dataset in the corpus, but there is no established label for this "label-stable observation shift".
2. Citizen-science labels (Galaxy Zoo) carry annotator-disagreement structure that is not separately exposed; how this interacts with cross-instrument shift cannot be determined from the data.

## Boundary

Two-instrument only; cannot generalize to three-way (with LSST or Euclid) without additional data.

## Representational commitments

- Two-survey selection (KiDS, DESI)
- GAMA field restriction
- Citizen-science labels aggregated as vote distribution
- Image-as-unit (catalog cross-match resolves to galaxy)

# wilds-camelyon17 (within wilds-suite)

| Field | Value |
|-------|-------|
| Profile | (P=0, T=0, H=2, C=2, Ty=2) |
| Regime | blob |
| Origin | natural |
| Stability | stable |
| Member of | [camelyon17-hospital-shift](../pairs/camelyon17-hospital-shift.json) · wilds-shift-suite |
| Source | Bandi et al. 2019 TMI; Koh et al. 2021 ICML (WILDS framing) |
| License | CC0-1.0 |

## Facts

- ~450k histopathology patches from 5 hospitals (tumor vs normal); hospital-shift canonical.

## Hypotheses

1. Camelyon17's hospital-shift is a *different kind* of shift from NIH↔CheXpert↔PadChest hospital-shift because histopathology vs radiology modality matters.
2. Staining variability accounts for most of the hospital-shift, while tissue-population factors contribute less.

## Intuitions

1. Histopathology foundation-model results from 2024–25 likely exist.
2. Staining-normalization ablations quantifying staining vs population contributions have probably been published.

## Unknowns

1. Whether the environment identity for "hospital" is the *institution*, the *scanner*, or the *technician* cannot be determined; Camelyon17 fixes one labelling and sub-hospital factors are not recoverable from the data.
2. Patch-as-unit (vs slide-as-unit) is a hidden preparation choice.

## Boundary

A static dataset (T=0, P=0); the natural axis is the cross-modality pair between a radiology entry and this pathology entry.

## Representational commitments

- Patch-as-unit (slide-context lost)
- Binary tumor label (multi-class pathology collapsed)
- Hospital-as-env (sub-hospital factors not exposed)
- WILDS preprocessing — raw WSIs (700 GB) replaced with patches

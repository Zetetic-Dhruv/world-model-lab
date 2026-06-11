# wilds-suite

| Field | Value |
|-------|-------|
| Profile (suite-level) | (P=0, T=1, H=2, C=1, Ty=2) |
| Regime | blob (mostly) |
| Origin | natural |
| Stability | churning (v2.0 in 2022) |
| Member of | [wilds-shift-suite](../pairs/wilds-shift-suite.json) |
| Source | Koh et al. 2021 ICML; Sagawa et al. 2022 (v2) |
| License | MIT |

> **Suite** — 10 sub-datasets (Camelyon17, iWildCam, FMoW-WILDS, PovertyMap, RxRx1, GlobalWheat, OGB-MolPCBA, CivilComments, Amazon-WILDS, Py150). [wilds-camelyon17](./wilds-camelyon17.md) broken out as a core entry within the suite.

## Facts

- 10 sub-datasets curated for natural shift; v2.0 unlabeled-data extension; Koh et al. ICML 2021.

## Hypotheses

1. No single method wins across all 10 sub-datasets; method ranking is shift-type-dependent (Koh's empirical answer was no universal winner; needs re-verification on 2024–25 state-of-the-art models).
2. WILDS's curation introduces a selection-for-clear-shift bias.

## Intuitions

1. A WILDS v3 may have been announced; the WILDS-vs-Wild-Time naming confusion likely appears in the 2023–25 literature (distinct projects).
2. Foundation-model cross-WILDS results have probably been published.

## Unknowns

1. WILDS deliberately *picked* clearly-shifted environments; there is no established framing for how curator choice propagates to out-of-benchmark generalizability.
2. Profile coverage is sparse — no profile region has ≥3 WILDS sub-datasets, so within-region statistics are weak.

## Boundary

A suite requiring descent to sub-datasets for routing; suite-level claims need cross-sub-dataset agreement, which is empirically rare.

## Curation

Koh, Sagawa et al. (Stanford), ICML 2021; motivation: OOD generalization in real-world deployment.

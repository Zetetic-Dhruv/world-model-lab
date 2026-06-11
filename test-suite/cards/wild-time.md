# wild-time

| Field | Value |
|-------|-------|
| Profile (suite-level) | (P=0, T=3, H=1, C=1, Ty=2) |
| Regime | blob (mixed image + text) |
| Origin | natural |
| Stability | stable |
| Member of | [wild-time-temporal-suite](../pairs/wild-time-temporal-suite.json) |
| Source | Yao et al. 2022 NeurIPS |
| License | MIT |

> **Suite** — 5 sub-datasets (Yearbook, FMoW, MIMIC-prognosis, HuffPost, arXiv), each with its own profile and pair. Under the regime scheme, "blob" covers both image (Yearbook, FMoW) and text (HuffPost, arXiv); MIMIC-prognosis is event-like clinical features.

## Facts

- 5 sub-datasets sharing a chronological eval protocol; NeurIPS 2022; multi-task per sub-dataset (MIMIC = readmission + mortality).

## Hypotheses

1. Time-as-environment shifts are not mechanistically *consistent* across the 5 sub-datasets; each has its own incompatible drift mechanism rather than a shared method ordering at any given epoch.
2. The suite's published "time generalization" claim holds only on the simpler sub-datasets (Yearbook) rather than uniformly across all 5.

## Intuitions

1. A Wild-Time leaderboard with 2024–25 updates and foundation-model results on Yearbook likely exists.
2. Cross-comparisons with WILDS-Time and the TGB temporal axis probably exist (distinct projects despite naming overlap).

## Unknowns

1. "Temporal shift" is treated as a single primitive, but the underlying mechanism differs categorically across the 5 sub-datasets; there is no established label for shift-mechanism heterogeneity *within* a single environment axis. The Wild-Time framing presupposes their unification.

## Boundary

A suite requiring descent to sub-dataset manifests before routing methods.

## Representational commitments

- Time-as-env (suite-level)
- 5 specific sub-datasets chosen as representative
- MIMIC sub-dataset uses Wild-Time's derived clinical-features version, not raw MIMIC (user-excluded)

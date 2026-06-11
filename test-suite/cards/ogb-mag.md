# ogb-mag

| Field | Value |
|-------|-------|
| Profile | (P=0, T=2, H=2, C=0, Ty=3) |
| Regime | KG |
| Origin | natural (MAG snapshot; MAG deprecated 2022) |
| Stability | churning |
| Pairs with | — |
| Member of | [ogb-mag-year-split](../pairs/ogb-mag-year-split.json) |
| Source | Hu et al. 2020 NeurIPS |
| License | ODC-BY |

## Facts

- 1.94M nodes (paper/author/institution/field) × 21M edges × 4 relation types; year-based OGB standard split (≤2017 / 2018 / ≥2019).

## Hypotheses

1. Year-shift on MAG is a concept shift rather than a covariate shift: citation patterns themselves evolve (e.g., the rise of ML papers re-shapes the field-of-study graph), not merely the volume of papers.
2. The field-of-study taxonomy itself drifts across years, confounding the env with the concept-of-env.

## Intuitions

1. The Microsoft Academic Graph deprecation (2022) likely has a downstream effect on OGB-MAG as a historical snapshot.
2. A heterogeneous-graph foundation-model literature probably emerged in 2023–25.

## Unknowns

1. "Paper" as atomic unit is a representational commitment OGB-MAG forces — sub-paper structure (claims, sections, figures) is invisible, and whether finer atomism would re-shape the cross-year shift cannot be determined from this data.

## Boundary

C=0; no within-paper compositional handle.

## Representational commitments

- Paper-as-atom — sections/claims/figures invisible
- Field-of-study taxonomy is the MAG curatorial snapshot
- 4 node types is OGB's selection from MAG's richer schema
- 2019 cutoff fixes the historical horizon

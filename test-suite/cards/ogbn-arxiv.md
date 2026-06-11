# ogbn-arxiv (within ogb-suite)

| Field | Value |
|-------|-------|
| Profile | (P=0, T=2, H=2, C=0, Ty=3) |
| Regime | KG |
| Origin | natural (MAG-derived; pre-2020 snapshot) |
| Stability | stable |
| Member of | [ogbn-arxiv-year-split](../pairs/ogbn-arxiv-year-split.json) · ogb-application-suite |
| Source | Hu et al. 2020 NeurIPS |
| License | ODC-BY |

## Facts

- 170k CS arXiv papers as citation network; year split (≤2017 / 2018 / ≥2019); MAG-derived.

## Hypotheses

1. The 2017–2018 boundary is a natural break (arXiv categorization shift, volume jump), not merely a convenient cut point.
2. Method degradation does not track the time gap monotonically (2019 vs 2020 vs 2021) but is threshold-like, jumping at the categorization break.

## Intuitions

1. Extensions of ogbn-arxiv with later test years likely already exist in the literature.
2. ogbl-citation2 (similar temporal structure on full arXiv) probably provides a useful cross-comparison.

## Unknowns

1. The field-of-study taxonomy itself evolved post-2017 (ML categories proliferated); whether the resulting shift is *concept* shift or *covariate* shift cannot be determined from the data.
2. The underlying MAG snapshot predates MAG's 2022 deprecation — the source data is partially dated.

## Boundary

C=0; the 170k-paper scale limits fine-grained per-environment analysis.

## Representational commitments

- Paper-as-atom (no sub-paper structure)
- 40-class CS subject taxonomy (subset of MAG)
- 2019 cutoff fixes the historical horizon

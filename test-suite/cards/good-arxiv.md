# good-arxiv (within good-suite)

| Field | Value |
|-------|-------|
| Profile | (P=0, T=2, H=2, C=0, Ty=3) |
| Regime | KG (text + graph) |
| Origin | natural |
| Stability | stable |
| Pairs with | [ogbn-arxiv](../manifests/ogbn-arxiv/) |
| Member of | good-arxiv-cross-shift · good-cross-shift-suite |
| Source | Gui et al. 2022 NeurIPS |
| License | MIT |

## Facts

- arXiv papers + citation graph; covariate splits (year/domain/word) and concept splits (year/domain/word) on the same underlying data.

## Hypotheses

1. Within GOOD-Arxiv, covariate-year and concept-year splits at the *same year boundary* expose different methods, providing a rare clean test of GOOD's theoretical dichotomy.
2. GOOD-Arxiv's text+graph hybrid behaves intermediate between pure-text (GOODSST2) and pure-graph (ogbn-arxiv) rather than categorically distinct from both.

## Intuitions

1. A three-way cross-comparison of GOOD-Arxiv vs ogbn-arxiv vs Wild-Time-arXiv is feasible — three different splits on similar underlying data that would directly expose split-as-confounder.
2. Foundation-model results from 2024–25 on GOOD-Arxiv specifically likely exist.

## Unknowns

1. Whether the empirical separation of covariate-year vs concept-year tracks the theoretical intent cannot be determined — GOOD operationalizes the distinction via specific split rules, so the paper's distinction is operational, not theoretical.

## Boundary

Inherits the C=0 of the arXiv data; small-scale per split.

## Representational commitments

- Covariate vs concept operationalized via specific split rules
- 3 domain choices × 2 shift types = 6 splits curated
- Text and graph treated as one domain

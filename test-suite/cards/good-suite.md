# good-suite

| Field | Value |
|-------|-------|
| Profile (suite-level) | (P=0, T=2, H=2, C=1, Ty=2) |
| Regime | KG |
| Origin | mixed (real + synthetic) |
| Stability | stable |
| Member of | [good-cross-shift-suite](../pairs/good-cross-shift-suite.json) |
| Source | Gui et al. 2022 NeurIPS |
| License | MIT |

> **Suite** — 51 splits across 11 datasets × 3 shift types. Real sub-datasets (GOODHIV, GOODArxiv, GOODSST2, …) are naturally occurring; **synthetic sub-datasets (GOODMotif, GOODCMNIST) are not**. [good-arxiv](./good-arxiv.md) broken out within the suite.

## Facts

- 51 splits across 11 datasets × 3 shift types; *explicit covariate-vs-concept-shift labels*; 12 baselines × 10 runs in NeurIPS 2022.

## Hypotheses

1. Methods that win on covariate shift are categorically separate from those that win on concept shift within the *same* dataset, and GOOD is the cleanest data for distinguishing the two.
2. Synthetic-GOOD conclusions (Motif, CMNIST) do not transfer to real-GOOD (HIV, Arxiv).

## Intuitions

1. A GOOD-2 or successor benchmark likely exists or is emerging in 2024–25.
2. Follow-up work expanding the shift-type taxonomy beyond the binary covariate/concept split probably exists.

## Unknowns

1. Whether covariate-vs-concept is the *right* partition or conflates finer distinctions (covariate-with-fixed-conditional vs covariate-with-drifting-margins) — GOOD presupposes the binary, and there is no established finer decomposition.
2. The synthetic sub-datasets are not naturally occurring; suite-level treatment glosses this over.

## Boundary

The synthetic sub-datasets are not naturally occurring and the real sub-datasets carry the suite's contribution: an explicit covariate-vs-concept labelling across many graph datasets.

## Curation

Gui et al. (DIVE Lab), NeurIPS 2022 — splits per (dataset, domain, shift-type) triple; motivation: explicit covariate-vs-concept distinction for graph OOD.

# tgb (Temporal Graph Benchmark)

| Field | Value |
|-------|-------|
| Profile (suite-level) | (P=0, T=3, H=1, C=0, Ty=2) |
| Regime | KG (temporal) |
| Origin | natural |
| Stability | stable |
| Member of | [tgb-temporal-suite](../pairs/tgb-temporal-suite.json) |
| Source | Huang et al. 2023 NeurIPS |
| License | MIT |

> **Suite** — sub-datasets (tgbl-wiki, tgbl-review, tgbl-flight, tgbl-comment, tgbn-trade, tgbn-genre, tgbn-reddit) carry their own profiles and pairs. Suite-level claims require cross-sub-dataset agreement.

## Facts

- 7+ sub-datasets sharing chronological 70/15/15 split; NeurIPS 2023 baselines.

## Hypotheses

1. Per sub-dataset, "future-as-test" is genuinely OOD rather than a longer tail of a stationary process, so a *non*-OOD baseline is the right comparison.
2. Method ranking does not transfer *across* TGB sub-datasets; each is its own incompatible regime.

## Intuitions

1. A TGB 2.0 release has likely shipped.
2. Concept-drift detection literature has probably been applied explicitly to dynamic-link prediction since 2023.

## Unknowns

1. Whether the temporal environment axis is single (time) or compound (time × sub-dataset × edge-type) cannot be determined; the suite's structure presupposes time-as-primary.
2. Each sub-dataset's shift *mechanism* differs (Wikipedia growth is not trade-policy regime change), so treating them under one banner may be a category error.

## Boundary

A suite requiring descent to individual sub-datasets for routing. H=1 (flat within each sub-dataset); C=0. The assumption that the sub-datasets are homogeneous is itself a limitation, not just a structural score.

## Representational commitments

- Time-as-primary-env (suite-level commitment)
- Each sub-dataset selected for a distinct temporal shift mechanism
- 70/15/15 split fraction fixed across sub-datasets
- Edge-level temporal stamps; node-stability across time assumed

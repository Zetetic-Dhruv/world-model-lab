# ogb-suite (non-MAG)

| Field | Value |
|-------|-------|
| Profile (suite-level) | (P=0, T=1, H=2, C=1, Ty=2) |
| Regime | KG (some blob for ogbg-mol*) |
| Origin | natural |
| Stability | churning |
| Member of | [ogb-application-suite](../pairs/ogb-application-suite.json) |
| Source | Hu et al. 2020 NeurIPS |
| License | MIT |

> **Suite** — 13+ sub-datasets across node/link/graph property prediction. [ogbn-arxiv](./ogbn-arxiv.md) and [ogb-mag](./ogb-mag.md) are broken out separately.

## Facts

- 13+ sub-datasets with application-specific splits (year, scaffold, species, throughput, snapshot timestamp); Hu et al. NeurIPS 2020.

## Hypotheses

1. The "application-specific" splits produce consistent shift mechanisms *within* each shift-type: all year-splits behave similarly across sub-datasets.
2. A single cross-OGB method wins on temporal, scaffold, and species splits simultaneously.

## Intuitions

1. An OGB leaderboard SOTA for 2024–25 and graph-foundation-model attempts spanning OGB likely exist.
2. A year-split critique literature probably exists, since ogbn-arxiv's boundary is debated.

## Unknowns

1. Whether OGB's splits are *natural* or *curator-chosen* cannot be determined — year-2017, a specific scaffold-bin, and a specific Wikidata-snapshot were all designer decisions, and there is no established label for "split-designer as confounder".

## Boundary

Profile coverage is narrow — mostly (P=0, T=0–2, H=1–2, C=0–3, Ty=2–3); silent on process and most temporal dynamics.

## Curation

Hu, Fey, Zitnik et al. (Stanford), NeurIPS 2020 — application-specific splits per sub-dataset; motivation: realistic ML on graph data.

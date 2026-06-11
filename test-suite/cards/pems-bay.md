# pems-bay

| Field | Value |
|-------|-------|
| Profile | (P=2, T=3, H=2, C=0, Ty=1) |
| Regime | event |
| Origin | natural |
| Stability | stable |
| Pairs with | [metr-la](../manifests/metr-la/) |
| Member of | [metr-pems-bay](../pairs/metr-pems-bay.json) |
| Source | Li et al. 2018 ICLR (DCRNN) |
| License | MIT |

## Facts

- 325 SF Bay Area freeway sensors × 5-min resolution × 6 months (Jan–May 2017).

## Hypotheses

1. Transfer is directionally asymmetric — Bay→LA is harder than LA→Bay because LA has a more complex topology.
2. The 5-year offset (LA 2012 / Bay 2017) injects a *temporal* shift orthogonal to the spatial axis.

## Intuitions

1. A multi-PEMS literature (PEMS03/04/07/08) likely exists that positions Bay relative to the other sites.
2. Cross-PEMS foundation-model results probably already exist.

## Unknowns

1. There is no established label for the directional asymmetry of transfer (A→B ≠ B→A); the paired-dataset schema is undirected by default and does not yet represent this distinction.

## Boundary

Sibling of METR-LA; shares the same modality and compositional ceiling.

## Representational commitments

- Sensor-location-as-hierarchy
- 5-minute temporal grain
- Speed-only modality
- 5-year offset from METR-LA temporal range (introduces incidental shift)

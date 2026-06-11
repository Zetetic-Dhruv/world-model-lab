# metr-la

| Field | Value |
|-------|-------|
| Profile | (P=2, T=3, H=2, C=0, Ty=1) |
| Regime | event |
| Origin | natural |
| Stability | stable |
| Pairs with | [pems-bay](../manifests/pems-bay/) · chengdu-traffic · shenzhen-traffic |
| Member of | [metr-pems-bay](../pairs/metr-pems-bay.json) · traffic-cross-city-4way |
| Source | Li et al. 2018 ICLR (DCRNN) |
| License | MIT |

## Facts

- 207 LA freeway sensors × 5-min resolution × 4 months (Mar–Jun 2012).

## Hypotheses

1. METR-LA → PEMS-BAY is a *weak* shift (both California freeway, similar urbanization), exposing the threshold below which "cross-city" is indistinguishable from "cross-fold".
2. Adding Chengdu/Shenzhen as test envs separates "cross-administrative-unit" from "cross-traffic-culture" shift.

## Intuitions

1. Cross-city baselines from Liu 2023 CIKM and ECML/PKDD 2024 FEPCross likely exist.
2. 2024–25 traffic-foundation-model work aggregating all four cities probably exists.

## Unknowns

1. Network topology (which sensor connects to which) is supplied externally as adjacency, not discovered from the data — there is no established label for "imposed-vs-discovered topology" as a shift-relevant axis.
2. Within-day periodicity dominates variance; distinguishing temporal-shift signal from periodic baseline is itself unsolved here.

## Boundary

Ty=1, C=0; restricted to spatial-temporal OOD. Single measurement modality.

## Representational commitments

- Sensor-location-as-hierarchy (network topology imposed)
- 5-minute temporal grain fixed
- Speed-only modality (no flow, no occupancy)
- Adjacency matrix supplied externally

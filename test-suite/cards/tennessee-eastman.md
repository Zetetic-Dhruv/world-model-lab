# tennessee-eastman

| Field | Value |
|-------|-------|
| Profile | (P=3, T=2, H=2, C=1, Ty=3) |
| Regime | event |
| Origin | **simulator** |
| Stability | stable |
| Member of | [te-cross-mode](../pairs/te-cross-mode.json) |
| Source | Reinartz et al. 2021 (extended); Downs & Vogel 1993 (original) |
| License | CC0-1.0 |

## Facts

- Chemical-process simulator with 28 fault types × 6 operating modes (Reinartz 2021 ext.); 11 manipulated + 41 measured variables.

## Hypotheses

1. The (mode × fault) combinations are not all behaviorally distinct; degeneracies collapse the effective cardinality below 6×28.
2. The simulator's PI-controller architecture leaks into the OOD signal as a hidden confounder common to all environments.

## Intuitions

1. Tennessee-Eastman likely appears in the 2023–25 causal-discovery literature.
2. Ablations replacing specific simulator components (kinetics, controller) to expose simulator-dependent results probably exist.

## Unknowns

1. "Mode" vs "fault" are both deviations from baseline; the dataset's hierarchy (mode = expected operating regime; fault = unexpected perturbation) is a curated taxonomy, and there is no established label distinguishing "expected" from "unexpected" perturbation as separate shift types.

## Boundary

A simulator dataset with Ty=3 whose heterogeneity comes entirely from the simulator's own variable naming — typed structure is *generated*, not discovered.

## Representational commitments

- PI-controller architecture fixed
- Specific kinetic model (Downs-Vogel 1993)
- 11 manipulated + 41 measured variable selection
- Mode-vs-fault taxonomy is the curator's distinction

# nasa-cmapss

| Field | Value |
|-------|-------|
| Profile | (P=3, T=2, H=2, C=1, Ty=2) |
| Regime | event |
| Origin | **simulator** |
| Stability | stable |
| Member of | [cmapss-cross-subset](../pairs/cmapss-cross-subset.json) |
| Source | Saxena et al. 2008 PHM; NASA PCoE |
| License | Public domain (US govt work) |

## Facts

- FD001–FD004 simulating turbofan run-to-failure with operating-conditions × fault-modes factorial (1×1, 6×1, 1×2, 6×2).

## Hypotheses

1. The (op-condition × fault-mode) factorization is not orthogonal in the simulator's output: interactions dominate one axis.
2. Method ordering on CMAPSS does not predict ordering on N-CMAPSS (its more realistic NASA successor), only ordering on yet-more-CMAPSS.

## Intuitions

1. Cross-subset transfer surveys from 2023–25 and CMAPSS-vs-N-CMAPSS empirical comparisons likely exist.
2. Degradation-model-ablation work varying the simulator's Wiener-process assumption probably exists.

## Unknowns

1. The simulator's degradation model is a representational commitment baked into every datum; whether real engine degradation actually follows this form cannot be determined from CMAPSS.
2. The "fault mode" labels presuppose the same underlying physics across instances — also untestable from inside.

## Boundary

Simulated data — every claim using CMAPSS is conditional on the simulator's structural choices; cannot stand as evidence for generalization to real data.

## Representational commitments

- Wiener-process-like degradation model fixed
- 21 sensors selected
- Cycle-as-time-unit (no absolute timestamps)
- 4 subset definitions are the curator's factorial design

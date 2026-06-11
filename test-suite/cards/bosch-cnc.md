# bosch-cnc

| Field | Value |
|-------|-------|
| Profile | (P=3, T=3, H=2, C=0, Ty=1) |
| Regime | event |
| Origin | natural |
| Stability | stable |
| Member of | [bosch-cross-machine](../pairs/bosch-cross-machine.json) · bosch-temporal-drift |
| Source | Tnani et al. 2022 CIRP; boschresearch/CNC_Machining |
| License | CC-BY-NC-SA-4.0 |

## Facts

- 3 CNC mills × 15 machining programs × 6×6-month windows (Oct 2018–Aug 2021) of triaxial accelerometer + binary anomaly label.

## Hypotheses

1. Cross-machine shift (M1+M2 → M3) is a categorically different shift type from within-machine 2-year temporal drift, because the two are not factorable through a single hidden "machine state" environment.
2. The 6-month window carries mechanism (tool-wear cycles) rather than merely data-collection cadence, because wear accumulates on a comparable timescale.

## Intuitions

1. Follow-ups to Tnani 2022 CIRP for 2023–25 and SSMSPC extensions likely exist.
2. Industrial-AI work distinguishing brownfield drift from greenfield deployment probably uses this dataset.

## Unknowns

1. The "machining process" boundary is fixed by program-identity in the data, so whether a process is a *labelled* unit or a *discovered* segmentation cannot be settled here.
2. Tool-wear is plausibly *both* a process and a temporal phenomenon; the accelerometer alone cannot decide between them.

## Boundary

With Ty=1 (accelerometer only) and no temperature/current/acoustic complement, the dataset caps what typed shift it can show.

## Representational commitments

- Machining-program-identity defines process boundary
- Accelerometer-only modality
- 6-month windowing reflects data-collection cadence
- Binary anomaly label collapses fault types

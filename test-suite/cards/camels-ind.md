# camels-ind

| Field | Value |
|-------|-------|
| Profile | (P=3, T=3, H=2, C=0, Ty=2) |
| Regime | event |
| Origin | natural |
| Stability | stable |
| Pairs with | [camels-us](../manifests/camels-us/) · [camels-dk](../manifests/camels-dk/) |
| Member of | [camels-three-way](../pairs/camels-three-way.json) |
| Source | CAMELS-IND release (Zenodo) |
| License | CC-BY-4.0 |

## Facts

- ~470 Indian catchments × multi-decade daily; schema-aligned with CAMELS-US per CAMELS family standard.

## Hypotheses

1. India introduces categorically new mechanisms (monsoon discharge regime) rather than a transitive 3-way structure (US→DK ≈ US→IND ∘ IND→DK), because tropical monsoon hydrology is not interpolable from temperate-zone pairs.
2. Monsoon-dominated hydrology is a *mechanism* shift relative to the temperate zone, requiring structural-profile extension beyond P=3.

## Intuitions

1. Cross-CAMELS analyses spanning US/DK/IND/CL/AUS/BR/CH/GB likely exist or are being assembled.
2. Transfer-learning work specific to Indian hydrology probably exists.

## Unknowns

1. Whether the CAMELS schema *can* adequately represent monsoon dynamics at daily resolution with standard forcings cannot be determined, since the schema-alignment that makes the pair tractable may lossily project monsoon signal.
2. Catchment partition is plausibly more problematic in non-temperate river networks, but this cannot be assessed from the data.

## Boundary

Inherits the CAMELS family's C=0; adds tropical/monsoon hydrology as the third leg of the cross-country triple.

## Representational commitments

- Catchment partition imposed
- Daily resolution may coarsen monsoon storm dynamics
- Schema-aligned with US (potentially lossy)
- Indian sub-continent geographic restriction

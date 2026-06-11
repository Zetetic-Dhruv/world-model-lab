# climatebench

| Field | Value |
|-------|-------|
| Profile | (P=3, T=3, H=2, C=0, Ty=2) |
| Regime | array |
| Origin | **simulator** (NorESM2 CMIP6) |
| Stability | stable |
| Member of | [climatebench-ssp245-holdout](../pairs/climatebench-ssp245-holdout.json) |
| Source | Watson-Parris et al. 2022 JAMES |
| License | CC-BY-4.0 |

## Facts

- NorESM2 simulations under multiple SSP + idealized forcing scenarios; held-out SSP245 standard OOD test.

## Hypotheses

1. Held-out SSP245 reveals *forced-response* generalization (the policy-relevant target), rather than letting methods exploit historical-baseline information shared across all scenarios.
2. The SSP245 hold-out is a *curator-chosen* representative of "in-between" scenarios rather than a naturally occurring mid-mitigation regime.

## Intuitions

1. ClimateBench v2 or extensions likely exist in the 2024–25 literature.
2. Cross-CMIP6-model emulation benchmarks probably exist, and method ordering likely transfers NorESM2↔UKESM1↔other only partially.
3. Climate foundation models (Aurora, ClimaX) likely have documented OOD behavior on SSP scenarios.

## Unknowns

1. Whether climate-emulator generalization (NorESM2-to-NorESM2 cross-scenario) maps to climate-*prediction* generalization (any-model-to-real-Earth) cannot be determined here — these are categorically different shifts that the literature often conflates.
2. The SSP scenario tree is itself a curatorial choice in CMIP6; held-out SSP245 represents an "unseen specific scenario", not an "unseen real future".

## Boundary

Simulator-derived data — climate-model emulation, not Earth-prediction. Useful for testing method behavior on climate-like data; cannot directly evidence real-world climate predictions.

## Representational commitments

- NorESM2 climate model
- SSP scenario tree (CMIP6 curatorial)
- Annual temporal grain
- Forcing-response framing (4 forcings → 2 responses)

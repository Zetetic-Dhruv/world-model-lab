# camels-us

| Field | Value |
|-------|-------|
| Profile | (P=3, T=3, H=2, C=0, Ty=2) |
| Regime | event |
| Origin | natural |
| Stability | stable |
| Pairs with | [camels-dk](../manifests/camels-dk/) · [camels-ind](../manifests/camels-ind/) |
| Member of | [camels-cross-country](../pairs/camels-cross-country.json) |
| Source | Newman et al. 2015 (dataset), [DOI 10.5065/D6MW2F4D](https://doi.org/10.5065/D6MW2F4D), NCAR/RAL |
| License | Public domain (US govt work) |

## Facts

- 671 watersheds × daily 1980–2014; EA-LSTM in-country median NSE ≈ 0.74 (Kratzert 2019 *HESS*).

## Hypotheses

1. Country is a categorically different environment unit than held-out catchment, because held-out-catchment PUB shift does not share the *same structure* as US→DK cross-country shift.
2. The 60 static attributes are insufficient to characterize the environment, because a hidden axis (e.g., climate zone) is confounded with country.

## Intuitions

1. US-internal PUB extensions of Kratzert 2019 for 2024–25 likely exist.
2. CAMELS-US probably appears in causal-hydrology / Beven-school work distinguishing process knowledge from data fit.

## Unknowns

1. "Catchment" is an imposed cartographic partition, not a discovered one; there is no established framing for imposed-versus-discovered structural primitives, and this dataset surfaces that gap.
2. Whether daily resolution is the right time grain or coarsens away mechanism cannot be determined from the data.

## Boundary

C=0 (no compositional handle); process is causal-chain, not branching-mechanism — silent on multi-pathway shift.

## Representational commitments

- Catchment partition imposed (cartographic decision)
- Daily resolution coarsens sub-daily storm-hydrograph dynamics
- 60 static attributes are a curatorial selection
- Three forcing datasets (Daymet, NLDAS, Maurer) — choice itself biases analysis

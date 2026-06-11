# camels-dk

| Field | Value |
|-------|-------|
| Profile | (P=3, T=3, H=2, C=0, Ty=2) |
| Regime | event |
| Origin | natural |
| Stability | stable |
| Pairs with | [camels-us](../manifests/camels-us/) · [camels-ind](../manifests/camels-ind/) |
| Member of | [camels-cross-country](../pairs/camels-cross-country.json) |
| Source | Liu et al. 2025, *Earth Syst. Sci. Data* 17:1551–1572, [doi:10.5194/essd-17-1551-2025](https://doi.org/10.5194/essd-17-1551-2025) |
| License | CC-BY-4.0 |
| Snapshot date | 2026-05-26 |

## Facts

- 3,330 Danish catchments at daily resolution, 1989–2019; schema-aligned with CAMELS-US per Liu 2025 ESSD.

## Hypotheses

1. The matched (P=3, T=3) structural profile of CAMELS-US and CAMELS-DK predicts transfer-method ordering, consistent with the published PUB ranking.
2. The country axis is a mechanism shift rather than a pure covariate shift: snow-melt and groundwater regimes in Denmark differ systematically from the US median.
3. The cross-country generalization gap is predictable from the datasets' typed-process structure alone, before any model training.

## Intuitions

1. Nearing et al.'s 2024 *Nature* global-flood model likely underperforms on CAMELS-DK specifically — the paper is US-heavy and DK performance is buried or absent.
2. A combined US+DK+IND meta-analysis ("META-CAMELS") likely already exists or is emerging in the 2024–25 literature.

## Unknowns

1. Measurement-network change and climate change cannot be disentangled as joint sources of within-DK temporal drift — both are mechanism shifts with the same data signature, and there is no established label for "measurement-process drift" as a distinct shift type.
2. Whether schema-alignment with CAMELS-US lossily projects DK-specific signal (e.g. groundwater-mediated baseflow dynamics present in Denmark but poorly represented by US-derived static attributes).

## Boundary

C=0 — the dataset is silent on compositional shift. Its contribution is the country (cross-population) axis, not expanded structural coverage.

## Representational commitments

- Catchment partition is an imposed cartographic decision, not discovered from the data.
- Daily temporal resolution coarsens away sub-daily mechanism (storm hydrograph shape).
- Forcing variables chosen to be Daymet-equivalent — biases the schema toward measurements the US release also has.
- Static attribute set is schema-aligned with US, potentially lossy for groundwater-driven DK hydrology.

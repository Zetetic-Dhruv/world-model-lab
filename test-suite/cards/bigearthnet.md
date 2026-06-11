# bigearthnet

| Field | Value |
|-------|-------|
| Profile | (P=1, T=0, H=2, C=0, Ty=3) |
| Regime | blob |
| Origin | natural |
| Stability | stable |
| Pairs with | [so2sat-lcz42](../manifests/so2sat-lcz42/) |
| Member of | [bigearth-cross-country](../pairs/bigearth-cross-country.json) |
| Source | Sumbul et al. 2019 IGARSS |
| License | CC-BY-4.0 |

## Facts

- 590,326 Sentinel-2 patches over 10 European countries (Jun 2017 – May 2018); 43 CORINE Land Cover classes, multi-label.

## Hypotheses

1. Methods that succeed on BigEarthNet cross-country shift also transfer to So2Sat on overlapping geography, because both share the same European-geography shift structure.
2. CORINE multi-label classification is not a clean compositional handle (each patch carries multiple LC labels), because it conflates spatial composition with semantic composition.

## Intuitions

1. BigEarthNet v2 (BigEarthNet-MM with Sentinel-1 SAR added) likely exists with its own extensions and benchmarks, alongside foundation earth-observation results.
2. Cross-EO-benchmark transfer studies (BigEarthNet vs So2Sat vs EuroSAT vs SEN12MS) likely exist.

## Unknowns

1. CORINE's 43-class taxonomy is European-specific; whether it generalizes to non-EU geographies cannot be determined from BigEarthNet alone, and the 10-country selection is a curatorial choice.
2. Whether the dataset's Sentinel-2 single-revisit (10-day) versus averaged-over-period preparation affects implicit temporal information cannot be assessed from the prepared data.

## Boundary

Europe-only; T=0; C=0 even though multi-label exists (composition is label-level, not entity-level). Stability is stable but the underlying Sentinel-2 mission continues — a version axis exists but is unmaterialized.

## Representational commitments

- 10-country EU selection
- 43-class CORINE taxonomy (Europe-specific)
- Jun 2017 - May 2018 period
- 120x120 patch size

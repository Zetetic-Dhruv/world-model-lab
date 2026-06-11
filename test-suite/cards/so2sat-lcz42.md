# so2sat-lcz42

| Field | Value |
|-------|-------|
| Profile | (P=1, T=0, H=2, C=1, Ty=3) |
| Regime | blob |
| Origin | natural |
| Stability | stable |
| Pairs with | [bigearthnet](../manifests/bigearthnet/) |
| Member of | [so2sat-cultural-10](../pairs/so2sat-cultural-10.json) |
| Source | Zhu et al. 2020 (So2Sat LCZ42) |
| License | CC-BY-4.0 |

## Facts

- ~500k Sentinel-1 + Sentinel-2 patches across 42+10 cities; 17 Local Climate Zone classes; "Cultural-10" cross-cultural-zone split.

## Hypotheses

1. The Cultural-10 split (cities from different cultural zones) is a *curated* maximum-shift selection rather than a representative *natural* geographic shift.
2. SAR and multispectral provide *complementary* rather than *redundant* signals across the cultural-10 shift.

## Intuitions

1. So2Sat foundation-model results likely exist for 2024–25.
2. Cross-comparisons with BigEarthNet (similar Sentinel data, different geographic coverage) probably exist.
3. LCZ classification updates beyond the original 17-class scheme likely exist.

## Unknowns

1. "Cultural zone" is an entangled handle — urban form correlates with culture but also with climate, history, and regulation; whether the cultural-10 split isolates culture from these confounders cannot be determined from the dataset.
2. The LCZ classification scheme is itself a representational commitment (chosen by Stewart-Oke); finer LCZ subtypes or alternative urban-form taxonomies are not accessible.

## Boundary

Snapshot (T=0); urban form changes slowly but is not exposed temporally. Patch-level only.

## Representational commitments

- 17-class LCZ taxonomy (Stewart-Oke)
- Cultural-10 city selection
- Sentinel-1 + Sentinel-2 dual modality
- 32x32 patch size

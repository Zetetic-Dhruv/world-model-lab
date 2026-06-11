# neon

| Field | Value |
|-------|-------|
| Profile | (P=2, T=3, H=3, C=1, Ty=3) |
| Regime | event + array (remote sensing) |
| Origin | natural |
| Stability | live-updating |
| Member of | [neon-cross-domain](../pairs/neon-cross-domain.json) |
| Source | Keller et al. 2008 BioScience (NEON design) |
| License | CC-BY-4.0 |

## Facts

- 81 field sites across 20 ecoclimatic Domains; standardized 30-year longitudinal collection (NSF, ~10 years collected so far).

## Hypotheses

1. Cross-Domain hold-out is a *concept* shift rather than a *covariate* shift: different Domains exhibit categorically different ecosystem dynamics, not just the same ecological mechanisms under different climate zones.
2. Standardized protocols do not fully eliminate measurement-as-confounder: site-specific factors (technician, equipment age) leak through despite standardization.

## Intuitions

1. NEON's own published cross-Domain analyses likely exist, alongside comparisons with FLUXNET (the smaller-scale earlier eddy-covariance network).
2. Ecological-foundation-model attempts using NEON probably appeared in 2024–25.

## Unknowns

1. The 20 ecoclimatic Domains are themselves a curatorial classification (NEON design choice); whether this captures the natural ecological partition or imposes one cannot be determined from NEON data alone.
2. Live collection is non-uniform (some sites started later), so whether year-as-env is a coherent axis cannot be decided from the metadata.

## Boundary

USA-only (20 Domains all CONUS+AK+PR); no global cross-continent shift accessible. H=3 (richest hierarchy in corpus) but no compositional handle within individual.

## Representational commitments

- 20-Domain ecoclimatic classification (NEON design)
- Standardized protocols across sites
- Three product types (sensor / observational / remote-sensing)
- Hierarchical sampling design

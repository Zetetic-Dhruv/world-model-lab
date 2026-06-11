# assistments

| Field | Value |
|-------|-------|
| Profile | (P=2, T=3, H=3, C=0, Ty=2) |
| Regime | event |
| Origin | natural |
| Stability | stable |
| Member of | [assistments-cross-cohort](../pairs/assistments-cross-cohort.json) |
| Source | Heffernan & Heffernan 2014 IJAIED (ASSISTments) |
| License | CC-BY-4.0 |

## Facts

- 2009/2012/2015/2017 cohorts; US middle/high school math; cross-cohort year-shift canonical OOD.

## Hypotheses

1. Cohort-year shift is primarily *curriculum* shift (math standards changed) rather than *platform* shift (ASSISTments evolved) or *population* shift (different schools enrolled), because standards revisions alter the skill taxonomy directly.
2. Method ranking on 2009→2012 predicts ranking on 2012→2015, because the cohort-shift mechanism is stable across adjacent time-bins.

## Intuitions

1. Foundation-model work using ASSISTments data likely exists.
2. CSEDM workshop benchmarks for 2024–25 probably include these cohorts.

## Unknowns

1. Each cohort is a different *enrolled school set*, not a longitudinal panel; whether year-as-environment conflates time with school cannot be determined from anonymized data.
2. ASSISTments has documented data-quality issues (gameable answers, technician-fixed bugs) that affect different cohort years differently, so cohort-as-environment is confounded with data-quality-fix history.

## Boundary

K-12 US math only; small relative to EdNet but with a richer multi-year design.

## Representational commitments

- K-12 US math curriculum
- 4 cohort years (curator's snapshots)
- Skill-and-problem taxonomy from ASSISTments curriculum
- Data-quality fixes across cohorts (acknowledged confounder)

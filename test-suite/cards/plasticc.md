# plasticc

| Field | Value |
|-------|-------|
| Profile | (P=2, T=3, H=2, C=0, Ty=2) |
| Regime | event |
| Origin | **mixed** (real objects + simulated photometry) |
| Stability | stable |
| Member of | [plasticc-train-test](../pairs/plasticc-train-test.json) |
| Source | Kessler et al. 2019 PASP |
| License | CC-BY-4.0 |

## Facts

- 7,848 training + 3,492,890 test light curves across 14 classes + class-99 unknown; training set non-representative by design (spec-confirmed DDF).

## Hypotheses

1. Class-99 detection is a *novel-class OOD* problem categorically distinct from covariate shift, rather than something strong covariate methods solve incidentally.
2. The deliberate train/test non-representativeness produces a *measurable* shift that methods can characterize, rather than methods merely learning class proxies.

## Intuitions

1. A 2024–25 photometric-classification literature likely exists following LSST first light.
2. ZTF live-test comparisons probably exist that serve as a post-PLAsTiCC reality check.

## Unknowns

1. Class-99 lumps all unknowns into one bucket — uncertainty stratification by type vs by instance is not represented in the schema.
2. The training-label modality (spec-confirmation) categorically differs from the inference modality (photometric); whether this is part of the OOD shift or simply label noise cannot be determined.

## Boundary

Mixed origin (partly simulated); photometric-only; class-99 is monolithic — silent on within-novel-class structure.

## Representational commitments

- LSST passband choice (ugrizy)
- Class-99 as monolithic unknown bucket
- Photometric (not spectroscopic) modality
- DDF-as-train vs WFD-as-test footprint

# nih-chestxray14

| Field | Value |
|-------|-------|
| Profile | (P=0, T=1, H=2, C=2, Ty=2) |
| Regime | blob |
| Origin | natural |
| Stability | stable |
| Pairs with | chexpert · padchest (both form-gated, deferred to batch-2 access) |
| Member of | [cxr-cross-country](../pairs/cxr-cross-country.json) (3-way, deferred) |
| Source | Wang et al. 2017 CVPR; NIH Clinical Center |
| License | Public domain (US govt work) |

## Facts

- 112,120 frontal CXR / 30,805 patients (Bethesda); 14 thorax labels NLP-extracted from radiology reports.

## Hypotheses

1. The label-pipeline shift (NIH NLP vs CheXpert NLP vs PadChest manual) dominates over the population shift, and the three-way pairing can separate the two.
2. Foundation-model performance on NIH exposes a label-signature distinct from the image-signature.

## Intuitions

1. Multi-cohort foundation-model evaluations from 2024–25 (CXR-Foundation, MedImageInsight, BioViL) likely report NIH-specific failure modes.
2. Comparisons that add MIMIC-CXR as a fourth leg probably exist in the literature, though it is excluded from this corpus.

## Unknowns

1. Radiologist-disagreement noise on uncertain multi-label cases cannot be separated from genuine label noise in the data, and there is no established label distinguishing "disagreement-as-data" from "noise-as-data".

## Boundary

T=1 (study-within-patient ordering exists but weakly); cannot drive temporal-shift claims. Process P=0 — no causal chain in the image itself.

## Representational commitments

- Frontal-CXR only — lateral views excluded
- Labels via NLP from reports — label noise tied to NLP pipeline choice
- Study-as-unit rather than patient-trajectory
- 14-class taxonomy is the NIH curators' choice

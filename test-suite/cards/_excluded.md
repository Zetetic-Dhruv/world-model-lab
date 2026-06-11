# Excluded datasets

Datasets considered for the test suite but excluded with documented rationale. Re-admission requires the conditional check listed.

| Dataset | Reason | Conditional admit |
|---------|--------|-------------------|
| MIMIC-III / MIMIC-IV | Excluded per spec | No |
| UK Biobank | Institutional PI required (fails the natural-access criterion) | No |
| Danish national registries | National-system gating | No |
| eICU (PhysioNet) | CITI training ~2h friction; no institutional PI required | Yes — on-demand only; admit when a task needs an EHR dataset beyond what NIH ChestX provides |
| LOBSTER full archive | Paid | No (small free samples may be admitted under a separate `lobster-sample` manifest if a task needs HFT microstructure) |
| Camelyon17 raw WSIs (~700 GB) | Size-only, free | Indirect — admit via the `wilds-camelyon17` preprocessed split |
| CheXpert raw | Stanford form-gated (email) | Yes — admit when the NIH→CheXpert→PadChest 3-way pair is materialized |
| PadChest raw | BIMCV form-gated | Yes — same trigger as CheXpert |
| GDELT 2.0 | Heavy auto-coding noise; CAMEO ontology biases | Low-priority only, `data_quality: auto-coded`, never primary data |
| ICEWS | Human-curated but same CAMEO biases | Low-priority only, `data_quality: human-coded-with-cameo-biases` |
| GOODMotif, GOODCMNIST | Constructed shift on rotated MNIST / planted motif — violates the natural-shift criterion | Synthetic probe only, `data_origin.primary: synthetic`, methodological use only |

## Re-admission protocol

Adding any of the conditional-admit entries:

1. Profile the dataset (P/T/H/C/Ty).
2. Fill in its `epistemic_state` (facts / hypotheses / intuitions / unknowns).
3. Write `manifests/<name>/manifest.json`.
4. Update `_index.json`.
5. Commit with a message documenting what triggered admission.
6. Remove the row from this file (or annotate "admitted YYYY-MM-DD").

## Permanent exclusion

Permanent exclusions (MIMIC family, UK Biobank, Danish registries, GDELT/ICEWS as primary) are documented here for provenance; do not re-litigate without explicit direction.

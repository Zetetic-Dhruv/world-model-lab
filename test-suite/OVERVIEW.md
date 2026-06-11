# Test Suite — Benchmark Overview

The test suite is a curated collection of ~43 publicly accessible datasets that carry **naturally occurring** out-of-distribution (OOD) shift. It is the evaluation **test bench for auto-didactic neuro-symbolic world models** (see [`AUTODIDAXIS.md`](./AUTODIDAXIS.md)): a self-teaching model builds its own theory on a train environment and is asked to generalize to an OOD-test environment whose shift has a genuine real-world cause — testing whether the self-built theory is *real* or merely fit to its training regime. The same structure tests any method for **discovering causal mechanisms, structural change, and broken symmetries** in real scientific and natural data.

Per-dataset records live in `manifests/<dataset>/manifest.json` (machine-readable) and `cards/<dataset>.md` (narrative). This document holds the benchmark-level structure.

## Design goals

A benchmark ⟨datasets, shift-axes, pairs⟩ that:

1. Spans the five structural axes (process / temporal / hierarchical / compositional / typed) at distinct profile points.
2. Carries natural (not synthetic) OOD shifts as honest test articles.
3. Spans enough domains — earth science, biomedical, physics, chemistry, economics, … — to test domain-independence.
4. Is reproducibly accessible: pinned, checksummed, version-stable.

## Inclusion criteria

A dataset is admitted iff it satisfies all ten:

1. **Natural shift.** The shift axis is exogenous — cross-population, cross-time, cross-instrument, cross-composition, cross-mechanism. Resampled, augmented, or random-holdout splits are excluded; they carry no real mechanism change.
2. **Pair-formable.** A single dataset cannot exhibit OOD. The experimental unit is a pair ⟨train_env, ood_test_env⟩ over a shared schema, directional (train → test). `pairs/<pair>.json` is the entry point a method consumes; `manifests/<dataset>.json` is its building block.
3. **Structured profile.** Each dataset carries a profile `(P, T, H, C, Ty) ∈ {0,1,2,3}⁵`. Methods declare the structure they target and are matched to datasets with that profile. See `schemas/asymmetry_profile.md`.
4. **Epistemic annotation.** Each dataset carries machine-readable `facts` / `hypotheses` / `intuitions` / `unknowns` arrays plus a `boundary`. Cards carry the narrative; manifests carry the record.
5. **Stability typing.** Every source is classified `stability ∈ {stable, churning, live-updating, closed-after-period}` and pinned with `snapshot_date`. Churning / live-updating sources are snapshotted; stable sources are referenced by URL + SHA. Without this, a re-run is not the same experiment.
6. **Regime classification.** Each dataset is one of `regime ∈ {event, KG, blob, array}`, which determines materialization: `traces.parquet` for event-like; `edges.parquet` for KG; metadata-only for blob; Zarr-native for array.
7. **Provenance preserved.** `url + doi + snapshot_date + checksums + license + host_type` are all required. This makes the benchmark citable and the experiments re-runnable.
8. **Origin transparency.** `data_origin.primary ∈ {natural, simulator, synthetic, mixed}`:
   - **natural** — real-world measurement; the primary data.
   - **simulator** — physics/process-model output (CMAPSS, Tennessee Eastman, OC20/22, ClimateBench, LHC Olympics, ATLAS HiggsML); admitted, but every claim is conditional on the simulator's structural assumptions.
   - **synthetic** — the shift is *constructed* (rotated MNIST, GOODMotif, GOODCMNIST); a methodological probe only, never primary evidence.
   - **mixed** — a blend (PLAsTiCC real objects + simulated photometry; WeatherBench reanalysis = observation + assimilation model). Described in `data_origin.blend`.
9. **Suite vs dataset.** Some entries (WILDS, OGB, TGB, GOOD, Wild-Time, TDC) are *suites of sub-datasets sharing infrastructure*, not atomic datasets. A suite manifest references its members via `member_of`; each sub-dataset is independently profiled. Suite-level claims require cross-member agreement.
10. **Representational commitments flagged.** Every dataset embeds choices invisible from inside it (catchment partition in CAMELS, paper-as-atom in OGB-MAG, catalyst-as-element-list in OC20). Each manifest lists the major ones in `representational_commitments`.

## Pipeline (per dataset)

| Stage | Operation | Produces |
|-------|-----------|----------|
| fetch | `fetch(dataset)` | `raw/` contents + checksum |
| validate | `validate(dataset)` | schema + checksum conformance |
| build | `build_traces(dataset)` | `traces.parquet` / `edges.parquet` |
| profile | `profile(dataset)` | P/T/H/C/Ty scores in the manifest |
| materialize | `materialize_splits(pair)` | `splits/env_*/` partitions |
| measure | `measure(method, pair)` | performance + structural diagnostics |
| match | `match(method)` | profile-compatible pairs |
| revise | `revise(dataset)` | updated epistemic annotation (new commit) |

## Data model

```
Dataset        := ⟨name, source, regime, data_origin, profile,
                   env_axis, ood_protocol, scale, schema, stability,
                   epistemic_state, representational_commitments,
                   checksums, baselines⟩
Profile        := (P, T, H, C, Ty) ∈ {0,1,2,3}⁵
EnvAxis        := ⟨name, field, values, shift_type, shift_mechanism, pair_with⟩
ShiftType      := covariate | concept | label | mechanism
Pair           := ⟨members, shift_axis, pair_type, train_env, ood_test_env,
                   epistemic_state, baselines⟩
PairType       := cross-population | cross-time | cross-instrument
                | cross-composition | cross-source | cross-language
                | cross-experimental-batch | replication
                | intra-dataset-env-split
Regime         := event | KG | blob | array
Stability      := stable | churning | live-updating | closed-after-period
DataOrigin     := { primary ∈ {natural, simulator, synthetic, mixed}, blend? }
TraceRow       := (trace_id, dataset, entity_id, entity_type,
                   parent_id, parent_type, event_type, t, t_order,
                   env_id, env_axis, split, payload)
EdgeRow        := (edge_id, dataset, head_id, head_type, relation,
                   tail_id, tail_type, source_resource, env_id, payload)
EpistemicState := { facts: [...], hypotheses: [...], intuitions: [...],
                    unknowns: [...], boundary: "..." }
```

JSON Schemas live in `schemas/`; a typed Lean4 specification lives in `Lean4/`.

## Contents

43 dataset entries in `manifests/<dataset>/manifest.json`; ~25 pairs in `pairs/<pair>.json`; narrative entries in `cards/<dataset>.md`. See `_index.json` for the global registry and materialization status.

## Benchmark-level epistemic state

### Facts

- The five-axis profile (P/T/H/C/Ty) fingerprints all 43 entries at distinct profile points.
- The pair unit factors the benchmark into ~25 pairs.
- The four-regime split covers all 43 entries.

### Hypotheses

- The structural profile *predicts* method behavior, not merely *describes* it — testable by matching methods to profiles across the 43 entries.
- Domain-independence holds across all four regimes simultaneously.
- Method-class failures map to specific profile regions, giving a structural account of where each method breaks.
- The joint (C × T) profile region is empty; filling it requires admitting a new dataset that exercises compositional and temporal shift at once.

### Intuitions

- Foundation-model 2024–25 results likely already exist for most core entries.
- Cross-benchmark consistency studies likely exist — does method ranking on WILDS predict ranking on GOOD?
- A curation-as-confounder literature likely exists for ML benchmarks specifically.
- Cross-KG transfer benchmarks (train-Hetionet / test-PrimeKG) likely exist or are emerging.

### Unknowns

- Joint compositional + temporal shift is not exhibited by any current pair (no entry scores ≥2 on both C and T).
- "Curator as a shift dimension" is not yet represented in the schema, though WILDS/GOOD/DrugOOD/TDC all expose curator-chosen splits.
- "Version-as-environment" (KG snapshots: Hetionet frozen, PrimeKG Dec-2023, OGB-MAG 2022) is not yet exposed in the schema, though the discrete-version axis exists across siblings.
- Mixed-origin (reanalysis, sim-augmented) lacks a clean field beyond the `blend` free-text.

## Curation principles

- **Add schema, don't coerce.** If a dataset doesn't fit, the answer is more schema (a new regime, field, or shift_type), not lossy coercion. Adding fields is cheap; coercion is expensive.
- **The profile must do work.** Each structural dimension must differentiate downstream behavior. If a dimension never differentiates, drop it.
- **Matches are conjectures.** Each method-to-dataset match is a conjecture the benchmark tests, not a guarantee.

## Status

43 datasets catalogued; 40 materialized; 3 access-blocked (credential-gated or access-restricted sources). Schemas stable. The Lean4 specification builds.

# Test suite — the bench for real autodidaxis

*This is the **test suite** for [world-model-lab](../). It asks the one question a self-teaching world model can't ask of itself: is the theory it built **real** — does it survive contact with the world, or has it merely overfit the regime it was trained in?*

It is ~43 publicly accessible datasets chosen for **naturally occurring** out-of-distribution (OOD) shift. Each ships with a machine-readable manifest and a four-part **epistemic state** — `facts`, `hypotheses`, `intuitions`, `unknowns` — that is the rubric a self-taught theory gets scored against.

---

## Why a self-teaching world model needs this bench

A world model that "learns by itself" is easy to demo and hard to *trust*. Self-supervised prediction on i.i.d. data tells you whether a model fits its training regime — not whether the theory it built is true of the world. Real autodidaxis has to clear a higher bar, and each part of the test suite exists to probe one piece of it:

1. **It must generalize across genuine mechanism change** — a different population, instrument, time period, or composition, where the underlying mechanism really differs. A self-built theory that only survives synthetic shift is not real. → the test suite admits **natural shift only**; the experimental unit is a directed pair ⟨train environment → OOD-test environment⟩.
2. **It must know what it knows** — propose hypotheses, confirm or refute them against evidence, and *refuse to fabricate* where the data is genuinely silent. → every dataset carries an **epistemic state** (`facts` / `hypotheses` / `intuitions` / `unknowns` + a `boundary` line) — the checkable rubric for the autodidactic loop's output.
3. **It must discover the right structure, at the right timescale** — causal process, temporal extent, hierarchy, composition, typed relations. → every dataset is fingerprinted by a **five-axis profile** `(P, T, H, C, Ty) ∈ {0..3}⁵` — the predictability regimes the L0–L3 hierarchy has to span.
4. **Its symbols must be groundable, and the loop measured — not asserted.** → KG-regime and typed datasets, plus a typed **Lean 4 specification** of the schema, are the substrate for the formal layer and constraint factors; the theorem → constraint loop can be run against pairs and *scored*.

The requirement-by-requirement mapping to the world modeller's design is in **[`AUTODIDAXIS.md`](./AUTODIDAXIS.md)**.

## How a run is scored

```
            train environment                         OOD-test environment
   ┌──────────────────────────────┐         ┌──────────────────────────────────┐
   │  self-teaching world model    │         │  held out — genuine mechanism      │
   │  builds its own theory here    │  ────▶  │  shift, never seen in training     │
   └──────────────────────────────┘         └──────────────────────────────────┘
                                                          │
                                                          ▼
                                   score the emitted theory against the dataset's
                                   epistemic_state rubric:
                                     · does it recover the FACTS?
                                     · does it settle the HYPOTHESES in the right direction
                                       under the shift (not just in-distribution)?
                                     · does it stay silent past the BOUNDARY — no fabrication
                                       in the UNKNOWNS?
```

A model that scores well **in-distribution but collapses across the pair** was memorising a regime, not teaching itself a theory. That gap is the quantity the test suite is designed to expose.

## Structure

```
test-suite/
├── README.md                          ← you are here
├── AUTODIDAXIS.md                     ← maps the bench onto the world modeller's requirements
├── OVERVIEW.md                        ← spec: inclusion criteria, pipeline, data model
├── _index.json                        ← registry + materialization status
├── schemas/                           ← manifest / pair / trace / edges JSON Schemas + profile guide
├── manifests/<dataset>/manifest.json  ← per-dataset record + epistemic_state
├── pairs/<pair>.json                  ← train → OOD-test pair definitions
├── cards/<dataset>.md                 ← per-dataset narrative entries
├── Lean4/                             ← typed specification of the schema (a malformed entry fails to compile)
└── scripts/                           ← fetch / validate / build
```

## Reading order

1. **[`AUTODIDAXIS.md`](./AUTODIDAXIS.md)** — why this is the bench for the world modeller, requirement by requirement.
2. **[`OVERVIEW.md`](./OVERVIEW.md)** — inclusion criteria, pipeline, and data model.
3. **`schemas/asymmetry_profile.md`** — what the `(P, T, H, C, Ty)` scores mean.
4. **`cards/camels-dk.md`** + **`manifests/camels-dk/manifest.json`** — a worked example.

## Status

43 datasets catalogued; 40 materialized; 3 access-blocked. Schemas stable; the Lean 4 specification builds. Whether the autodidactic loop *converges, oscillates, or collapses* is the open research question world-model-lab poses — the test suite is where that gets measured, not settled in advance.

## License & citation

This test suite — its schemas, manifests, cards, scripts, and documentation — is © 2026 Dhruv Gupta under [CC BY 4.0](./LICENSE). Each dataset's underlying data follows its own source license, declared per dataset in `manifest.json` → `source.license`. To cite, see [`CITATION.cff`](./CITATION.cff).

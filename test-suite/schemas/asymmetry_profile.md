# Asymmetry Profile — P/T/H/C/Ty Scoring Guide

Each dataset in the test suite carries a tuple `(process, temporal, hierarchical, compositional, typed)` in `{0,1,2,3}⁵`. This is the **routing field**: methods declare the structure they target, and the benchmark matches them to datasets with that profile.

## Scoring scale

- **0** — absent. The dimension does not exist in the dataset's structure.
- **1** — weak / optional. Present but does not constrain the task.
- **2** — structurally important. Drives a non-trivial part of the dataset's content.
- **3** — dominant. The dataset is largely defined by this asymmetry.

The 0–3 ordinal is deliberate: small enough to assign reliably by inspection, large enough to differentiate.

## Dimensions

### P — Process

An ordered causal chain within an entity's lifetime: disease progression, chemical decay, machining cycle, hydrological flow precip→soil→discharge.

| Score | Criterion | Corpus exemplar |
|-------|-----------|-----------------|
| 0 | No causal chain (single snapshot) | PrimeKG, Hetionet |
| 1 | Weak ordering but no causal chain | M6, FRED-MD |
| 2 | Process exists but data shows endpoints | OC20 (relaxation trajectory), LHC Olympics |
| 3 | Process drives the task | CAMELS, Bosch CNC, CMAPSS |

### T — Temporal

Explicit timestamps. Can exist *without* process (independent measurements at different times) and process can exist *without* timestamps (relaxation steps).

| Score | Criterion | Corpus exemplar |
|-------|-----------|-----------------|
| 0 | Snapshot, no temporal axis | PrimeKG, OC20, So2Sat |
| 1 | Implicit ordering only | NIH ChestX-ray14 (study order within patient) |
| 2 | Timestamps span moderate range | OGB-MAG (year), TGB sub-datasets |
| 3 | Long extent or fine grain dominates | CAMELS (30 y daily), GHCN (175 y), FRED-MD |

### H — Hierarchical

Nesting relationships: catchment → sub-basin, hospital → unit → patient, Domain → site → plot → individual.

| Score | Criterion | Corpus exemplar |
|-------|-----------|-----------------|
| 0 | Flat (single-level entities) | M6 (assets) |
| 1 | Weak hierarchy (one parent type) | TGB (entity → time) |
| 2 | Structured hierarchy (2–3 levels) | CAMELS, OGB-MAG, ChestX |
| 3 | Deep hierarchy (4+ levels) | eICU, EdNet, NEON |

### C — Compositional

Substructure that composes the entity: molecule = atoms+bonds; image = pixels; event = jets+leptons.

| Score | Criterion | Corpus exemplar |
|-------|-----------|-----------------|
| 0 | Atomic entities (no decomposition in data) | CAMELS, PrimeKG, FRED-MD |
| 1 | Weak composition (few labelled sub-parts) | CMAPSS, NEON |
| 2 | Composition present (image-as-pixels, event-as-particles) | ChestX, LHC Olympics, ATLAS HiggsML |
| 3 | Composition drives the task | OC20/OC22 (adsorbate–catalyst), DrugOOD scaffold, ogbg-molhiv |

### Ty — Typed

Heterogeneous entity/edge types: KG with multiple node/edge types, KG with relation taxonomy.

| Score | Criterion | Corpus exemplar |
|-------|-----------|-----------------|
| 0 | Homogeneous (single entity, single relation) | M6, MovieLens (almost) |
| 1 | Weak typing (few categorical labels, uniform structure) | Bosch CNC, METR-LA |
| 2 | Typed structure (several entity/event types) | CAMELS, OC20, ChestX, FRED-MD |
| 3 | Heterogeneous typing dominates | PrimeKG (10 nodes × 30 edges), Hetionet, OGB-MAG, GDELT, Amazon |

## Joint reading

The 5-tuple is a *fingerprint*, not a rank. (3,3,2,0,2) ≠ better-than (0,0,2,0,3); they exercise different structures. Matching uses profile-match, not profile-sum.

**Profile-region coverage** in the test suite (informal):
- **(P=3, T=3, ·, 0, ·)** earth-science: CAMELS family, NEON
- **(P=2, T=0, ·, 3, ·)** chemistry: OC20/OC22
- **(0, 0, ·, 0, 3)** biomedical KG: PrimeKG, Hetionet
- **(0, ≥2, ·, 0, 3)** temporal-KG: OGB-MAG, TGB, GDELT
- **(0, 0, ·, 2, 2)** clinical-imaging: NIH ChestX, Camelyon17
- **(3, 3, ≥2, 0, ≥1)** industrial: Bosch CNC, CMAPSS, TE

**Empty region (a known gap in coverage):** joint (C≥2 × T≥2). No core entry exercises compositional + temporal shift simultaneously.

## How to assign

1. Read the dataset card / paper abstract.
2. Identify the primary structural commitments per dimension.
3. Score conservatively — when uncertain between two adjacent scores, take the lower.
4. Commit. Updates are commits, not edits — the Git history is the audit trail.

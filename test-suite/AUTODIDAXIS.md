# Test Suite ↔ the Auto-Didactic Neuro-Symbolic World Modeller

This document maps the world-modeller's design requirements onto concrete test suite structures, so it is explicit *which part of the bench tests which part of the framework*. Section numbers (§) refer to the world-modeller's `design-docs/explorations.md`.

The framework's deliverable is a system whose **autodidactic loop** (§12.6) lets a model continuously improve its own theory of a world: `world model → discrete tokens → formal layer → theorems → constraint factors → world model`. The test suite does not train that loop's perceptual front-end — it **evaluates the loop's product**: the theory the system ends up holding, and whether that theory is real.

---

## The five framework requirements (§1) → test suite

§1 asks for world models that are (1) short- and long-term dynamical, (2) symbolically interfaceable, (3) modality-agnostic, (4) planning-capable, and (5) auto-didactic. The test suite is the substrate on which (2) and (5) — the parts that are *claims about a theory*, not about a perceptual encoder — become measurable.

| §1 requirement | What it demands | Where test suite tests it |
|---|---|---|
| **Long-term dynamics / consequences** | minute-scale, roughly-symbolic prediction (the "owner will be upset" horizon) | the `hypotheses` and `intuitions` in each `epistemic_state` are exactly these consequence-level claims, stated declaratively and checkable |
| **Symbolic interface** | a discrete, groundable symbol set at the right level | KG-regime + `typed` (Ty≥2) datasets supply real typed predicate vocabularies; the Lean 4 spec types them |
| **Modality-agnostic** | same machinery across vision, sensors, instruments | test suite spans earth-science, biomedical KGs, physics/HEP, chemistry, clinical imaging, economics — one schema, many modalities |
| **Planning-capable** | representations usable downstream | out of scope here (planning is tested in-sim by the framework's own MPC work) — test suite tests the *theory* the planner would consume |
| **Auto-didactic** | the theory improves itself and stays true | **the whole point** — see "the loop" below |

## Predictability regimes (§§3–5) → the `(P,T,H,C,Ty)` profile

§4.1 makes the key point: *different concepts emerge at different timescales*, so the symbolic interface is a learning problem, not a fixed level. The framework's answer is the L0–L3 temporal hierarchy (§5). The test suite fingerprints every dataset by the structure a hierarchy would have to discover:

- **P (process)** — the ordered causal chain a low level must learn (gravity→contact→deformation; precip→soil→discharge).
- **T (temporal)** — temporal extent / grain; the span the upper levels integrate over.
- **H (hierarchical)** — nesting the levels must respect (catchment→sub-basin; hospital→unit→patient).
- **C (compositional)** — substructure (molecule = atoms+bonds; event = particles); what a slot/object representation must factor.
- **Ty (typed)** — heterogeneous entity/edge types; the level at which a symbolic interface (§6) first pays off.

A method declares the regime it targets; test suite routes it to datasets at that profile. The empty `(C≥2 × T≥2)` region (see `schemas/asymmetry_profile.md`) is a known coverage gap, not a claim of completeness.

## The symbolic interface and the formal layer (§6, §11) → regimes + the Lean spec

§11's hard problem is the **axiomatization gap**: going from learned tokens to a *typed* predicate vocabulary (`broken : Vase → Prop`), because formal axioms presuppose stable typed predicates. The test suite front-loads exactly that vocabulary:

- **KG-regime** datasets (PrimeKG, Hetionet, OGB-MAG, …) are already typed node/edge graphs — a ready-made target for §6 Route A (a learned, grounded discrete symbol set) and for §11's PDDL/ASP-first or Lean-as-verifier tiers.
- The **`Lean4/`** specification types the schema itself: a manifest that violates an admission rule *fails to compile*. It is the same discipline §11 wants for plans — invariants checked by a type system, not asserted.
- §11's own staging (PDDL/ASP → Lean-as-verifier → Lean-as-planner) can be derisked on the KG pairs, where the predicate vocabulary is given and the only question is whether a self-derived theory transfers across source resources.

## Constraint factors (§12) → `epistemic_state` + `representational_commitments`

§12 injects top-down knowledge as differentiable **constraint factors** lateral to the prediction loss, and §12.4 names the load-bearing problem: *which latents does a constraint bind to?* test suite supplies pre-stated constraints and the honesty checks §12 needs:

- A dataset's **`facts`** are the constraints a correct theory must satisfy; its **`hypotheses`** are candidate constraints to be confirmed or refuted (§12.5's "data-fit validation"); its **`representational_commitments`** are the modelling choices that, left implicit, cause §12.5's *ontology rot* — test suite names them explicitly so a constraint set can be audited against them.
- The **`unknowns`** are the guardrail against §12.6's *echo-chamber collapse*: places where the loop must not manufacture a theorem. A loop that "proves" something in a dataset's `unknowns` is hallucinating, by construction.

## The autodidactic loop (§12.6) → what test suite actually measures

§12.6's loop has three named failure modes — echo-chamber collapse, constraint accretion, ontology takeover — and the design doc flags that *whether the loop converges, oscillates, or collapses* is its single most interesting open question. The test suite turns that question into a measurement:

1. Run the loop on a pair's **train environment**; let it emit its theory (facts it asserts, hypotheses it settles, constraints it accretes).
2. Evaluate that theory on the **OOD-test environment** of the same pair.
3. Score against the pre-registered `epistemic_state`:
   - **convergence to truth** — recovers `facts`, settles `hypotheses` in the direction the natural shift reveals;
   - **honest boundaries** — stays silent in the `unknowns`, respects the `boundary`;
   - **no ontology takeover** — performance does not *degrade* relative to a non-self-taught baseline as the constraint set grows.

A loop that improves in-distribution while failing (2)–(3) is exhibiting exactly the collapse modes §12.6 warns about. That is the signal test suite is built to surface.

## What test suite does *not* test (honest scope)

- **The perceptual front-end (L0/L1).** Raw-video JEPA training and the §4.3 LeWorldModel anti-collapse recipe are derisked on video corpora (Something-Something-V2, Physion, IntPhys per §9), not here. The test suite begins above the encoder — at the theory the encoder's tokens feed.
- **Embodiment / action-conditioning (§10.1).** test suite pairs are observational; action-conditioned control belongs to the framework's in-sim evaluation.
- **Loop convergence as such.** test suite *measures* convergence-to-truth on held-out natural shift; it does not prove the loop converges. That remains §12.6's open question — which is the point of having a bench at all.

## Requirement-coverage summary

| Framework piece | test suite structure | Status |
|---|---|---|
| §1 long-term / symbolic claims | `epistemic_state` (facts/hypotheses/intuitions/unknowns) | ✅ direct |
| §5 predictability regimes | `(P,T,H,C,Ty)` profile | ✅ direct |
| §6 / §11 typed symbol vocabulary | KG/typed regimes + `Lean4/` spec | ✅ substrate provided |
| §12 constraint validity & honesty | `facts` / `representational_commitments` / `unknowns` | ✅ direct |
| §12.6 loop convergence-to-truth | scored train→OOD-test runs over pairs | ✅ measured |
| §4.3 L0/L1 anti-collapse | — | ⛔ out of scope (video corpora) |
| §7 / §10.1 planning, embodiment | — | ⛔ out of scope (in-sim) |

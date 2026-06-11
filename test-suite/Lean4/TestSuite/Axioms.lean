import TestSuite.Manifest
import TestSuite.Pair

/-!
# TestSuite.Axioms

The TestSuite corpus's ten validity constraints A₁–A₁₀, expressed as `Prop`s.

**Type-level vs Prop-level.** Many constraints are encoded at the type level in
`TestSuite.Schema` and `TestSuite.Manifest` — e.g., `Regime` is an inductive of
exactly four values, `AsymmetryProfile` fields are `Fin 4` (A₃), `Stability`
is exactly four values (A₅), and `DataOrigin`'s `payload` is *dependent* on its
`primary`, so the `mixed`→blend constraint (A₈) is enforced by construction.
For those, the corresponding `Prop` here is tautological (`True`).

Other constraints are content-bearing (non-empty strings, matching enums,
structural shape) and need explicit `Prop` definitions.

**A shift in the content-bearing set (see `../BREAKS.md`).** Two constraints
changed character:
- A₆ *became* content-bearing. With the regime-dependent `schema` field now
  present (BP₁/BP₂), A₆ checks that the declared `regime` matches the
  `schema` variant — it is no longer merely "regime is one of four".
- A₈ *left* the content-bearing set. The `mixed`→blend constraint is now a
  type-level invariant of `DataOrigin` (BP₄), so the `Prop` is tautological.
-/

namespace TestSuite.Axioms

open TestSuite

/-! ## A₁ — Natural-shift criterion -/

/-- A₁: env-axis shift is exogenous (natural/simulator/mixed); synthetic is
inadmissible as primary data. Synthetic-origin datasets are a methodological probe only. -/
def A1_NaturalShift (m : DatasetManifest) : Prop :=
  m.dataOrigin.primary ≠ DataOriginPrimary.synthetic

instance (m : DatasetManifest) : Decidable (A1_NaturalShift m) :=
  by unfold A1_NaturalShift; exact inferInstance

/-! ## A₂ — Pair as the experimental unit -/

/-- A₂: a pair has a non-empty train env, a non-empty OOD test env,
and the two differ. (A single dataset cannot exhibit OOD by itself.) -/
def A2_PairPrimitive (p : DatasetPair) : Prop :=
  p.trainEnv ≠ [] ∧ p.oodTestEnv ≠ [] ∧ p.trainEnv ≠ p.oodTestEnv

instance (p : DatasetPair) : Decidable (A2_PairPrimitive p) :=
  by unfold A2_PairPrimitive; exact inferInstance

/-! ## A₃ — Structured asymmetry (type-level) -/

/-- A₃: the asymmetry profile is well-typed. Tautological because
`AsymmetryProfile` fields are `Fin 4` and so always in 0..3. -/
def A3_StructuredAsymmetry (_m : DatasetManifest) : Prop := True

/-! ## A₄ — boundary statement in the manifest -/

/-- A₄: the boundary string is non-empty (the manifest carries a real
boundary statement, not just placeholder). Uses the dataset-level epistemic state
`EpistemicState .dataset`, whose `boundary` is a `String`. -/
def A4_KUInManifest (m : DatasetManifest) : Prop :=
  m.epistemicState.boundary ≠ ""

instance (m : DatasetManifest) : Decidable (A4_KUInManifest m) :=
  by unfold A4_KUInManifest; exact inferInstance

/-! ## A₅ — Stability typing (type-level) -/

/-- A₅: stability is one of the four declared values. Tautological because
`Stability` is an inductive of exactly four cases. -/
def A5_StabilityTyped (_m : DatasetManifest) : Prop := True

/-! ## A₆ — Regime classification (content-bearing) -/

/-- A₆: the declared `regime` matches the `schema` field's variant. With the
regime-dependent `schema` field now present (BP₁/BP₂ fix), this is no longer
tautological: a manifest whose `regime` is `event` but whose `schema` is
`.kg …` fails the check. Delegates to `DatasetManifest.regimeMatchesSchema`. -/
def A6_RegimeClassified (m : DatasetManifest) : Prop :=
  m.regimeMatchesSchema

instance (m : DatasetManifest) : Decidable (A6_RegimeClassified m) :=
  by unfold A6_RegimeClassified; exact inferInstance

/-! ## A₇ — Source provenance preserved -/

/-- A₇: url, license, hostType all non-empty (the manifest carries real
provenance, not placeholders). -/
def A7_ProvenancePreserved (m : DatasetManifest) : Prop :=
  m.source.url ≠ "" ∧ m.source.license ≠ "" ∧ m.source.hostType ≠ ""

instance (m : DatasetManifest) : Decidable (A7_ProvenancePreserved m) :=
  by unfold A7_ProvenancePreserved; exact inferInstance

/-! ## A₈ — Data origin transparency (now type-level: the BP₄ payoff) -/

/-- A₈: data origin is transparent. **Tautological by construction**: the
`mixed`→blend constraint is enforced at the type level by `DataOrigin`'s
dependent `payload` (a `mixed` origin *must* carry a `String` blend; every
other origin carries `Unit`). The ill-formed states `(natural, some "…")`
and `(mixed, none)` are unconstructible, so no runtime `Prop` is required.
This is the BP₄ payoff — see `../BREAKS.md`. -/
def A8_DataOriginTransparent (_m : DatasetManifest) : Prop := True

/-! ## A₉ — Suite vs dataset distinction -/

/-- A₉: convention is that suite-level manifests reference their
sub-datasets via `memberOf` and/or document them in `representationalCommitments`.
This Prop is content-bearing only for known suites — for now, just `True`
(checked at the corpus level instead). -/
def A9_SuiteOrDataset (_m : DatasetManifest) : Prop := True

/-! ## A₁₀ — Representational-commitment flag -/

/-- A₁₀: representational_commitments must be non-empty (the manifest
exposes the modeling choices the dataset embeds). -/
def A10_RepresentationalCommitments (m : DatasetManifest) : Prop :=
  m.representationalCommitments ≠ []

instance (m : DatasetManifest) : Decidable (A10_RepresentationalCommitments m) :=
  by unfold A10_RepresentationalCommitments; exact inferInstance

/-! ## Composite: well-formed manifest -/

/-- The conjunction of the content-bearing axioms on a single manifest.
The content-bearing set is {A₁, A₄, A₆, A₇, A₁₀}: A₆ joined it
(regime↔schema consistency) and A₈ left it (now type-level). The remaining
type-level axioms (A₃, A₅, A₈, A₉) hold by construction. -/
def WellFormed (m : DatasetManifest) : Prop :=
  A1_NaturalShift m ∧
  A4_KUInManifest m ∧
  A6_RegimeClassified m ∧
  A7_ProvenancePreserved m ∧
  A10_RepresentationalCommitments m

instance (m : DatasetManifest) : Decidable (WellFormed m) :=
  by unfold WellFormed; exact inferInstance

end TestSuite.Axioms

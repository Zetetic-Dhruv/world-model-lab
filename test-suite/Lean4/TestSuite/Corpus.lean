import TestSuite.Manifest
import TestSuite.Pair
import TestSuite.Axioms

/-!
# TestSuite.Corpus

The top-level corpus value, bundling every manifest and pair, mirroring the
corpus specification.

This file represents the corpus AS A LEAN4 VALUE. Constructing a `Corpus`
value enforces structural consistency between the manifests, the pairs, and
the corpus-level notes.

Design decisions (see `../BREAKS.md`):
- The bespoke per-level corpus struct is gone — the corpus epistemic state is the unified
  level-parametric `EpistemicState .corpus` (BP₆). One concept, three levels.
- Corpus/pair properties are stated over the `Prop`-valued predicates
  (`DatasetManifest.Materialized`, `DatasetPair.WellFormedDirection`), not
  the old `… = true` Bool encodings — restoring proof relevance.
-/

namespace TestSuite

/-- The full corpus as a Lean4 value. It bundles the dataset manifests, the
comparison pairs, and a corpus-level epistemic state. The validity constraints
live in `TestSuite.Axioms`; the types in `TestSuite.Schema` and `TestSuite.Manifest`;
the per-dataset and per-pair data in the lists below. -/
structure Corpus where
  /-- Each dataset entry. -/
  manifests : List DatasetManifest
  /-- Each pair (experimental unit per A₂). -/
  pairs     : List DatasetPair
  /-- The corpus-level epistemic state (the unified `EpistemicState` at corpus level;
  the corpus carries no single `boundary`, so the level supplies `Unit`). -/
  epistemicState : EpistemicState .corpus
deriving Repr

/-! ## Corpus-level properties -/

namespace Corpus

/-- All manifests in the corpus are individually well-formed. -/
def allWellFormed (c : Corpus) : Prop :=
  ∀ m ∈ c.manifests, Axioms.WellFormed m

/-- All manifests are materialized (≥1 checksum recorded — `scale.raw_bytes`
has been verified against a real artifact). -/
def allMaterialized (c : Corpus) : Prop :=
  ∀ m ∈ c.manifests, m.Materialized

/-- All pairs satisfy A₂'s direction well-formedness. -/
def allPairsWellFormed (c : Corpus) : Prop :=
  ∀ p ∈ c.pairs, p.WellFormedDirection

/-- Every pair's members are real datasets in the corpus. -/
def allPairsMembersExist (c : Corpus) : Prop :=
  ∀ p ∈ c.pairs, ∀ mname ∈ p.members,
    ∃ m ∈ c.manifests, m.name = mname

/-- The corpus is "publication-ready" — all axiom-derived properties hold
+ all pairs are structurally well-formed + every pair references existing
datasets. -/
def publicationReady (c : Corpus) : Prop :=
  c.allWellFormed ∧ c.allPairsWellFormed ∧ c.allPairsMembersExist

end Corpus

end TestSuite

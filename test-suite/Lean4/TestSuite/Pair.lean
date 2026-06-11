import TestSuite.Manifest

/-!
# TestSuite.Pair

Lean4 mirror of `schemas/pair.schema.json` — the experimental unit per A₂.

Design decisions (see `../BREAKS.md`):
- `PairEpistemicState` removed; uses the unified `EpistemicState .pair` (no duplication).
- `memberArity` (BP₃): `intraDatasetEnvSplit` requires exactly one member,
  every cross-* type requires ≥2. Enforced by the `membersValid` proof field,
  discharged by `decide` at construction — a malformed pair fails the build.
- Direction/arity predicates are `Prop` + `Decidable`, not `Bool`.
-/

namespace TestSuite

/-- The required member-list arity for each pair type (BP₃).
`intraDatasetEnvSplit` is an env split *inside one dataset* (exactly one
member); every cross-* type bridges ≥2 datasets. -/
def PairType.memberArity : PairType → (List String → Prop)
  | .intraDatasetEnvSplit => fun l => l.length = 1
  | _                     => fun l => 2 ≤ l.length

/-- The pair — the unit experiments consume (A₂). The `membersValid` field
ties `members` to `pairType` so a `crossPopulation` pair with `<2` members,
or an `intraDatasetEnvSplit` with `≠1`, is unconstructible. -/
structure DatasetPair where
  schemaVersion : String
  name          : String
  pairType      : PairType
  members       : List String
  /-- Proof that `members` has the arity required by `pairType` (BP₃);
  default discharges it by `decide` for concrete values. -/
  membersValid  : pairType.memberArity members := by decide
  sharedSchema  : Bool := false
  shiftAxis     : String  -- must match members' envAxis.name
  trainEnv      : List String
  oodTestEnv    : List String
  epistemicState       : EpistemicState .pair
  baselines     : List Baseline := []

/-- `Repr` for `DatasetPair` (the `membersValid` proof field defeats
`deriving`; we print the data fields). -/
instance : Repr DatasetPair where
  reprPrec p _ :=
    "DatasetPair " ++ repr p.name
      ++ " type:" ++ repr p.pairType
      ++ " members:" ++ repr p.members
      ++ " train:" ++ repr p.trainEnv
      ++ " test:" ++ repr p.oodTestEnv

/-! ## Decidable predicates (Prop, not Bool) -/

/-- A₂'s direction requirement: non-empty train env, non-empty OOD test env,
and the two differ. -/
def DatasetPair.WellFormedDirection (p : DatasetPair) : Prop :=
  0 < p.trainEnv.length ∧ 0 < p.oodTestEnv.length ∧ p.trainEnv ≠ p.oodTestEnv

instance (p : DatasetPair) : Decidable p.WellFormedDirection := by
  unfold DatasetPair.WellFormedDirection; infer_instance

/-- A cross-dataset pair (≥2 member datasets). -/
def DatasetPair.IsCrossDataset (p : DatasetPair) : Prop :=
  2 ≤ p.members.length

instance (p : DatasetPair) : Decidable p.IsCrossDataset := by
  unfold DatasetPair.IsCrossDataset; infer_instance

/-- An intra-dataset pair (env split inside a single dataset). -/
def DatasetPair.IsIntraDataset (p : DatasetPair) : Prop :=
  p.members.length = 1

instance (p : DatasetPair) : Decidable p.IsIntraDataset := by
  unfold DatasetPair.IsIntraDataset; infer_instance

end TestSuite

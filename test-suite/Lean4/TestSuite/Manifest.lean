import TestSuite.Schema

/-!
# TestSuite.Manifest

Lean4 mirror of `schemas/manifest.schema.json`.

Design decisions (see `../BREAKS.md`):
- `DataOrigin` payload is now **dependent** on `primary` (BP₄): `mixed`
  carries a required `String` blend; other origins carry `Unit`. The
  ill-formed `(natural, some "...")` is now unconstructible.
- `DatasetManifest.schema : RegimeSchema` is now **present** (BP₁/BP₂); A₆
  becomes content-bearing via `regimeMatchesSchema`.
- `checksums : List Checksum` uses the refined `SHA256` (BP₅).
- `epistemicState : EpistemicState .dataset` uses the unified level-parametric epistemic state.
- Predicates (`Materialized`, `IsSimulator`, `IsNatural`,
  `regimeMatchesSchema`) are `Prop` + `Decidable`, not `Bool` — restoring
  proof relevance.
-/

namespace TestSuite

/-- Source provenance per A₇. -/
structure DatasetSource where
  url       : String
  doi       : Option String := none
  hostType  : String
  license   : String
  stability : Stability
deriving Repr

/-- The payload a `DataOrigin` carries, dependent on its primary class:
`mixed` requires a free-text `blend`; everything else carries no payload. -/
abbrev DataOriginPrimary.PayloadType : DataOriginPrimary → Type
  | .mixed => String
  | _      => Unit

/-- Data origin per A₈, with the blend constraint enforced at the type level:
`payload`'s type is `String` exactly when `primary = mixed`. -/
structure DataOrigin where
  primary : DataOriginPrimary
  payload : primary.PayloadType

/-- `Repr` for `DataOrigin` (dependent payload defeats `deriving`). -/
instance : Repr DataOrigin where
  reprPrec o _ := match o.primary, o.payload with
    | .mixed,     b => "DataOrigin.mixed " ++ repr b
    | .natural,   _ => "DataOrigin.natural"
    | .simulator, _ => "DataOrigin.simulator"
    | .synthetic, _ => "DataOrigin.synthetic"

/-- Smart constructors so call sites stay clean and the dependent payload
is discharged automatically. -/
def DataOrigin.natural   : DataOrigin := ⟨.natural,   ()⟩
def DataOrigin.simulator : DataOrigin := ⟨.simulator, ()⟩
def DataOrigin.synthetic : DataOrigin := ⟨.synthetic, ()⟩
def DataOrigin.mixed (blend : String) : DataOrigin := ⟨.mixed, blend⟩

/-- The OOD env axis. -/
structure EnvAxis where
  name           : String
  field          : Option String := none
  values         : List String := []
  shiftType      : ShiftType
  shiftMechanism : String
  pairWith       : List String := []
deriving Repr

/-- One element of an OOD protocol's split sequence. -/
structure OodSplit where
  envId   : String
  purpose : String
  ref     : Option String := none
deriving Repr

/-- The OOD protocol declaring train/test splits over env values. -/
structure OodProtocol where
  name   : String
  ref    : String
  splits : List OodSplit
deriving Repr

/-- Scale tuple; `events`/`rawBytes` may be `none` until materialization. -/
structure Scale where
  entities   : Option Nat
  events     : Option Nat
  rawBytes   : Option Nat
  nEnvLevels : Option Nat
deriving Repr

/-- Optional baseline measurement, populated as evaluation proceeds. -/
structure Baseline where
  name   : String
  ref    : String
  metric : String
  value  : Float
  env    : Option String := none
deriving Repr

/-- For curated suites (WILDS, GOOD, DrugOOD, TDC): the curator's choices. -/
structure CurationProvenance where
  curator    : Option String := none
  ref        : Option String := none
  splitRule  : Option String := none
  motivation : Option String := none
deriving Repr

/-- A dataset manifest — the Lean4 mirror of one
`manifests/<dataset>/manifest.json`. -/
structure DatasetManifest where
  schemaVersion               : String
  name                        : String
  version                     : String
  snapshotDate                : String
  source                      : DatasetSource
  domain                      : String
  regime                      : Regime
  /-- The regime-dependent schema declaration (BP₁/BP₂). Its variant must
  match `regime` — see `regimeMatchesSchema`. -/
  schema                      : RegimeSchema
  dataOrigin                  : DataOrigin
  asymmetryProfile            : AsymmetryProfile
  envAxis                     : EnvAxis
  oodProtocol                 : OodProtocol
  scale                       : Scale
  epistemicState                     : EpistemicState .dataset
  representationalCommitments : List String  -- A₁₀
  derivedFrom                 : List String := []
  memberOf                    : List String := []
  checksums                   : List Checksum := []
  baselines                   : List Baseline := []
  curationProvenance          : Option CurationProvenance := none
deriving Repr

/-! ## Decidable predicates (Prop, not Bool — proof relevance preserved) -/

/-- The manifest is at least partially materialized (≥1 checksum recorded). -/
def DatasetManifest.Materialized (m : DatasetManifest) : Prop :=
  0 < m.checksums.length

instance (m : DatasetManifest) : Decidable m.Materialized := by
  unfold DatasetManifest.Materialized; infer_instance

/-- The manifest is simulator-origin (A₈ caveat applies to all its claims). -/
def DatasetManifest.IsSimulator (m : DatasetManifest) : Prop :=
  m.dataOrigin.primary = DataOriginPrimary.simulator

instance (m : DatasetManifest) : Decidable m.IsSimulator := by
  unfold DatasetManifest.IsSimulator; infer_instance

/-- The manifest is natural-origin (admissible as primary substrate). -/
def DatasetManifest.IsNatural (m : DatasetManifest) : Prop :=
  m.dataOrigin.primary = DataOriginPrimary.natural

instance (m : DatasetManifest) : Decidable m.IsNatural := by
  unfold DatasetManifest.IsNatural; infer_instance

/-- A₆ (content-bearing): the manifest's declared `regime` matches the
variant of its `schema` field. -/
def DatasetManifest.regimeMatchesSchema (m : DatasetManifest) : Prop :=
  m.regime = m.schema.regime

instance (m : DatasetManifest) : Decidable m.regimeMatchesSchema := by
  unfold DatasetManifest.regimeMatchesSchema; infer_instance

end TestSuite

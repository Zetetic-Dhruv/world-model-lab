/-!
# TestSuite.Schema

Foundational types mirroring the JSON Schemas in `../schemas/`.

Each inductive encodes an A-axiom at the type level:
- `Regime` → A₆ (regime classification)
- `Stability` → A₅ (stability typing)
- `DataOriginPrimary` → A₈ (data origin)
- `AsymmetryProfile` → A₃ (the 5-axis structural profile, `Fin 4` bounds)
- `ShiftType` → shift_type enum on `EnvAxis`
- `PairType` → pair_type enum (A₂)

This file also carries the additions documented in `../BREAKS.md`:
- `RegimeSchema` — the regime-dependent `schema` field (BP₁/BP₂)
- `EpistemicState level` — the level-parametric epistemic state (one type replacing three
  near-identical structures, BP₆)
- `SHA256` / `Checksum` — refined checksum type (BP₅)
-/

namespace TestSuite

/-! ## Enumerations (type-level axiom encodings) -/

/-- Storage regime for a dataset (A₆). Determines what gets materialized:
`event`→traces.parquet, `KG`→edges.parquet, `blob`→metadata-only,
`array`→Zarr-native. -/
inductive Regime
  | event
  | KG
  | blob
  | array
deriving Repr, DecidableEq, BEq

/-- Source-host stability (A₅). `stable` → reference by URL+SHA; the rest →
snapshot to GCS. -/
inductive Stability
  | stable
  | churning
  | liveUpdating
  | closedAfterPeriod
deriving Repr, DecidableEq, BEq

/-- Data origin (A₈). `synthetic` is a methodological probe only (A₁). -/
inductive DataOriginPrimary
  | natural
  | simulator
  | synthetic
  | mixed
deriving Repr, DecidableEq, BEq

/-- Shift mechanism classification for the env axis. -/
inductive ShiftType
  | covariate
  | concept
  | label
  | mechanism
deriving Repr, DecidableEq, BEq

/-- Pair-type taxonomy (A₂). -/
inductive PairType
  | crossPopulation
  | crossTime
  | crossInstrument
  | crossComposition
  | crossSource
  | crossLanguage
  | crossExperimentalBatch
  | replication
  | intraDatasetEnvSplit
deriving Repr, DecidableEq, BEq

/-- The 5-asymmetry profile (P/T/H/C/Ty) per A₃. `Fin 4` enforces 0..3 at
the type level — A₃ is satisfied by construction, not by a runtime check. -/
structure AsymmetryProfile where
  process       : Fin 4
  temporal      : Fin 4
  hierarchical  : Fin 4
  compositional : Fin 4
  typed         : Fin 4
deriving Repr

/-- Sum of the five dimension scores (in 0..15). `abbrev` for proof
transparency: `decide` unfolds it without an explicit `unfold`. -/
abbrev AsymmetryProfile.totalScore (p : AsymmetryProfile) : Nat :=
  p.process.val + p.temporal.val + p.hierarchical.val
    + p.compositional.val + p.typed.val

/-- Max of the five dimension scores. `abbrev` for the same reason. -/
abbrev AsymmetryProfile.maxScore (p : AsymmetryProfile) : Nat :=
  Nat.max p.process.val (Nat.max p.temporal.val
    (Nat.max p.hierarchical.val (Nat.max p.compositional.val p.typed.val)))

/-! ## Refined checksum type (BP₅ fix)

`List (String × String)` admitted ill-formed pairs like `("foo", "bar")`.
A `Checksum` carries a `SHA256` whose well-formedness is a refinement proof,
discharged at construction time by `native_decide`. A malformed SHA fails the
build. (Kernel `decide` cannot reduce `String.length`/`String.toList` on string
*literals*, so the guarantee is compiler-checked — see BP₅ in `../BREAKS.md`.) -/

/-- Decidable test for the canonical `"sha256:<64 lowercase hex>"` form.
Total length is 7 ("sha256:") + 64 = 71. We work over `toList` to avoid the
slice-length API. -/
def isSha256 (s : String) : Bool :=
  s.length == 71 &&
  s.startsWith "sha256:" &&
  (s.toList.drop 7).all (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f'))

/-- A SHA256 string refined to the canonical form. `Subtype` inherits `Repr`
from `String`, so this composes with `deriving Repr` downstream. -/
def SHA256 := { s : String // isSha256 s = true }

instance : Repr SHA256 := ⟨fun x n => reprPrec x.val n⟩

/-- A path → checksum binding. The `sha` field's proof obligation is
discharged by `native_decide` at construction — a malformed checksum fails to
compile. -/
structure Checksum where
  path : String
  sha  : SHA256
deriving Repr

/-- Smart constructor: `mkChecksum "path" "sha256:<hex>"` with the
well-formedness proof discharged by `native_decide`. (`decide` cannot reduce
`String.length`/`String.toList` on string *literals* in the kernel, so the
digest-format guarantee is compiler-checked — see BP₅ in `../BREAKS.md`.) -/
def mkChecksum (path sha : String) (h : isSha256 sha = true := by native_decide) :
    Checksum :=
  ⟨path, ⟨sha, h⟩⟩

/-! ## Regime-dependent schema (BP₁/BP₂ fix)

The JSON `schema` field is regime-dependent. Representing it as a sum type
makes A₆ content-bearing: a manifest's `regime` must match its `schema`
variant (`DatasetManifest.regimeMatchesSchema`). -/

/-- Schema fields for an `event`-regime dataset. -/
structure EventSchema where
  entityTypes      : List String
  eventTypes       : List String
  staticAttributes : List String := []
deriving Repr

/-- Schema fields for a `KG`-regime dataset. -/
structure KGSchema where
  nodeTypes : List String
  edgeTypes : List String
deriving Repr

/-- Schema fields for a `blob`-regime dataset (metadata-only traces). -/
structure BlobSchema where
  metadataFields : List String
deriving Repr

/-- Schema fields for an `array`-regime dataset (Zarr-native). -/
structure ArraySchema where
  dimensions : List String
  variables  : List String := []
  zarrNative : Bool := true
deriving Repr

/-- The regime-dependent `schema` field. The constructor used pins the
regime; `RegimeSchema.regime` recovers it for the A₆ consistency check. -/
inductive RegimeSchema
  | event (s : EventSchema)
  | kg    (s : KGSchema)
  | blob  (s : BlobSchema)
  | array (s : ArraySchema)
deriving Repr

/-- Recover the regime implied by a schema variant. -/
def RegimeSchema.regime : RegimeSchema → Regime
  | .event _ => .event
  | .kg _    => .KG
  | .blob _  => .blob
  | .array _ => .array

/-! ## Level-parametric epistemic state

Previously there were three near-identical structures, one per owner level —
needless duplication. They are one concept parameterized by *owner level*. Only
the dataset level carries a single `boundary` statement (A₄); pair/corpus levels
do not — encoded by a dependent `boundary` field. See BP₆ in `../BREAKS.md`. -/

/-- The owner level of an epistemic state. -/
inductive EpistemicLevel
  | dataset
  | pair
  | corpus
deriving Repr, DecidableEq, BEq

/-- Only dataset-level notes carry a single `boundary` statement.
`abbrev` so the field type reduces during instance resolution. -/
abbrev EpistemicLevel.BoundaryType : EpistemicLevel → Type
  | .dataset => String
  | _        => Unit

/-- The epistemic-state block, parameterized by owner level. The four lists
hold established `facts`, testable `hypotheses`, `intuitions` about likely
external work, and structural `unknowns`. The `boundary` field is `String` at
dataset level and `Unit` elsewhere (A₄). -/
structure EpistemicState (level : EpistemicLevel) where
  facts      : List String := []
  hypotheses : List String := []
  intuitions : List String := []
  unknowns   : List String := []
  boundary   : level.BoundaryType

/-- `Repr` for the dependent `EpistemicState`. Deriving fails to synthesize
`Repr level.BoundaryType` for symbolic `level`, so we provide the instance
with that hypothesis explicit; at concrete levels it resolves to
`Repr String` / `Repr Unit`. -/
instance {level : EpistemicLevel} [Repr level.BoundaryType] : Repr (EpistemicState level) where
  reprPrec s _ := repr (s.facts, s.hypotheses, s.intuitions, s.unknowns, s.boundary)

/-- Smart constructor for a dataset-level epistemic state (boundary required). -/
def EpistemicState.mkDataset (facts hypotheses intuitions unknowns : List String)
    (boundary : String) : EpistemicState .dataset :=
  ⟨facts, hypotheses, intuitions, unknowns, boundary⟩

/-- Smart constructor for a pair-level epistemic state (no boundary). -/
def EpistemicState.mkPair (facts hypotheses intuitions unknowns : List String) : EpistemicState .pair :=
  ⟨facts, hypotheses, intuitions, unknowns, ()⟩

/-- Smart constructor for a corpus-level epistemic state (no boundary). -/
def EpistemicState.mkCorpus (facts hypotheses intuitions unknowns : List String) : EpistemicState .corpus :=
  ⟨facts, hypotheses, intuitions, unknowns, ()⟩

end TestSuite

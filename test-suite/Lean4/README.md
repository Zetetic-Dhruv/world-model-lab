# Test Suite — a typed specification of the corpus

*The test suite corpus is ~43 publicly available datasets chosen for **naturally
occurring** distribution shift. Each dataset ships with a JSON **manifest**
recording its provenance, structure, shift axis, scale, and a four-part
epistemic annotation. This directory re-expresses that manifest schema in Lean 4 — as
types and decidable predicates. The point is not redundancy. A JSON Schema
validates a manifest after the fact; these types make a malformed manifest
**fail to compile**. Conformance becomes typechecking, and the corpus's
admission rules become machine-checked propositions.*

**Status.** Builds clean on Lean `v4.30.0`. No Mathlib dependency — the
specification is small and compiles in seconds. One dataset (CAMELS-DK) is
carried as a fully worked, machine-checked example; the other 42 follow the
same recipe. Companion design notes live in [`BREAKS.md`](./BREAKS.md).

---

## What's here

The library is split in two:

- **`TestSuite/`** — the *specification*. Enumerations, record types for a
  manifest and its parts, the row shapes of the data files, and the ten
  validity constraints A₁–A₁₀ as decidable predicates.
- **`Examples/`** — *instances*. Each real dataset written as a Lean value.
  A dataset's manifest typechecking is its first conformance gate; a handful
  of one-line lemmas then check it against the constraints by computation.

The interesting engineering is in a handful of fields where a record "with the
right field types" is not enough — where one field constrains another, or a
string must have a particular shape. Those decisions, and the alternatives
weighed against each, are documented in [`BREAKS.md`](./BREAKS.md) and
summarised below.

## Repository layout

```
Lean4/
├── lean-toolchain        leanprover/lean4:v4.30.0
├── lakefile.lean         Lake config — two libraries, no Mathlib
├── README.md             this file
├── BREAKS.md             design notes: each non-obvious type choice + alternatives
├── TestSuite.lean             umbrella import of the whole specification
├── TestSuite/
│   ├── Schema.lean       enums, AsymmetryProfile, RegimeSchema, SHA256/Checksum, EpistemicState
│   ├── Manifest.lean     DatasetManifest + sub-records + manifest predicates
│   ├── Pair.lean         DatasetPair (a train/OOD comparison) + pair predicates
│   ├── Trace.lean        TraceRow / EdgeRow — row shapes for the data files
│   ├── Axioms.lean       the ten validity constraints A₁–A₁₀ as predicates
│   └── Corpus.lean          Corpus — the whole corpus as one value + corpus-wide checks
├── Examples.lean         umbrella import for the Examples library
└── Examples/
    └── CamelsDk.lean     CAMELS-DK as a value + 9 machine-checked conformance lemmas
```

## The type architecture

Four layers, bottom up.

**1. Enumerations.** The closed vocabularies are `inductive`s, so an illegal
value cannot be named. `Regime` (`event | KG | blob | array`) classifies how a
dataset is stored; `Stability`, `DataOriginPrimary`, `ShiftType`, and
`PairType` cover host stability, data origin, the kind of shift, and the kind
of comparison. The five-axis structural fingerprint is a record of `Fin 4`,
so each axis is *constructed* in `0..3` — no range check needed:

```lean
structure AsymmetryProfile where
  process       : Fin 4
  temporal      : Fin 4
  hierarchical  : Fin 4
  compositional : Fin 4
  typed         : Fin 4
```

**2. The manifest.** `DatasetManifest` bundles the sub-records. Most fields are
ordinary; four carry the design weight (`schema`, `dataOrigin`, `epistemicState`,
`checksums`) and are discussed in the next section.

```lean
structure DatasetManifest where
  name             : String
  regime           : Regime
  schema           : RegimeSchema          -- its shape must match `regime`
  dataOrigin       : DataOrigin            -- dependent record (see below)
  asymmetryProfile : AsymmetryProfile      -- five axes, each 0–3
  envAxis          : EnvAxis               -- the distribution-shift axis
  oodProtocol      : OodProtocol           -- the train / OOD-test split
  scale            : Scale
  epistemicState   : EpistemicState .dataset            -- per-dataset epistemic state
  checksums        : List Checksum := []   -- refined sha256 bindings
  -- … domain, source, representationalCommitments, baselines, provenance …
deriving Repr
```

Manifest-level questions are decidable predicates, not booleans, so they can be
used both as runtime checks and as the hypotheses of later proofs:

```lean
def DatasetManifest.Materialized        (m) : Prop := 0 < m.checksums.length
def DatasetManifest.IsNatural           (m) : Prop := m.dataOrigin.primary = .natural
def DatasetManifest.regimeMatchesSchema (m) : Prop := m.regime = m.schema.regime
```

**3. Pairs and rows.** `DatasetPair` is a single train/OOD comparison — the
unit an experiment actually consumes. `TraceRow` and `EdgeRow` (in
`Trace.lean`) are the row shapes of the on-disk data: an event row tied to an
entity and tagged with its environment, and a typed knowledge-graph edge
tagged with its source resource. The canonical encoding is Parquet; these
types are the row contracts.

**4. The corpus.** `Corpus` bundles every manifest and every pair into one
value, with whole-corpus properties stated over it — e.g. *every pair's
members are real datasets in the corpus*:

```lean
def Corpus.allPairsMembersExist (c : Corpus) : Prop :=
  ∀ p ∈ c.pairs, ∀ name ∈ p.members, ∃ m ∈ c.manifests, m.name = name
```

## Making illegal states unrepresentable

A JSON Schema can say *this field is a string*. It cannot easily say *this
field's value is constrained by that field's value*, and it cannot say *this
string is a valid sha-256 digest* without a regex it then hopes nobody edits.
Those are exactly the manifest's load-bearing invariants. Here is how the types
carry them. Each choice — including the alternatives rejected — is recorded in
[`BREAKS.md`](./BREAKS.md).

### Origin and blend — a dependent record

Only one data origin, `mixed`, needs a free-text description of *how* it was
mixed. A flat `blend : Option String` admits two nonsense states at once: a
blend attached to a non-mixed origin, and a `mixed` origin with no blend. So
the payload's **type** depends on the origin:

```lean
abbrev DataOriginPrimary.PayloadType : DataOriginPrimary → Type
  | .mixed => String
  | _      => Unit

structure DataOrigin where
  primary : DataOriginPrimary
  payload : primary.PayloadType

def DataOrigin.natural : DataOrigin := ⟨.natural, ()⟩
def DataOrigin.mixed (blend : String) : DataOrigin := ⟨.mixed, blend⟩
```

`DataOrigin.mixed "…"` demands a string. `DataOrigin.natural` cannot take one.
The constraint "a mixed dataset declares its blend" (A₈) is now a fact about
the type — there is nothing left to check at runtime.

### Pair arity — a proof carried in the data

A within-dataset split has exactly one member; every cross-dataset comparison
has at least two. The required count depends on the pair's kind. So the
structure carries a small proof, defaulted by computation, that the two agree:

```lean
def PairType.memberArity : PairType → (List String → Prop)
  | .intraDatasetEnvSplit => fun l => l.length = 1
  | _                     => fun l => 2 ≤ l.length

structure DatasetPair where
  pairType     : PairType
  members      : List String
  membersValid : pairType.memberArity members := by decide
  -- …
```

Write a cross-population pair with one member and the file does not build: the
default `by decide` cannot produce the missing proof.

### Checksums — and a detour through the kernel

The naive type for content hashes is `List (String × String)`. It accepts
`("hello", "world")`. So the digest is refined to a subtype, and a smart
constructor discharges the proof:

```lean
def isSha256 (s : String) : Bool :=
  s.length == 71 && s.startsWith "sha256:" &&
  (s.toList.drop 7).all (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f'))

def SHA256 := { s : String // isSha256 s = true }

structure Checksum where
  path : String
  sha  : SHA256

def mkChecksum (path sha : String) (h : isSha256 sha = true := by native_decide) :
    Checksum := ⟨path, ⟨sha, h⟩⟩
```

The obvious way to discharge that proof is `by decide`. It does not work.
Lean's kernel special-cases `String` *equality* on literals — which is why the
"non-empty" checks elsewhere reduce fine — but it will not run `String.length`
or `String.toList` over a literal. `decide` gets stuck. The fix is
`native_decide`, which compiles `isSha256` and runs it. That trades a kernel
check for a compiler check (it adds `Lean.ofReduceBool` to that call site, and
*only* there). The guarantee survives — a malformed digest still fails the
build — but where the guarantee is enforced moved. That is the kind of thing
[`BREAKS.md`](./BREAKS.md) exists to record; the kernel-pure alternative (the
digest as 64 typed nibbles) is noted there as the swap to make if an audited,
compiler-free build is ever required.

### One epistemic state, three levels — a parameterised type

Every manifest, every pair, and the corpus all carry the same four lists —
facts, hypotheses, intuitions, and unknowns. Only a *dataset* additionally carries a one-line
statement of what it cannot evidence. Three near-identical structures would be
duplication; one structure with a field that is always `Unit` would be a lie.
Instead, one type is indexed by level, and the boundary field's type depends on
that index:

```lean
inductive EpistemicLevel | dataset | pair | corpus

abbrev EpistemicLevel.BoundaryType : EpistemicLevel → Type
  | .dataset => String
  | _        => Unit

structure EpistemicState (level : EpistemicLevel) where
  facts hypotheses intuitions unknowns : List String := []  -- the four lists
  boundary : level.BoundaryType                             -- String for datasets, absent otherwise
```

### The full set

| Concern | Naive type (and what it admits) | Chosen type |
|---|---|---|
| Schema block | absent entirely | `RegimeSchema` sum type (one shape per regime) |
| Regime vs. schema | two free fields that can disagree | `regimeMatchesSchema` predicate (constraint A₆) |
| Pair arity | `List String` (any length) | tag-dependent `memberArity` proof field |
| Origin blend | `Option String` (`(natural, some …)`) | dependent `payload` (constraint A₈) |
| Checksum | `String × String` (`("foo","bar")`) | `SHA256` refinement + `mkChecksum` |
| Epistemic state | three duplicate structs | one level-indexed `EpistemicState` |
| Schema detail | full nested field/unit typing | deliberately flattened to name-lists |

The last row is a *scope* decision, not an oversight: the Lean side checks that
the schema block exists and matches the regime; the JSON Schema checks
field-level detail. [`BREAKS.md`](./BREAKS.md) gives the reasoning and the
condition under which that line should move.

## The validity constraints (A₁–A₁₀)

`TestSuite/Axioms.lean` states the corpus's ten admission rules. Some are enforced
by the types and so are tautologies here; the rest are decidable predicates.

| | Rule | How it is enforced |
|---|---|---|
| A₁ | origin is not synthetic | predicate |
| A₂ | a pair has non-empty, *distinct* train and OOD environments | predicate |
| A₃ | the five-axis profile is in range | type-level (`Fin 4`) |
| A₄ | the dataset carries a non-empty boundary note | predicate |
| A₅ | host stability is one of four values | type-level (`inductive`) |
| A₆ | the declared regime matches the schema's shape | predicate (`regimeMatchesSchema`) |
| A₇ | source URL, license, and host type are all non-empty | predicate |
| A₈ | a mixed-origin dataset declares its blend | type-level (dependent payload) |
| A₉ | suites reference their member datasets | corpus-level convention |
| A₁₀ | the dataset's representation choices are listed | predicate |

The content-bearing checks combine into one decidable conjunction:

```lean
def WellFormed (m : DatasetManifest) : Prop :=
  A1_NaturalShift m ∧ A4_KUInManifest m ∧ A6_RegimeClassified m ∧
  A7_ProvenancePreserved m ∧ A10_RepresentationalCommitments m
```

A₃, A₅, and A₈ are absent from this list on purpose: they hold by construction,
so there is nothing to decide. A₉ is a property of the corpus, not a single
manifest, and is checked over `Corpus`.

## Conformance by example

`Examples/CamelsDk.lean` writes the CAMELS-DK manifest as a Lean value. That it
typechecks is the first test — every JSON field must have a home of the right
type, and any drift between the JSON and the spec surfaces as a compile error.
Nine lemmas then check it by computation:

```lean
example : Axioms.A1_NaturalShift   camelsDk := by decide   -- not synthetic
example : Axioms.A6_RegimeClassified camelsDk := by decide -- regime matches schema
example : Axioms.WellFormed        camelsDk := by decide   -- all content checks pass
example : camelsDk.Materialized              := by decide   -- a checksum is present
example : camelsDk.regime = Regime.event     := rfl
```

Because these are `example`s, a broken invariant is a broken build. The same
file is the template for every other dataset.

## Build

```bash
cd Lean4
lake build
```

Lake fetches the toolchain pinned in `lean-toolchain` (`v4.30.0`) and compiles
both libraries. A successful run ends with `Build completed successfully`.

## Extending

To add a dataset's mirror:

1. Read its `manifests/<dataset>/manifest.json`.
2. Create `Examples/<DatasetName>.lean` with
   `def <name> : DatasetManifest := { … }`, using `.event {…}` / `.kg {…}` /
   etc. for the `schema` field and `mkChecksum` for hashes.
3. Add `example : Axioms.WellFormed <name> := by decide` to gate it.
4. Import it from `Examples.lean`.
5. (Optional) state and prove dataset-specific properties.

If the literal will not typecheck, the manifest and the spec disagree — fix
whichever is wrong before moving on.

## Scope and limitations

- **Static, not a parser.** This mirrors the schema; it does not yet read JSON
  at runtime. A `Lean.FromJson` layer is the natural next addition.
- **Row types are contracts, not validators.** `TraceRow`/`EdgeRow` fix the
  on-disk row shape; enforcing it against actual Parquet is out of scope here.
- **Schema detail is flattened by choice** (see the table above and
  `BREAKS.md`).
- **One worked example so far.** CAMELS-DK is complete; the remaining 42
  manifests and the pairs are the obvious next batch, each gated by its own
  typecheck.
- **Mathlib-free, deliberately.** The specification stays light and fast;
  Mathlib can be added if and when corpus-level theorems need it.

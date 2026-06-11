# Design decisions — the test suite type mirror

This document records the type-design decisions behind the Lean mirror: the
places where a straightforward record "with the right field types" was *not*
enough, and a more precise type was chosen instead. For each decision it gives
the problem, the type that was picked, the alternatives weighed against it, and
the condition under which the decision should be revisited.

The seven are labelled **BP₁–BP₇** and are referenced by those labels from
comments in the `.lean` sources. Think of them as architecture decision
records for the type layer.

An earlier pass of this mirror recorded no such decisions at all: every field
was a flat record, plain `def`s and booleans stood in for propositions, and
nothing in the types distinguished a *valid* manifest from a merely
*well-typed* one. The decisions below are the result of tightening that — the
recurring theme is **make illegal states unrepresentable**, and where that
isn't possible, **make the residual check decidable and explicit**.

---

## BP₁ — The regime-dependent `schema` field

**Problem.** The JSON `schema` block has a categorically different shape per
regime: an `event` dataset has `entity_types` / `event_types`; a `KG` dataset
has `node_types` / `edge_types`; a `blob` dataset has `metadata_fields`; an
`array` dataset has `dimensions` / `variables`. The first version of the mirror
left the field out entirely — a whole JSON key with no home in the types. The
four shapes do not collapse into one record without a pile of `Option` fields
that are individually meaningless.

**Decision** (`Schema.lean`): a sum type, one constructor per regime.
```lean
inductive RegimeSchema
  | event (s : EventSchema) | kg (s : KGSchema)
  | blob  (s : BlobSchema)  | array (s : ArraySchema)
```

**Alternatives.**
- *Flat optionals* — `structure Schema where entityTypes : Option … ; nodeTypes
  : Option … ; …`. Rejected: every reader has to inspect which `Option`s are
  populated, which puts the case analysis back at runtime and loses the
  "exactly one shape" guarantee.
- *Indexed family* — `def RegimeSchema : Regime → Type`, one case each. This is
  stronger (the index *is* the regime), but it forces the manifest's `schema`
  field to depend on its `regime` field, which couples field order and defeats
  `deriving Repr`. Held in reserve — see BP₂.

**Revisit when** a function that dispatches on regime needs the schema's *type*
(not just a runtime check) to guarantee they agree — e.g. a `materialize` step
whose output type varies by regime. At that point switch to the indexed family
and make `schema` depend on `regime`.

---

## BP₂ — Keeping `regime` and `schema` in agreement (constraint A₆)

**Problem.** Even with BP₁ in place, the flat `regime : Regime` field and the
`schema : RegimeSchema` variant can still disagree (`regime := .event` with
`schema := .kg …`). No type makes them equal *without deleting one of them* —
but the JSON carries both on purpose (the flat `regime` drives routing and is
what a human skims first), so both stay, bridged by a decidable check. The only
thing relating the two is the shared `Regime` enum.

**Decision** (`Manifest.lean` + `Axioms.lean`): recover the schema's implied
regime and assert equality as constraint A₆.
```lean
def RegimeSchema.regime : RegimeSchema → Regime | .event _ => .event | …
def DatasetManifest.regimeMatchesSchema (m) : Prop := m.regime = m.schema.regime
def A6_RegimeClassified (m) : Prop := m.regimeMatchesSchema  -- was `True`
```
A₆ thus changed from a tautology ("regime is one of four", already guaranteed
by the `Regime` enum) into a real check. `decide` settles it per manifest, and
a mismatched manifest fails `WellFormed`.

**Alternative.** Delete the flat `regime` field and *define* `m.regime :=
m.schema.regime`. That makes A₆ vacuous again (nothing to disagree) but throws
away the flat field the JSON and routing code rely on.

**Revisit when** the redundancy starts causing bugs in practice (manifests
routinely built with mismatched fields). Then adopt the alternative together
with BP₁'s indexed family: one `schema`, with `regime` derived from it.

---

## BP₃ — Pair member count

**Problem.** A within-dataset split (`intraDatasetEnvSplit`) has exactly **one**
member. Every cross-dataset comparison has **two or more**. The required count
depends on the pair's kind, so a plain `List String` admits nonsense: a
cross-population pair with one member, or a within-dataset split with five.

**Decision** (`Pair.lean`): a predicate keyed by the tag, carried in the
structure as a proof that defaults by computation.
```lean
def PairType.memberArity : PairType → (List String → Prop)
  | .intraDatasetEnvSplit => fun l => l.length = 1
  | _                     => fun l => 2 ≤ l.length
structure DatasetPair where
  …
  membersValid : pairType.memberArity members := by decide
```
A malformed pair fails to compile: the default `by decide` cannot produce the
proof.

**Alternatives.**
- *Two constructors* — split into `IntraPair` (one member) and `CrossPair` (a
  list with a `2 ≤ length` proof). A cleaner guarantee, but it doubles every
  function over pairs, and the JSON has a single `pair_type` field, not two
  shapes — so it costs at the JSON boundary.
- *Sized vectors* — `members : Vector String n`. Overkill; members are never
  indexed positionally.

**Revisit when** proofs about pairs routinely need to case on the member count
(e.g. theorems that only hold for cross-dataset pairs). Then move the count
into the type with two constructors.

---

## BP₄ — Origin and blend

**Problem.** Only a `mixed` data origin needs a `blend` description; `natural`,
`simulator`, and `synthetic` do not. A flat `blend : Option String` admits two
nonsense states at once: a blend attached to a non-mixed origin
(`(natural, some "…")`), and a `mixed` origin with no blend (`(mixed, none)`).
The original constraint A₈ was a runtime check patching the second hole and
ignoring the first.

**Decision** (`Manifest.lean`): make the payload's *type* depend on the origin.
```lean
abbrev DataOriginPrimary.PayloadType : DataOriginPrimary → Type
  | .mixed => String | _ => Unit
structure DataOrigin where
  primary : DataOriginPrimary
  payload : primary.PayloadType
```
Both nonsense states are now unconstructible. This is *why* A₈ could drop from a
runtime check to a tautology — the work is done by the type. Smart constructors
(`DataOrigin.natural`, `DataOrigin.mixed "…"`) keep call sites tidy.

**Alternative.** A subtype over the flat record, `{ o // o.wellFormed }`. It
keeps the flat shape but re-introduces exactly the proof obligation A₈ used to
carry — so it does not buy the simplification. Rejected.

**Revisit when** some other origin needs a structured (non-`String`) payload —
say `simulator` carrying a config. Then just extend the `PayloadType` match;
the dependent design absorbs it without a reshape. A real reversal (back to a
flat subtype) is forced only if `DataOriginPrimary` itself must become
open/extensible.

---

## BP₅ — Checksum digests

**Problem.** The naive type for content hashes, `List (String × String)`,
accepts `("foo", "bar")` — neither a path nor a digest. The format
(`"sha256:"` followed by 64 lowercase hex characters) is a real invariant a
raw `String` cannot express.

**Decision** (`Schema.lean`): a refinement subtype plus a smart constructor.
```lean
def isSha256 (s : String) : Bool := …            -- length 71, prefix, hex tail
def SHA256 := { s : String // isSha256 s = true }
structure Checksum where path : String; sha : SHA256
def mkChecksum (path sha) (h : isSha256 sha = true := by native_decide) : Checksum := …
```
A malformed digest fails the build, because the default proof cannot be
produced.

**A subtlety found while building this: the proof needs `native_decide`, not
`decide`.** Lean's kernel special-cases `String` *equality* on literals — which
is why the "non-empty string" checks elsewhere (A₄, A₇) reduce fine under plain
`decide` — but it will not run `String.length` or `String.toList` over a
literal. `decide` gets stuck. So any check on a string's *content* (length,
prefix, per-character class) is out of reach for `decide` on a literal. The
consequence: the digest guarantee is enforced by the **compiler**
(`native_decide` compiles and runs `isSha256`), not by the kernel. This adds
`Lean.ofReduceBool` to the trusted base — but only at `mkChecksum` call sites.
The conformance lemmas (A₄, A₆, `WellFormed`, `Materialized`, …) never force the
digest proof: `List.length [x]` reduces without ever inspecting `x`. So they
stay `decide`-only.

**Alternatives.**
- *Typed nibbles* — represent the hex body as `Vector (Fin 16) 64`. Structurally
  exact, and crucially **kernel-checkable** (no `native_decide`, no
  `ofReduceBool`), because validity is the *type* rather than a check over a
  literal. Rejected for now: parsing the JSON string into 64 nibbles is a lossy
  bridge for a benefit nothing currently needs (we never compute on the
  nibbles). But it is the natural switch if kernel purity is ever required.
- *Plain `String`, validate in CI* — the JSON Schema already runs a regex.
  Rejected for the mirror, because it re-opens the very hole the mirror exists
  to close (a type that can hold an invalid value).

**Revisit when** either:
- *More algorithms* — if digests other than sha-256 are needed (blake3, …),
  generalise `isSha256` to take the algorithm and index `SHA256` by it. Trigger:
  a real artifact whose digest the current type can't represent.
- *Kernel purity* — if the project must drop the compiler from its trusted base
  (an audited release where the proof term must be `ofReduceBool`-free), switch
  to typed nibbles so validity is structural. Here the trigger is the trust
  boundary, not the mathematics.

---

## BP₆ — One epistemic state, three levels

**Problem.** A manifest, a pair, and the whole corpus each carry the same four
note-lists (established facts; open questions the data raises; likely-relevant
external work; structural unknowns). But only a *dataset* additionally carries a
one-line statement of what it cannot evidence. The original mirror had three
near-identical structures — pure duplication. Yet collapsing them into one
shared record would force a meaningless boundary field onto pairs and the
corpus.

**Decision** (`Schema.lean`): index one structure by level, and make the
boundary field's *type* depend on the level.
```lean
inductive EpistemicLevel | dataset | pair | corpus
abbrev EpistemicLevel.BoundaryType : EpistemicLevel → Type | .dataset => String | _ => Unit
structure EpistemicState (level : EpistemicLevel) where
  facts hypotheses intuitions unknowns : List String := []  -- the four annotation lists
  boundary : level.BoundaryType
```
`EpistemicState .dataset` has a `String` boundary (A₄ checks it non-empty);
`EpistemicState .pair` and `.corpus` have `Unit` — no boundary to state. One concept,
three levels, no duplication. The cost is that the dependent field defeats
`deriving Repr`, so a hand-written `Repr` instance is provided.

**Alternative.** A single flat structure with `boundary : Option String`. One
type, no index — but it re-admits the meaningless `(pair, some boundary)` state
and turns A₄ into a runtime check at every level instead of a type-level fact at
the dataset level. Rejected.

**Revisit when** a pair or the corpus needs its own *kind* of boundary (say a
corpus-wide frontier statement of a different type). Then extend the
`BoundaryType` match. A real reversal (back to a flat `Option`) is forced only
if the set of levels becomes open/extensible.

---

## BP₇ — How much of the schema to type

**Problem.** The JSON schema blocks are nested: an event type has a name *and* a
list of typed fields with units; an entity type has a name *and* a key (*and*
maybe a parent type). The Lean `EventSchema` / `KGSchema` / … reduce all of
this to `List String` — names only. A fully faithful mirror would recursively
type every nested object, which balloons the type surface for detail no current
proof consumes.

**Decision** (`Schema.lean`): keep name-lists.
```lean
structure EventSchema where
  entityTypes eventTypes : List String
  staticAttributes : List String := []
```
The mirror's job here is BP₁/BP₂ — the field *exists* and its shape *matches the
regime* — not field-level fidelity. That fidelity is the JSON Schema's job, and
duplicating it in Lean buys nothing until a proof actually needs a field's type.
This is a deliberate scope line, not an oversight: it is *where* the Lean types
and the JSON Schema divide the labour.

**Alternative.** Full nesting:
```lean
structure FieldSpec where name : String; type : FieldType; unit : Option String
structure EventType where name : String; fields : List FieldSpec
structure EntityType where name key : String; parent : Option String
```
Faithful, but roughly five times the type surface plus a real cost at the JSON
boundary. Held in reserve.

**Revisit when** a theorem needs to reason about a field's *type or unit* — for
example, checking that `discharge_mm` is `mm/day` in both members of a pair, or
a units-consistency lemma across a cross-instrument comparison. Trigger: a
target property you cannot even *state* because the field detail is absent from
the type. Until then, the flattening is the right economy.

---

## Summary

| | Concern | Chosen type | Revisit when |
|---|---|---|---|
| BP₁ | `schema` field | `RegimeSchema` sum type | a regime-dispatched function needs a typed schema |
| BP₂ | regime ↔ schema (A₆) | `regimeMatchesSchema` check | the redundancy causes real mismatches |
| BP₃ | pair member count | tag-dependent proof field | theorems need to case on the count |
| BP₄ | origin → blend (A₈) | dependent `payload` | `DataOriginPrimary` becomes extensible |
| BP₅ | checksum digest | `SHA256` refinement + `mkChecksum` | more algorithms, or kernel-only trust |
| BP₆ | epistemic state | level-indexed `EpistemicState` | the set of levels becomes open |
| BP₇ | schema detail | flattened to name-lists | a proof needs a field's type/unit |

The shape of the fix-pass: constraint **A₆ moved *into* the checked set** (BP₂)
and **A₈ moved *out* of it** (BP₄) — the same total content, relocated from a
runtime check to a type-level invariant where the types could carry it. That
relocation, together with the notes-block unification (BP₆) and the conversion
of boolean flags into decidable propositions, is the difference between a mirror
that merely *parses* the corpus and one that can *prove* things about it.

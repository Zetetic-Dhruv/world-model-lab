import TestSuite.Schema

/-!
# TestSuite.Trace

Lean4 mirror of the row-level schemas:
- `schemas/trace.schema.json` — event regime, becomes `traces.parquet`
- `schemas/edges.schema.json` — KG regime, becomes `edges.parquet`

The canonical encoding is Parquet; this Lean type is the row-shape
specification (useful for validators and for stating row-level properties).
-/

namespace TestSuite

/-- One row of an event-regime `traces.parquet`.
A row = one event tied to an entity, optionally timestamped, with explicit
env tagging. -/
structure TraceRow where
  /-- Globally unique row id (UUID v4 or stable hash). -/
  traceId    : String
  dataset    : String
  entityId   : String
  entityType : String
  parentId   : Option String := none
  parentType : Option String := none
  eventType  : String
  /-- ISO-8601 timestamp; `none` for atemporal regimes. -/
  t          : Option String := none
  /-- Logical ordering when `t` is absent (e.g., relaxation step). -/
  tOrder     : Option Int := none
  envId      : String
  envAxis    : String
  split      : String  -- "train" | "val" | "ood-val" | "ood-test" | "test"
  /-- JSON-serialized dataset-specific payload. -/
  payload    : String
deriving Repr

/-- One row of a KG-regime `edges.parquet`.
A row = one typed edge with explicit `sourceResource` tagging. The
`sourceResource` field is the primary distribution-shift axis for KG-regime
datasets (which source resource contributed the edge). -/
structure EdgeRow where
  edgeId         : String
  dataset        : String
  headId         : String
  headType       : String
  relation       : String
  tailId         : String
  tailType       : String
  sourceResource : String
  envId          : Option String := none
  /-- JSON-serialized payload (weight, confidence, evidence_count, …). -/
  payload        : String
deriving Repr

end TestSuite

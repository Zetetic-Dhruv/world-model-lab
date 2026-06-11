import Lake
open Lake DSL

-- TestSuite Lean4 mirror — stand-alone (no Mathlib dependency) so the schema
-- layer typechecks fast. Mathlib will be added when corpus-level theorems
-- need it.

package «testsuite» where
  version := v!"0.1.0"

@[default_target]
lean_lib «TestSuite» where
  -- The mirror of the JSON Schemas + the A1-A10 axioms.

@[default_target]
lean_lib «Examples» where
  -- Each TestSuite manifest as a Lean4 value (typecheck = schema conformance).
  globs := #[.submodules `Examples]

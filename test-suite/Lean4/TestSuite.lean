import TestSuite.Schema
import TestSuite.Manifest
import TestSuite.Pair
import TestSuite.Trace
import TestSuite.Axioms
import TestSuite.Corpus

/-!
# TestSuite — top-level module

The TestSuite corpus specification as a Lean4 library. Importing `TestSuite` brings in
all the schema mirrors, the A₁–A₁₀ constraints, and the top-level corpus type.

Layout:
- `TestSuite.Schema`      — foundational enums + `AsymmetryProfile`
- `TestSuite.Manifest`    — `DatasetManifest` mirror
- `TestSuite.Pair`        — `DatasetPair` mirror
- `TestSuite.Trace`       — `TraceRow` + `EdgeRow` mirrors
- `TestSuite.Axioms`      — A₁–A₁₀ as `Prop`s
- `TestSuite.Corpus`         — `Corpus` + corpus-level properties
-/

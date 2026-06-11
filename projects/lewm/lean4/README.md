# lewm-validity-lean

A machine-checked Lean 4 / Mathlib companion to a representation-probing validity study for
JEPA and world-model linear probing. It provides formal witnesses that three failure modes of
linear-probe evaluation — frame/episode split leakage, mutual-information estimator
non-invariance, and clustered-data effective-sample-size limits — are structural properties of
the evaluation setup, not finite-sample artifacts that vanish with more data per group.

## Module map

- **`LewmValidity.lean`** — the split-leakage gap and the one-way random-effects variance
  identity. The episode-vs-frame Bayes-risk gap equals an intrinsic explained-variance functional
  of the two σ-algebras (independent of the within-group frame count), and the grand-mean variance
  of a clustered estimator decomposes exactly as `σb²/G + σw²/(G·m)`, so the between-group floor
  `σb²/G` cannot be beaten by adding frames.
- **`EstimatorGeometry.lean`** — non-invariance of the KSG mutual-information estimator under an
  information-preserving rescaling, against the invariance of true MI, together with the
  standardization fix that restores agreement.
- **`KsgClosedForm.lean`** — the count-based KSG closed-form witness.
- **`EffectiveSampleSize.lean`** — group-concentration bounds underlying the effective-sample-size
  limit.
- **`NegativeRSquared.lean`** — the negative-R² anti-concentration result.
- **`Vendor/FLT/Chaining.lean`** — the nearest-neighbour-in-finset primitive, vendored from the
  Formal Learning Theory chaining utilities.
- **`Vendor/InformationTheory/`** — real-valued Kullback–Leibler divergence and mutual information
  built on Mathlib's `InformationTheory.klDiv`.

Mutual-information estimator asymptotics (the true `a → ∞` limit of the KSG estimator) are
documented as an open boundary rather than claimed; see the scope note in `EstimatorGeometry.lean`.

## Build

```bash
lake exe cache get      # fetch the pinned Mathlib (heavy on a cold cache)
lake build
```

The Lean toolchain is pinned in `lean-toolchain` and Mathlib is pinned in `lakefile.lean`. Every
theorem is `sorry`-free; `#print axioms` reports only `propext`, `Classical.choice`, and
`Quot.sound`.

## License

Apache-2.0, matching the file headers.

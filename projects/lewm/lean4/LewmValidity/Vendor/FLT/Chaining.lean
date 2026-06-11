/-
Copyright (c) 2026 Yuanhe Zhang. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yuanhe Zhang, Jason D. Lee, Fanghui Liu
-/
import Mathlib

/-!
# Chaining Infrastructure — vendored 1-NN primitive (minimal closure)

Vendored from the Formal Learning Theory (FLT) chaining utilities
(nearest-neighbour primitive), toolchain v4.29.0 — SAME as this companion.

Only the nearest-neighbour-in-Finset primitive is vendored here, which is the
minimal transitive closure needed by the companion: `nearestInFinset`,
`nearestInFinset_mem`, `dist_nearestInFinset_le`,
`dist_nearestInFinset_eq_zero_of_mem`. These four declarations depend only on
Mathlib (`Finset.exists_mem_eq_inf'`, `dist`, `dist_self`, `dist_nonneg`), so the
sole import is `Mathlib`. The declarations and their enclosing
`open`/`noncomputable section`/`variable` context are reproduced verbatim, so the
names live in the same (root) namespace as upstream.
-/

open Set Metric Real
open scoped BigOperators

noncomputable section

variable {A : Type*} [PseudoMetricSpace A]

/-- Choose a nearest point in a nonempty finset to a given point x.
    We use a simple choice-based definition: pick any element that minimizes distance. -/
def nearestInFinset (t : Finset A) (ht : t.Nonempty) (x : A) : A :=
  -- Just use the witness from the nonempty condition if we can't find argmin
  -- This is a simple definition that doesn't require complicated API
  Classical.choose (Finset.exists_mem_eq_inf' ht (fun y => dist x y))

/-- The nearest point in t is actually in t. -/
lemma nearestInFinset_mem (t : Finset A) (ht : t.Nonempty) (x : A) :
    nearestInFinset t ht x ∈ t := by
  unfold nearestInFinset
  exact (Classical.choose_spec (Finset.exists_mem_eq_inf' ht (fun y => dist x y))).1

/-- The distance to the nearest point is at most the distance to any point in t. -/
lemma dist_nearestInFinset_le (t : Finset A) (ht : t.Nonempty) (x : A) (y : A) (hy : y ∈ t) :
    dist x (nearestInFinset t ht x) ≤ dist x y := by
  unfold nearestInFinset
  have hspec := Classical.choose_spec (Finset.exists_mem_eq_inf' ht (fun y => dist x y))
  rw [← hspec.2]
  exact Finset.inf'_le _ hy

/-- If x is in the finset, the distance to the nearest point is 0. -/
lemma dist_nearestInFinset_eq_zero_of_mem (t : Finset A) (ht : t.Nonempty) (x : A) (hx : x ∈ t) :
    dist x (nearestInFinset t ht x) = 0 :=
  le_antisymm (by simpa only [dist_self] using dist_nearestInFinset_le t ht x x hx) dist_nonneg

end

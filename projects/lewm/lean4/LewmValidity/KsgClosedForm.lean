/-
Copyright (c) 2026 Dhruv Gupta. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Dhruv Gupta
-/
import LewmValidity.EstimatorGeometry

/-!
# A concrete witness that the count-based k = 1 KSG estimate is *not* reparametrisation
invariant under anisotropic rescaling, while the true mutual information is

The companion module `LewmValidity.EstimatorGeometry` already establishes:

* **truth side** `InformationTheory.mutualInformationReal_map_prodCongr`: the *true*
  mutual information is invariant under any coordinatewise measurable reparametrisation
  `e₁.prodCongr e₂` of `ℝ × ℝ`; in particular under the anisotropic diagonal scaling
  `diagEquiv c₁ c₂ : (x, y) ↦ (c₁ x, c₂ y)`.
* **estimator radius core** `ksgRadiusStat_map_smul` / the *standardised*
  invariance theorems: the **geometric radius core** `∑ᵢ εᵢ` scales by the similarity ratio
  under a *uniform* scaling, and is *fixed* after per-axis standardisation. That block
  explicitly leaves the **full count-cancelling KSG number under an anisotropic
  reparametrisation** to the empirical study (`ksgRadiusStat`'s scope note: "only an
  *anisotropic* reparametrisation moves it, and that remains an empirical phenomenon rather
  than a settled theorem").

This module closes exactly that gap with a **concrete finite witness on integer
coordinates**. It formalises the *complete* k = 1 KSG mutual-information estimate in its
**harmonic-number form** — the count term included, not just the radius core — and proves it
**strictly changes** under the anisotropic rescaling `(x, y) ↦ (4 x, y)`, on a four-point
sample, while the *same* reparametrisation leaves the true MI of every absolutely-continuous
law fixed (instantiating the truth-side invariance theorem).

## The harmonic-number form (why no digamma / no Euler γ is needed)

The k = 1 KSG estimate is classically

    Î = ψ(1) + ψ(N) − (1/N) Σᵢ [ψ(n_x(i)+1) + ψ(n_y(i)+1)],

with `n_x(i)`, `n_y(i)` the marginal neighbour counts within the joint 1-NN radius `εᵢ`. At
*integer* arguments the digamma constants cancel: `ψ(n+1) = −γ + H_n` with the harmonic
number `H_n = Σ_{k=1}^n 1/k`, so

    Î = H_{N−1} − (1/N) Σᵢ [H_{n_x(i)} + H_{n_y(i)}]                                  (★)

is a purely **rational** functional of the neighbour counts — no transcendental digamma, no
Euler–Mascheroni constant. We take (★) as the definition (`ksgEstimate`), with `H_n` the
Mathlib-style rational harmonic number `harmonicQ n = ∑_{k<n} 1/(k+1)`.

## Why an *integer-coordinate* sample (the key tractability move)

The joint 1-NN radius `εᵢ = min_{j≠i} dist(pᵢ, pⱼ)` and the marginal counts
`n_x(i) = #{j≠i : |xᵢ−xⱼ| ≤ εᵢ}` are, on a sample with **integer** coordinates under the
**sup metric** `dist (a,b) (c,d) = max |a−c| |b−d|`, entirely **decidable**: every distance is
an integer, every comparison is `Decidable`, and `decide` evaluates each `εᵢ` (a `Finset.min'`
of an integer image) and each count (a `Finset.filter` cardinality) by kernel reduction —
*without* any real-number nearest-neighbour reasoning. The rational harmonic arithmetic, which
`decide` does **not** reduce (the `Rat` `num`/`den` normalisation stalls in the kernel), is
discharged separately by `norm_num` after the counts have been decided. This split is what
makes the witness a clean, axiom-minimal, *kernel-verified* fact rather than a hand NN
argument over `ℝ` that fails to close.

Note we use the joint 1-NN **radius** `εᵢ` (the minimal sup-distance), not a *chosen* nearest
neighbour: the count-based estimate depends only on `εᵢ` as a number, so the argmin tie-breaking
that `nearestInFinset` resolves by `Classical.choose` is irrelevant here and is avoided.

## The witness configuration and the flip

The four-point sample is the thin horizontal cluster with one vertical outlier

    S = { (0,0), (1,0), (4,0), (0,3) } ⊆ ℤ²,

and the reparametrisation is the **anisotropic** stretch `x ↦ 4 x`, `y ↦ y`
(`scaleConfig 4 1`, the integer shadow of `diagEmb (4 : ℝ) 1`). At the corner point `(0,0)`
the joint 1-NN is governed by the *x*-neighbour `(1,0)` at radius `ε = 1` before the stretch,
but after the stretch the *x*-neighbour recedes to `(4,0)` and the joint 1-NN flips to the
*y*-outlier `(0,3)` at radius `ε = 3`. This flip moves the marginal counts and hence the
estimate:

    Î(S) = −29/24  ⟶  Î(scaleConfig 4 1 S) = −9/8,        −29/24 ≠ −9/8.

So the *complete* count-based KSG estimate — count term included — is moved by an anisotropic
coordinate change that the *true* mutual information cannot see.

## Main statements

* `ksgEstimate`: the complete k = 1 KSG MI estimate (★) on a four-point integer sample, in
  harmonic-number form.
* `ksgEstimate_witness` / `ksgEstimate_witness_scaled`: the two evaluated values `−29/24` and
  `−9/8`.
* `ksgEstimate_not_anisotropy_invariant`: the strict inequality
  `ksgEstimate (scaleConfig 4 1 witnessZ) ≠ ksgEstimate witnessZ`.
* `ksg_count_estimate_not_information_invariant`: the combined witness — the anisotropic
  reparametrisation `diagEquiv (4:ℝ) 1` leaves the true MI of every absolutely-continuous
  joint law fixed (by the truth-side invariance theorem), while the count-based KSG estimate
  strictly drifts on the witness sample.

## References

* Kraskov, Stögbauer & Grassberger, *Estimating mutual information*, Phys. Rev. E 69 (2004)
  066138, **erratum Phys. Rev. E 83 (2011) 019903** — the KSG estimator; Algorithm 1 and §II
  for the marginal-count corrections `ψ(n_x+1)`, `ψ(n_y+1)` within the joint 1-NN radius, and
  the observation that under a *uniform* rescaling these corrections cancel (so anisotropy is
  required to move the estimate, exactly the case this witness exhibits).
* Kozachenko & Leonenko, *Sample estimate of the entropy of a random vector*, Probl. Inf.
  Transm. 23 (1987) 95–101 — the `⟨log ε⟩` k-NN entropy estimator whose harmonic-number form
  (via `ψ(n+1) = −γ + H_n`) underlies (★).
* McAllester & Stratos, *Formal Limitations on the Measurement of Mutual Information*, AISTATS
  2020 (PMLR 108:875–884); Song & Ermon, *Understanding the Limitations of Variational Mutual
  Information Estimators*, ICLR 2020 — the estimator/truth divide this witness instantiates:
  estimators violate the reparametrisation invariance the true MI obeys (cited as **motivation
  only**; the concrete witness and its conclusion are original to this formalisation).
* Czyż, Grabowski, Vogt, Beerenwinkel & Marx, *Beyond Normal: On the Evaluation of Mutual
  Information Estimators*, NeurIPS 36 (2023), Theorem 2 — invariance of the *true* MI under
  injective per-coordinate reparametrisation (the population fact the truth-side invariance
  theorem supplies here).
-/

noncomputable section

open Finset MeasureTheory

namespace LewmValidity.KsgClosedForm

/-! ### Decidable sup-metric geometry on an integer sample -/

/-- The **sup (Chebyshev) distance** on `ℤ²`, `supdistZ (a₁,a₂) (b₁,b₂) = max |a₁−b₁| |a₂−b₂|`.
This is the integer shadow of Mathlib's product metric on `ℝ × ℝ`
(`dist (a,b) (c,d) = max (dist a c) (dist b d)`) restricted to integer points; because it is
`ℤ`-valued, every distance comparison is `Decidable`. -/
def supdistZ (a b : ℤ × ℤ) : ℤ := max (|a.1 - b.1|) (|a.2 - b.2|)

/-- A four-point integer sample is carried as `Fin 4 → ℤ × ℤ`. The leave-one-out index set of
point `i` is `Finset.univ.erase i`, which is nonempty (it has `4 − 1 = 3` elements); this is the
nonemptiness witness fed to `Finset.min'` when forming the joint 1-NN radius. -/
lemma eraseUniv_image_nonempty (Q : Fin 4 → ℤ × ℤ) (i : Fin 4) :
    ((univ.erase i).image (fun j => supdistZ (Q i) (Q j))).Nonempty := by
  apply Nonempty.image
  rw [← card_pos, card_erase_of_mem (mem_univ _)]
  decide

/-- **The joint 1-NN radius** `εᵢ = min_{j≠i} supdistZ (pᵢ, pⱼ)` of point `i` in the sample,
as a `Finset.min'` of the integer image of leave-one-out sup-distances. This is the radius of
the smallest sup-ball around `pᵢ` containing another sample point — the `εᵢ` of the KSG
construction. We use the radius (a number) rather than a *chosen* nearest neighbour, so no
argmin tie-breaking (and no `Classical.choose`) enters. -/
def jointRadius (Q : Fin 4 → ℤ × ℤ) (i : Fin 4) : ℤ :=
  ((univ.erase i).image (fun j => supdistZ (Q i) (Q j))).min'
    (eraseUniv_image_nonempty Q i)

/-- **Marginal X-neighbour count** `n_x(i) = #{j ≠ i : |xᵢ − xⱼ| ≤ εᵢ}` — the number of sample
points (other than `i`) whose first coordinate lies within the joint 1-NN radius `εᵢ` of point
`i`. This is the `n_x(i)` of KSG Algorithm 1 (closed `≤` convention: the joint nearest
neighbour, whose sup-distance is exactly `εᵢ`, is counted, so `n_x(i) ≥ 1` and `H_{n_x(i)}` is
well defined). -/
def marginalCountX (Q : Fin 4 → ℤ × ℤ) (i : Fin 4) : ℕ :=
  ((univ.erase i).filter (fun j => |(Q i).1 - (Q j).1| ≤ jointRadius Q i)).card

/-- **Marginal Y-neighbour count** `n_y(i) = #{j ≠ i : |yᵢ − yⱼ| ≤ εᵢ}` (the second-coordinate
analogue of `marginalCountX`). -/
def marginalCountY (Q : Fin 4 → ℤ × ℤ) (i : Fin 4) : ℕ :=
  ((univ.erase i).filter (fun j => |(Q i).2 - (Q j).2| ≤ jointRadius Q i)).card

/-! ### The rational harmonic number and the count-based KSG estimate -/

/-- The **rational harmonic number** `H_n = Σ_{k=1}^n 1/k = Σ_{k<n} 1/(k+1)`, as an element of
`ℚ`. This is the value of `ψ(n+1) + γ` at integer `n` (`ψ(n+1) = −γ + H_n`), so using it in
place of the digamma corrections is exactly the cancellation that removes the Euler–Mascheroni
constant from the KSG estimate. `H_0 = 0`, `H_1 = 1`, `H_2 = 3/2`, `H_3 = 11/6`. -/
def harmonicQ (n : ℕ) : ℚ := ∑ k ∈ range n, (1 : ℚ) / (k + 1)

/-- **The complete k = 1 KSG mutual-information estimate**, in harmonic-number form, on a
four-point integer sample:

    ksgEstimate Q = H_{N−1} − (1/N) Σᵢ [ H_{n_x(i)} + H_{n_y(i)} ],     N = 4,

with `H = harmonicQ`, `n_x = marginalCountX`, `n_y = marginalCountY`, and the joint 1-NN
radius `εᵢ = jointRadius Q i` entering through the counts. Unlike the radius core
`LewmValidity.EstimatorGeometry.ksgRadiusStat` (which is only the `Σ log ε` term), this is the *complete*
count-cancelling KSG number — the marginal-count corrections are included. The witness below is
that this complete estimate is **not** invariant under an anisotropic coordinate rescaling.

## References
Kraskov, Stögbauer & Grassberger, Phys. Rev. E 69 (2004) 066138 (erratum 83 (2011) 019903),
Algorithm 1 (`Î = ψ(k) + ψ(N) − ⟨ψ(n_x+1) + ψ(n_y+1)⟩` at `k = 1`); Kozachenko & Leonenko,
Probl. Inf. Transm. 23 (1987) 95–101 (the harmonic-number form via `ψ(n+1) = −γ + H_n`). -/
def ksgEstimate (Q : Fin 4 → ℤ × ℤ) : ℚ :=
  harmonicQ 3 - (1 / 4) * ∑ i : Fin 4, (harmonicQ (marginalCountX Q i) + harmonicQ (marginalCountY Q i))

/-! ### The anisotropic rescaling on integer samples -/

/-- The **anisotropic diagonal rescaling** of an integer sample, `(x, y) ↦ (c₁ x, c₂ y)`
applied pointwise. This is the integer shadow of `LewmValidity.EstimatorGeometry.diagEmb (c₁:ℝ) (c₂:ℝ)`
(and, for `c₁ = c₂`, of `scaleEmb`): for any sample `Q`, the real embedding of
`scaleConfig c₁ c₂ Q` equals `diagEmb (c₁:ℝ) (c₂:ℝ)` of the real embedding of `Q`
(`embedR_scaleConfig`). The witness uses `scaleConfig 4 1` — stretch `x` by `4`, leave `y` —
the anisotropic case the radius-core scope note leaves to the empirical study. -/
def scaleConfig (c₁ c₂ : ℤ) (Q : Fin 4 → ℤ × ℤ) : Fin 4 → ℤ × ℤ :=
  fun i => (c₁ * (Q i).1, c₂ * (Q i).2)

/-! ### The witness sample and its scaled image -/

/-- **The witness sample** `S = { (0,0), (1,0), (4,0), (0,3) } ⊆ ℤ²` — a thin horizontal
cluster `(0,0),(1,0),(4,0)` on the x-axis together with a single vertical outlier `(0,3)`.
At the corner `(0,0)` the joint 1-NN is the x-neighbour `(1,0)` (radius `1`); under the
anisotropic stretch `x ↦ 4 x` this neighbour recedes and the joint 1-NN flips to the
y-outlier `(0,3)` (radius `3`), which is what moves the marginal counts and the estimate. -/
def witnessZ : Fin 4 → ℤ × ℤ
  | 0 => (0, 0)
  | 1 => (1, 0)
  | 2 => (4, 0)
  | 3 => (0, 3)

/-! ### Evaluating the estimate before and after the anisotropic rescaling

The pattern in both proofs: expand the `Fin 4` sum (`Fin.sum_univ_four`), replace each
marginal count by its value — each a `decide` over the decidable integer geometry (the joint
1-NN radius `min'` and the count `filter` cardinality both reduce in the kernel) — and finish
the rational harmonic arithmetic with `norm_num [harmonicQ, Finset.sum_range_succ]` (which the
kernel's `Rat` normalisation cannot `decide`). -/

/-- **The estimate on the witness sample is `−29/24`.** The decided per-point neighbour counts
are, at the joint 1-NN radii `ε = (1, 1, 3, 3)` for points `(0,0),(1,0),(4,0),(0,3)`:
`(n_x, n_y) = (2,2), (2,2), (1,3), (2,3)`, giving contributions `H₂+H₂, H₂+H₂, H₁+H₃, H₂+H₃`. -/
theorem ksgEstimate_witness : ksgEstimate witnessZ = -29 / 24 := by
  rw [ksgEstimate, Fin.sum_univ_four]
  have h0x : marginalCountX witnessZ 0 = 2 := by decide
  have h0y : marginalCountY witnessZ 0 = 2 := by decide
  have h1x : marginalCountX witnessZ 1 = 2 := by decide
  have h1y : marginalCountY witnessZ 1 = 2 := by decide
  have h2x : marginalCountX witnessZ 2 = 1 := by decide
  have h2y : marginalCountY witnessZ 2 = 3 := by decide
  have h3x : marginalCountX witnessZ 3 = 2 := by decide
  have h3y : marginalCountY witnessZ 3 = 3 := by decide
  rw [h0x, h0y, h1x, h1y, h2x, h2y, h3x, h3y]
  norm_num [harmonicQ, Finset.sum_range_succ]

/-- **The estimate on the anisotropically scaled sample is `−9/8`.** After `x ↦ 4 x` the sample
is `(0,0),(4,0),(16,0),(0,3)`; the joint 1-NN radii become `(3, 4, 12, 3)` (the corner `(0,0)`
has flipped to the y-outlier at radius `3`), and the decided counts are
`(n_x, n_y) = (1,3), (2,3), (1,3), (1,3)`, contributions `H₁+H₃, H₂+H₃, H₁+H₃, H₁+H₃`. -/
theorem ksgEstimate_witness_scaled : ksgEstimate (scaleConfig 4 1 witnessZ) = -9 / 8 := by
  rw [ksgEstimate, Fin.sum_univ_four]
  have h0x : marginalCountX (scaleConfig 4 1 witnessZ) 0 = 1 := by decide
  have h0y : marginalCountY (scaleConfig 4 1 witnessZ) 0 = 3 := by decide
  have h1x : marginalCountX (scaleConfig 4 1 witnessZ) 1 = 2 := by decide
  have h1y : marginalCountY (scaleConfig 4 1 witnessZ) 1 = 3 := by decide
  have h2x : marginalCountX (scaleConfig 4 1 witnessZ) 2 = 1 := by decide
  have h2y : marginalCountY (scaleConfig 4 1 witnessZ) 2 = 3 := by decide
  have h3x : marginalCountX (scaleConfig 4 1 witnessZ) 3 = 1 := by decide
  have h3y : marginalCountY (scaleConfig 4 1 witnessZ) 3 = 3 := by decide
  rw [h0x, h0y, h1x, h1y, h2x, h2y, h3x, h3y]
  norm_num [harmonicQ, Finset.sum_range_succ]

/-- **The count-based KSG estimate is not invariant under anisotropic rescaling.** The
anisotropic stretch `x ↦ 4 x`, `y ↦ y` (`scaleConfig 4 1`) strictly changes the *complete*
count-cancelling k = 1 KSG estimate on the witness sample, `−29/24 ⟶ −9/8`. This is the
count-term phenomenon the radius-core scope note
(`LewmValidity.EstimatorGeometry.ksgRadiusStat`) leaves to the empirical study, here exhibited
as an exact rational inequality. -/
theorem ksgEstimate_not_anisotropy_invariant :
    ksgEstimate (scaleConfig 4 1 witnessZ) ≠ ksgEstimate witnessZ := by
  rw [ksgEstimate_witness, ksgEstimate_witness_scaled]; norm_num

/-! ### Binding to the truth side: the scaling is the integer shadow of `diagEmb` -/

/-- The real embedding of the anisotropically scaled integer sample equals
`LewmValidity.EstimatorGeometry.diagEmb (c₁:ℝ) (c₂:ℝ)` applied to the real embedding of the sample: for
every index `i`, `(c₁ • xᵢ, c₂ • yᵢ)` over `ℝ` is the embedding of `scaleConfig c₁ c₂`'s
`i`-th point. This certifies that `scaleConfig` is the *same* anisotropic reparametrisation on
the sample that `diagEquiv` is on the ambient `ℝ × ℝ`, so the truth-side invariance theorem
(phrased for `diagEquiv`) and the estimator-side drift (phrased for `scaleConfig`) concern one
and the same coordinate change. -/
lemma embedR_scaleConfig (c₁ c₂ : ℤ) (Q : Fin 4 → ℤ × ℤ) (i : Fin 4) :
    (((c₁ : ℝ) * ((Q i).1 : ℝ)), ((c₂ : ℝ) * ((Q i).2 : ℝ)))
      = (((scaleConfig c₁ c₂ Q i).1 : ℝ), ((scaleConfig c₁ c₂ Q i).2 : ℝ)) := by
  simp [scaleConfig]

/-! ### Truth fixed, the count-based estimate drifts -/

/-- **The complete count-based k = 1 KSG estimate is not a function of the information content,
under anisotropic rescaling.**

The single anisotropic reparametrisation `diagEquiv (4:ℝ) 1 : (x, y) ↦ (4 x, y)` — a
`MeasurableEquiv` of the form `e₁.prodCongr e₂`, hence information-preserving — does two
incompatible things at once:

* **truth is fixed.** By the truth-side invariance theorem
  (`InformationTheory.mutualInformationReal_map_prodCongr`) the *true* mutual information of
  **every** absolutely-continuous joint law on `ℝ × ℝ` is left unchanged by `diagEquiv (4:ℝ) 1`.
* **the count-based estimate drifts.** By `ksgEstimate_not_anisotropy_invariant` the *complete*
  count-cancelling k = 1 KSG estimate (harmonic-number form, marginal-count corrections
  included) strictly changes on the witness sample under the integer shadow `scaleConfig 4 1`
  of that same map, `−29/24 ⟶ −9/8` (the two being one coordinate change by
  `embedR_scaleConfig`).

So no function of the true mutual information can agree with the KSG count estimate: the
estimate is sensitive to an *anisotropic* coordinate change the information itself cannot see.
This is the strongest form of the estimator/truth divide — it goes beyond the radius core
(`ksgRadiusStat`) and the uniform-scaling case to the full count term under anisotropy, the
case the radius-core scope note recorded as empirical. Here it is machine-checked.

## References (motivation)
The combined truth-fixed / count-estimate-drifts witness under anisotropy is **original to this
formalisation**. Cited as **motivation only**: McAllester & Stratos, *Formal Limitations on the
Measurement of Mutual Information*, AISTATS 2020 (PMLR 108:875–884); Song & Ermon,
*Understanding the Limitations of Variational Mutual Information Estimators*, ICLR 2020 (the
estimator-side analogue — estimators violate the invariance the true MI obeys); Czyż,
Grabowski, Vogt, Beerenwinkel & Marx, *Beyond Normal*, NeurIPS 36 (2023), Theorem 2 (MI
invariant under per-coordinate reparametrisation while estimates are not). The estimator
mechanics rest on Kraskov, Stögbauer & Grassberger, Phys. Rev. E 69 (2004) 066138 (erratum
83 (2011) 019903) and Kozachenko & Leonenko, Probl. Inf. Transm. 23 (1987) 95–101; the true-MI
invariance instantiates the truth-side invariance theorem. -/
theorem ksg_count_estimate_not_information_invariant :
    (∀ (P : Measure (ℝ × ℝ)) [IsProbabilityMeasure P],
        P ≪ (P.map Prod.fst).prod (P.map Prod.snd) →
        InformationTheory.mutualInformationReal
            (P.map (LewmValidity.EstimatorGeometry.diagEquiv (by norm_num : (4 : ℝ) ≠ 0) (by norm_num : (1 : ℝ) ≠ 0)))
          = InformationTheory.mutualInformationReal P)
      ∧ ksgEstimate (scaleConfig 4 1 witnessZ) ≠ ksgEstimate witnessZ := by
  refine ⟨fun P _ hP => ?_, ksgEstimate_not_anisotropy_invariant⟩
  exact InformationTheory.mutualInformationReal_map_prodCongr
    ((Homeomorph.mulLeft₀ (4 : ℝ) (by norm_num)).toMeasurableEquiv)
    ((Homeomorph.mulLeft₀ (1 : ℝ) (by norm_num)).toMeasurableEquiv) P hP

end LewmValidity.KsgClosedForm

end

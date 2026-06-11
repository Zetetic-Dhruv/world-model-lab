/-
Copyright (c) 2026 Dhruv Gupta. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Dhruv Gupta
-/
/-
  LewmValidity.EffectiveSampleSize — group-level (sub-Gaussian / Hoeffding) concentration
  for the one-way random-effects grand mean, at the *tail* level.

  ## What this adds over `LewmValidity.variance_grandMean_eq`

  `variance_grandMean_eq` pins the *variance* of the grand mean to `σb²/G + σw²/(G·m)`,
  exact. That is the strongest *second-moment* statement and is NOT weakened here. This
  module supplies the complementary *exponential tail* statement: under the extra (and
  honest) structural hypothesis that the between-group effects are almost surely bounded
  in a common interval of width `R`, the irreducible between-group component

      b̄ := G⁻¹ • ∑_{i ∈ Sb} f i          (the only part of the grand mean frames cannot touch)

  concentrates around its mean `𝔼[b̄]` at a Hoeffding rate governed by the *group count*
  `G`, not the frame count `m`:

      μ.real {ω | t ≤ b̄ ω − 𝔼[b̄]}        ≤ exp (−2 · G · t² / R²)        (one-sided)
      μ.real {ω | t ≤ |b̄ ω − 𝔼[b̄]|}      ≤ 2 · exp (−2 · G · t² / R²)     (two-sided)

  The exponent's only sample-size parameter is `G`. Adding frames `m` enlarges `Se` and
  shrinks the *within*-group variance term `σw²/(G·m)`, but `b̄` does not depend on `Se`
  at all, so no number of frames sharpens this tail past the group-count floor. This is
  the tail-level reading of "effective sample size = G".

  ## Method (native sub-Gaussian / Hoeffding stack)

  Mathlib carries the full sub-Gaussian concentration machinery:
    • `ProbabilityTheory.HasSubgaussianMGF` and the Chernoff bound `.measure_ge_le`;
    • `hasSubgaussianMGF_of_mem_Icc` — Hoeffding's lemma: a centered random variable a.s.
      in `Icc a b` has a sub-Gaussian MGF with parameter `((b−a)/2)²`;
    • `measure_sum_ge_le_of_iIndepFun` — Hoeffding's inequality for sums of independent
      sub-Gaussian variables: `μ.real {ε ≤ ∑ Xᵢ} ≤ exp(−ε²/(2 ∑ cᵢ))`.
  We instantiate the sum bound on the centered, `G⁻¹`-scaled family
  `Xᵢ ω := G⁻¹ · (f i ω − 𝔼[f i])`, `i ∈ Sb`. Each `Xᵢ` is sub-Gaussian with parameter
  `(G⁻¹)² · (R/2)²` (Hoeffding's lemma + `HasSubgaussianMGF.const_mul`), the family is
  independent (`iIndepFun.comp` of the model's single `iIndepFun f`), and
  `∑_{i∈Sb} Xᵢ = b̄ − 𝔼[b̄]` while `∑_{i∈Sb} cᵢ = G · (G⁻¹)²(R/2)² = R²/(4G)`. Hoeffding's
  sum bound then reads `exp(−t² / (2 · R²/(4G))) = exp(−2·G·t²/R²)`.

  Mathlib's own `Mathlib.Probability.Moments.SubGaussian` supplies everything used here, so
  no external sub-Gaussian library is required.

  References. Hoeffding, *Probability inequalities for sums of bounded random variables*,
  JASA 58(301) (1963), 13–30 (Hoeffding's lemma + the sum inequality used here). Boucheron,
  Lugosi & Massart, *Concentration Inequalities* (OUP, 2013), §2.3–2.6 (sub-Gaussian sums).
  One-way random-effects ANOVA: Searle, Casella & McCulloch, *Variance Components*
  (1992), §3. Effective-sample-size / design-effect tie: the bound's rate is governed by
  the group count `G`, not the frame count `m` — the design-effect / effective-`n` reading
  of the `σb²/G` floor; see Moulton, *Random group effects and the precision of regression
  estimates*, J. Econometrics 32(3) (1986), 385–397, and Kish, *Survey Sampling* (1965),
  §5.4, §8.2. (The full design-effect treatment lives in the companion `LewmValidity.lean`
  Check-4 block.)
-/
import LewmValidity
import Mathlib

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

namespace LewmValidity

namespace OneWayRandomEffects

variable {Ω ι : Type*} [MeasurableSpace Ω] (M : OneWayRandomEffects Ω ι)

/-! ## The between-group component `b̄` and its boundedness hypothesis -/

/-- The irreducible between-group component of the grand mean,
    `b̄ := G⁻¹ • ∑_{i ∈ Sb} f i`. This is the summand of `grandMean` indexed by the
    `G` group effects `Sb`; it is the part of the estimator that the within-group block
    `Se` (hence the frame count `m`) does not enter. Its variance is exactly `σb²/G`
    (the floor of `variance_grandMean_eq`), and this module bounds its exponential tail. -/
noncomputable def betweenMean : Ω → ℝ :=
  (M.G : ℝ)⁻¹ • ∑ i ∈ M.Sb, M.f i

/-- Boundedness witness for the between-group effects (added structure): every group
    effect `f i`, `i ∈ Sb`, is almost surely confined to a common interval `[a, a + R]`
    of width `R`. This is the only hypothesis Hoeffding's lemma needs beyond the model's
    independence; it is genuine extra structure (the bare `OneWayRandomEffects` posits
    only second moments), and it is what upgrades the variance floor to an exponential
    tail. The lower endpoint `a` is immaterial to the rate — only the width `R` enters. -/
structure BetweenBounded (a R : ℝ) : Prop where
  /-- Each between-group effect lies a.s. in the width-`R` interval `[a, a + R]`. -/
  mem_Icc : ∀ i ∈ M.Sb, ∀ᵐ ω ∂M.μ, M.f i ω ∈ Set.Icc a (a + R)

variable {M}

/-- Algebraic core: the centered between-group component is the sum over `Sb` of the
    centered, `G⁻¹`-scaled per-group effects. With `Xᵢ ω := G⁻¹ · (f i ω − 𝔼[f i])`,

        b̄ ω − 𝔼[b̄]  =  ∑_{i ∈ Sb} Xᵢ ω.

    Both sides are `G⁻¹ ∑ (f i ω − 𝔼[f i])`; expectation passes through the finite sum
    and the constant `G⁻¹` by linearity of the integral (`MemLp.integrable` supplies the
    integrability of each `f i`). -/
private lemma betweenMean_sub_integral_eq (ω : Ω) :
    M.betweenMean ω - ∫ x, M.betweenMean x ∂M.μ
      = ∑ i ∈ M.Sb, ((M.G : ℝ)⁻¹ * (M.f i ω - ∫ x, M.f i x ∂M.μ)) := by
  have hint : ∀ i ∈ M.Sb, Integrable (M.f i) M.μ := fun i _ => (M.hL2 i).integrable one_le_two
  have hbω' : ∀ x, M.betweenMean x = (M.G : ℝ)⁻¹ * ∑ i ∈ M.Sb, M.f i x := by
    intro x
    simp only [betweenMean, Pi.smul_apply, Finset.sum_apply, smul_eq_mul]
  have hEb : ∫ x, M.betweenMean x ∂M.μ
      = (M.G : ℝ)⁻¹ * ∑ i ∈ M.Sb, ∫ x, M.f i x ∂M.μ := by
    simp only [hbω']
    rw [integral_const_mul, integral_finset_sum M.Sb hint]
  rw [hbω' ω, hEb, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  ring

/-- Independence of the centered, scaled per-group family. Each `Xᵢ := G⁻¹ · (f i − 𝔼[f i])`
    is a fixed measurable function of `f i` alone, so the model's single mutual-independence
    hypothesis `iIndepFun f` pushes through `iIndepFun.comp`. -/
private lemma iIndepFun_centeredScaled :
    iIndepFun (fun i => fun ω => (M.G : ℝ)⁻¹ * (M.f i ω - ∫ x, M.f i x ∂M.μ)) M.μ := by
  have h := M.hindep.comp (fun i => fun x : ℝ => (M.G : ℝ)⁻¹ * (x - ∫ y, M.f i y ∂M.μ))
    (fun i => by fun_prop)
  exact h

/-- Each centered, scaled per-group effect `Xᵢ := G⁻¹ · (f i − 𝔼[f i])` is sub-Gaussian
    with parameter `(G⁻¹)² · (R/2)²`. Hoeffding's lemma (`hasSubgaussianMGF_of_mem_Icc`)
    makes the centered `f i − 𝔼[f i]` sub-Gaussian with parameter `((b−a)/2)² = (R/2)²`
    from the boundedness witness, and `HasSubgaussianMGF.const_mul` scales the parameter
    by `(G⁻¹)²`. -/
private lemma hasSubgaussianMGF_centeredScaled {a R : ℝ}
    (hb : M.BetweenBounded a R) (i : ι) (hi : i ∈ M.Sb) :
    HasSubgaussianMGF (fun ω => (M.G : ℝ)⁻¹ * (M.f i ω - ∫ x, M.f i x ∂M.μ))
      (⟨(M.G : ℝ)⁻¹ ^ 2, sq_nonneg _⟩ * (‖(a + R) - a‖₊ / 2) ^ 2) M.μ := by
  have hmeas : AEMeasurable (M.f i) M.μ := (M.hL2 i).aestronglyMeasurable.aemeasurable
  have hcenter : HasSubgaussianMGF (fun ω => M.f i ω - ∫ x, M.f i x ∂M.μ)
      ((‖(a + R) - a‖₊ / 2) ^ 2) M.μ := by
    have := hasSubgaussianMGF_of_mem_Icc (X := M.f i) (μ := M.μ) (a := a) (b := a + R)
      hmeas (hb.mem_Icc i hi)
    simpa using this
  have := hcenter.const_mul ((M.G : ℝ)⁻¹)
  exact this

/-! ## Group-level Hoeffding concentration -/

/-- **Group-level concentration (one-sided Hoeffding).** Under the boundedness witness
    `BetweenBounded a R` (group effects a.s. in a width-`R` interval), the irreducible
    between-group component `b̄ = G⁻¹ • ∑_{Sb} f` concentrates above its mean at a rate
    governed by the *group count* `G`:

        μ.real {ω | t ≤ b̄ ω − 𝔼[b̄]}  ≤  exp (−2 · G · t² / R²),   for `0 ≤ t`.

    The only sample-size parameter in the exponent is `G`. Because `b̄` does not depend on
    the within-group block `Se`, increasing the frame count `m` cannot sharpen this tail:
    the effective sample size is the number of groups, exactly as the variance floor
    `variance_grandMean_ge_between` asserts at the second-moment level.

    The bound is stated for `Measure.real` (the `ℝ`-valued measure), matching Mathlib's
    `measure_sum_ge_le_of_iIndepFun`; `betweenMean_measure_ge_le` below restates it for the
    `ℝ≥0∞`-valued measure via `ofReal`. -/
theorem betweenMean_measureReal_ge_le {a R : ℝ} (hb : M.BetweenBounded a R)
    (hG : 0 < M.G) (hR : 0 < R) {t : ℝ} (ht : 0 ≤ t) :
    M.μ.real {ω | t ≤ M.betweenMean ω - ∫ x, M.betweenMean x ∂M.μ}
      ≤ Real.exp (-2 * M.G * t ^ 2 / R ^ 2) := by
  classical
  -- Centered, scaled summand family and its per-term sub-Gaussian parameter.
  set X : ι → Ω → ℝ := fun i => fun ω => (M.G : ℝ)⁻¹ * (M.f i ω - ∫ x, M.f i x ∂M.μ) with hX
  set c : ι → ℝ≥0 := fun _ => ⟨(M.G : ℝ)⁻¹ ^ 2, sq_nonneg _⟩ * (‖(a + R) - a‖₊ / 2) ^ 2 with hc
  -- Hoeffding for the independent sub-Gaussian sum.
  have hsum := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun (μ := M.μ) (X := X) (c := c)
    iIndepFun_centeredScaled (s := M.Sb)
    (fun i hi => hasSubgaussianMGF_centeredScaled hb i hi) ht
  -- Rewrite the event: ∑ Xᵢ = b̄ − 𝔼[b̄].
  have hevent : {ω | t ≤ ∑ i ∈ M.Sb, X i ω}
      = {ω | t ≤ M.betweenMean ω - ∫ x, M.betweenMean x ∂M.μ} := by
    ext ω
    simp only [Set.mem_setOf_eq, hX]
    rw [← betweenMean_sub_integral_eq ω]
  rw [hevent] at hsum
  refine hsum.trans (le_of_eq ?_)
  -- Identify the Hoeffding exponent with −2·G·t²/R².
  have hGr : (M.G : ℝ) ≠ 0 := by exact_mod_cast hG.ne'
  -- ∑_{Sb} c = G · (G⁻¹)²(R/2)², as a real number.
  have hnormR : ‖(a + R) - a‖ = R := by
    rw [add_sub_cancel_left, Real.norm_of_nonneg hR.le]
  have hcoe : ((∑ _i ∈ M.Sb, c _i : ℝ≥0) : ℝ)
      = (M.G : ℝ) * ((M.G : ℝ)⁻¹ ^ 2 * (R / 2) ^ 2) := by
    rw [Finset.sum_const, M.hcardSb, nsmul_eq_mul]
    push_cast
    rw [hnormR]
  rw [Real.exp_eq_exp, hcoe]
  have hRr : R ^ 2 ≠ 0 := pow_ne_zero 2 hR.ne'
  field_simp

/-- **Group-level concentration (lower tail).** The symmetric lower-tail companion of
    `betweenMean_measureReal_ge_le`: the between-group component falls below its mean by at
    least `t` with the same group-count-governed probability,

        μ.real {ω | t ≤ 𝔼[b̄] − b̄ ω}  ≤  exp (−2 · G · t² / R²),   for `0 ≤ t`.

    Proof by reflecting the one-sided bound through `−`: each centered, scaled per-group
    effect's negation is sub-Gaussian with the same parameter (`HasSubgaussianMGF.neg`), and
    `∑ (−Xᵢ) = 𝔼[b̄] − b̄`. -/
theorem betweenMean_measureReal_le_le {a R : ℝ} (hb : M.BetweenBounded a R)
    (hG : 0 < M.G) (hR : 0 < R) {t : ℝ} (ht : 0 ≤ t) :
    M.μ.real {ω | t ≤ (∫ x, M.betweenMean x ∂M.μ) - M.betweenMean ω}
      ≤ Real.exp (-2 * M.G * t ^ 2 / R ^ 2) := by
  classical
  set X : ι → Ω → ℝ := fun i => fun ω => -((M.G : ℝ)⁻¹ * (M.f i ω - ∫ x, M.f i x ∂M.μ)) with hX
  set c : ι → ℝ≥0 := fun _ => ⟨(M.G : ℝ)⁻¹ ^ 2, sq_nonneg _⟩ * (‖(a + R) - a‖₊ / 2) ^ 2 with hc
  have hindepNeg : iIndepFun X M.μ :=
    iIndepFun_centeredScaled.comp (fun _ => fun x : ℝ => -x) (fun _ => by fun_prop)
  have hsubGNeg : ∀ i ∈ M.Sb, HasSubgaussianMGF (X i) (c i) M.μ :=
    fun i hi => (hasSubgaussianMGF_centeredScaled hb i hi).neg
  have hsum := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun (μ := M.μ) (X := X) (c := c)
    hindepNeg (s := M.Sb) hsubGNeg ht
  have hevent : {ω | t ≤ ∑ i ∈ M.Sb, X i ω}
      = {ω | t ≤ (∫ x, M.betweenMean x ∂M.μ) - M.betweenMean ω} := by
    ext ω
    have hrw : ∑ i ∈ M.Sb, X i ω
        = -(M.betweenMean ω - ∫ x, M.betweenMean x ∂M.μ) := by
      simp only [hX, Finset.sum_neg_distrib]
      rw [← betweenMean_sub_integral_eq ω]
    simp only [Set.mem_setOf_eq, hrw]
    constructor <;> intro h <;> linarith
  rw [hevent] at hsum
  refine hsum.trans (le_of_eq ?_)
  have hnormR : ‖(a + R) - a‖ = R := by
    rw [add_sub_cancel_left, Real.norm_of_nonneg hR.le]
  have hcoe : ((∑ _i ∈ M.Sb, c _i : ℝ≥0) : ℝ)
      = (M.G : ℝ) * ((M.G : ℝ)⁻¹ ^ 2 * (R / 2) ^ 2) := by
    rw [Finset.sum_const, M.hcardSb, nsmul_eq_mul]
    push_cast
    rw [hnormR]
  rw [Real.exp_eq_exp, hcoe]
  have hRr : R ^ 2 ≠ 0 := pow_ne_zero 2 hR.ne'
  field_simp

/-- **Group-level concentration (two-sided Hoeffding) — the main bound.** The
    irreducible between-group component `b̄` concentrates around its mean `𝔼[b̄]` with a
    two-sided sub-Gaussian tail whose only sample-size parameter is the *group count* `G`:

        μ.real {ω | t ≤ |b̄ ω − 𝔼[b̄]|}  ≤  2 · exp (−2 · G · t² / R²),   for `0 ≤ t`.

    Union bound over the two one-sided tails. The exponent does not involve the frame count
    `m`: since `b̄` depends only on the group block `Sb` and never on the frame block `Se`,
    enlarging `Se` (more frames per group) leaves this tail untouched. This is the tail-level
    statement of "effective sample size = G", refining the second-moment floor
    `variance_grandMean_ge_between` (`σb²/G ≤ Var b̄`). -/
theorem betweenMean_measureReal_abs_ge_le {a R : ℝ} (hb : M.BetweenBounded a R)
    (hG : 0 < M.G) (hR : 0 < R) {t : ℝ} (ht : 0 ≤ t) :
    M.μ.real {ω | t ≤ |M.betweenMean ω - ∫ x, M.betweenMean x ∂M.μ|}
      ≤ 2 * Real.exp (-2 * M.G * t ^ 2 / R ^ 2) := by
  classical
  set E := ∫ x, M.betweenMean x ∂M.μ with hE
  -- Split the absolute-value tail into the upper and lower one-sided tails.
  have hsub : {ω | t ≤ |M.betweenMean ω - E|}
      ⊆ {ω | t ≤ M.betweenMean ω - E} ∪ {ω | t ≤ E - M.betweenMean ω} := by
    intro ω hω
    simp only [Set.mem_setOf_eq, Set.mem_union] at hω ⊢
    rcases le_abs.mp hω with h1 | h1
    · exact Or.inl h1
    · exact Or.inr (by linarith)
  have hfin₁ : M.μ {ω | t ≤ M.betweenMean ω - E} ≠ ⊤ := measure_ne_top _ _
  have hfin₂ : M.μ {ω | t ≤ E - M.betweenMean ω} ≠ ⊤ := measure_ne_top _ _
  calc M.μ.real {ω | t ≤ |M.betweenMean ω - E|}
      ≤ M.μ.real ({ω | t ≤ M.betweenMean ω - E} ∪ {ω | t ≤ E - M.betweenMean ω}) :=
        measureReal_mono hsub (by finiteness)
    _ ≤ M.μ.real {ω | t ≤ M.betweenMean ω - E}
          + M.μ.real {ω | t ≤ E - M.betweenMean ω} := measureReal_union_le _ _
    _ ≤ Real.exp (-2 * M.G * t ^ 2 / R ^ 2) + Real.exp (-2 * M.G * t ^ 2 / R ^ 2) :=
        add_le_add (betweenMean_measureReal_ge_le hb hG hR ht)
          (betweenMean_measureReal_le_le hb hG hR ht)
    _ = 2 * Real.exp (-2 * M.G * t ^ 2 / R ^ 2) := by ring

/-- **Group-level concentration on the genuine (`ℝ≥0∞`-valued) measure.** The two-sided
    Hoeffding bound, restated for `μ` itself rather than `μ.real`, via `ofReal`:

        μ {ω | t ≤ |b̄ ω − 𝔼[b̄]|}  ≤  ENNReal.ofReal (2 · exp (−2 · G · t² / R²)).

    Same content as `betweenMean_measureReal_abs_ge_le`; this is the form directly comparable
    to Chebyshev statements like `meas_ge_le_variance_div_sq` (which also live in `ℝ≥0∞`). The
    `G`-governed exponential rate is strictly sharper than the `Θ(1/G)` Chebyshev decay the
    variance floor alone would give. -/
theorem betweenMean_measure_abs_ge_le {a R : ℝ} (hb : M.BetweenBounded a R)
    (hG : 0 < M.G) (hR : 0 < R) {t : ℝ} (ht : 0 ≤ t) :
    M.μ {ω | t ≤ |M.betweenMean ω - ∫ x, M.betweenMean x ∂M.μ|}
      ≤ ENNReal.ofReal (2 * Real.exp (-2 * M.G * t ^ 2 / R ^ 2)) := by
  rw [← ofReal_measureReal (μ := M.μ)
        (s := {ω | t ≤ |M.betweenMean ω - ∫ x, M.betweenMean x ∂M.μ|}) (measure_ne_top _ _)]
  exact ENNReal.ofReal_le_ofReal (betweenMean_measureReal_abs_ge_le hb hG hR ht)

/-- **Frames cannot tighten the group-level tail (formal `m`-invariance).** For two
    one-way random-effects models that share the between-group data — the same measure `μ`,
    the same effect family `f`, the same group index block `Sb`, and the same group count
    `G` — but differ *arbitrarily* in the within-group block `Se` and the frame count `m`,
    the between-group component, its mean, and its entire two-sided concentration bound are
    *literally identical*. The estimator `b̄` and every quantity in
    `betweenMean_measureReal_abs_ge_le` are functions of `(μ, f, Sb, G)` alone and never
    touch `Se` or `m`; hence no change to the frame count alters the tail by even an
    infinitesimal. This is the formal content of "adding frames cannot push the bound below
    the group-count floor": the bound does not move at all, rather than merely improving
    slowly.

    References (interpretation). This exact `m`-invariance statement is original — a
    corollary of the Hoeffding tail (`betweenMean_measureReal_abs_ge_le`) together with the
    definition of `betweenMean`. Conceptual anchor: within-unit replicates (here, "frames")
    are *pseudoreplicates* — not independent inferential units, so they cannot sharpen a
    bound governed by the number of independent units `G` (Hurlbert, *Pseudoreplication and
    the design of ecological field experiments*, Ecological Monographs 54(2) (1984),
    187–211). In-domain ML-evaluation reading: effective sample size in agent / world-model
    evaluation (Agarwal, Schwarzer, Castro, Courville & Bellemare, *Deep reinforcement
    learning at the edge of the statistical precipice*, NeurIPS 34 (2021)); clustered /
    dependent data make naive resampling overoptimistic (Hornung et al., *Evaluating machine
    learning models in non-standard settings*, arXiv:2310.15108 (2023)). -/
theorem betweenMean_tail_indep_frames {Ω ι : Type*} [MeasurableSpace Ω]
    (M₁ M₂ : OneWayRandomEffects Ω ι)
    (hμ : M₁.μ = M₂.μ) (hf : M₁.f = M₂.f) (hSb : M₁.Sb = M₂.Sb) (hG : M₁.G = M₂.G) :
    M₁.betweenMean = M₂.betweenMean
      ∧ (∫ x, M₁.betweenMean x ∂M₁.μ) = (∫ x, M₂.betweenMean x ∂M₂.μ)
      ∧ ∀ t : ℝ,
          M₁.μ.real {ω | t ≤ |M₁.betweenMean ω - ∫ x, M₁.betweenMean x ∂M₁.μ|}
            = M₂.μ.real {ω | t ≤ |M₂.betweenMean ω - ∫ x, M₂.betweenMean x ∂M₂.μ|} := by
  have hbm : M₁.betweenMean = M₂.betweenMean := by
    unfold betweenMean; rw [hf, hSb, hG]
  refine ⟨hbm, by rw [hbm, hμ], fun t => by rw [hbm, hμ]⟩

end OneWayRandomEffects

end LewmValidity

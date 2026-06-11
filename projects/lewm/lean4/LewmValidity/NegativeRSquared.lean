/-
Copyright (c) 2026 Dhruv Gupta. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Dhruv Gupta
-/
/-
  lewm-validity-lean — `negative_r2_prob_floor`: the anti-concentration counterpart to the
  Check-4 variance floor.

  The `variance_floor` / `OneWayRandomEffects` block (in `LewmValidity.lean`) gives the
  *upper*-tail picture: the group-averaged estimator's variance is bounded BELOW by
  `σb² / G`, and the exact decomposition pins the excess at `σw² / (G·m)`. Those are the
  statements an antitone/Hoeffding analysis can see — they say the estimator concentrates
  no faster than the group count allows.

  This module witnesses the *lower*-tail / anti-concentration phenomenon that the
  accompanying experiments show and that no upper-tail bound can rule out: with FEW groups,
  the method-of-moments / ANOVA estimator of the between-group variance component `σ̂b²`
  (equivalently the estimated decoding-R²) is NEGATIVE with probability bounded below by a
  positive constant `δ`, *even though the true between-group variance `σb²` is strictly
  positive*. More frames per group cannot fix this — only more groups can — because the
  bound is a property of the group-count `G` alone.

  STRATEGY (finite/discrete surrogate). We avoid continuous F/χ²-distribution theory, which
  this Mathlib does not carry for variance-component estimators, by building a concrete
  finite probability space:
    • each of `G` groups draws an iid effect `b_g` from the symmetric 3-point law on
      `{-1, 0, +1}` (states `Fin 3`) with `P(±1) = p`, `P(0) = 1 - 2p`, `0 < p < 1/2`;
    • the genuine between-group variance is `σb² = Var(b_g) = 2p > 0` (proven, not posited);
    • the one-frame-per-group method-of-moments estimator is the between-group SAMPLE
      variance of the observed group means minus the within-group estimate `w > 0`:
        `σ̂b²(c) = (1/G) Σ_g (val(c g) − mean)² − w`;
    • on the event "all groups land in the same state" the sample spread is `0`, so
      `σ̂b²(c) = −w < 0`; that event has probability `≥ (1 − 2p)^G =: δ > 0`, computed by
      a single-configuration lower bound (the constant config "all in the middle state").

  This is a complete existential witness: `σb² = 2p > 0` is a real positive number, the
  negative-estimate event is non-trivial, and `δ` is a real positive constant. The
  strongest form quantifies over all `0 < p < 1/2`, `0 < w`, and `1 ≤ G`; a fully closed
  corollary instantiates `G = 2`, `p = 1/3`, `w = 1/2`, `δ = 1/9`.
-/
import LewmValidity
import Mathlib

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

namespace LewmValidity

namespace NegativeRSquared

/-! ## The per-group effect distribution on `{-1, 0, +1}`

A single group effect `b_g` takes values in `Fin 3` decoded by `effectVal`
(`0 ↦ -1`, `1 ↦ 0`, `2 ↦ +1`) under the symmetric law `singleMass p`
(`P(±1) = p`, `P(0) = 1 - 2p`). With `0 < p < 1/2` this is a genuine probability law with
strictly positive variance `2p`. -/

/-- Decoding of the three effect states to real effect values: `-1, 0, +1`. -/
def effectVal : Fin 3 → ℝ
  | 0 => -1
  | 1 => 0
  | 2 => 1

/-- The symmetric 3-point mass function on the effect states, as an `ℝ≥0∞`-valued function:
    `P(state 0) = P(state 2) = p` and `P(state 1) = 1 - 2p`. -/
noncomputable def singleMass (p : ℝ) : Fin 3 → ℝ≥0∞
  | 0 => ENNReal.ofReal p
  | 1 => ENNReal.ofReal (1 - 2 * p)
  | 2 => ENNReal.ofReal p

/-- The three masses sum to one when `0 ≤ p ≤ 1/2`. -/
lemma singleMass_sum {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1 / 2) :
    ∑ i, singleMass p i = 1 := by
  have h2p : 0 ≤ 1 - 2 * p := by linarith
  rw [Fin.sum_univ_three]
  simp only [singleMass]
  rw [← ENNReal.ofReal_add hp0 h2p, ← ENNReal.ofReal_add (by linarith) hp0]
  rw [show p + (1 - 2 * p) + p = 1 by ring, ENNReal.ofReal_one]

/-- The per-group effect law `b_g ∼ singleMass p` as a `PMF (Fin 3)`. -/
noncomputable def singlePMF {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1 / 2) : PMF (Fin 3) :=
  PMF.ofFintype (singleMass p) (singleMass_sum hp0 hp1)

/-- The mean effect is `0` (the law is symmetric). -/
lemma expectation_effectVal {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1 / 2) :
    ∫ i, effectVal i ∂((singlePMF hp0 hp1).toMeasure) = 0 := by
  rw [PMF.integral_eq_sum]
  rw [Fin.sum_univ_three]
  simp only [singlePMF, PMF.ofFintype_apply, singleMass, effectVal, smul_eq_mul]
  rw [ENNReal.toReal_ofReal hp0, ENNReal.toReal_ofReal (by linarith : (0:ℝ) ≤ 1 - 2 * p)]
  ring

/-- **True between-group variance is `2p` — and strictly positive.** The genuine
    (measure-theoretic) variance of the group effect `b_g` under its law equals `2p`, which
    is `> 0` for `p > 0`. This certifies that the negative-estimate phenomenon below is NOT
    a vacuity: the quantity being estimated is really positive. -/
theorem variance_effectVal {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1 / 2) :
    variance effectVal (singlePMF hp0 hp1).toMeasure = 2 * p := by
  rw [variance_eq_integral measurable_from_top.aemeasurable]
  rw [expectation_effectVal hp0 hp1]
  simp only [sub_zero]
  rw [PMF.integral_eq_sum, Fin.sum_univ_three]
  simp only [singlePMF, PMF.ofFintype_apply, singleMass, effectVal, smul_eq_mul]
  rw [ENNReal.toReal_ofReal hp0, ENNReal.toReal_ofReal (by linarith : (0:ℝ) ≤ 1 - 2 * p)]
  ring

/-- The true between-group variance is strictly positive when `p > 0`. -/
theorem variance_effectVal_pos {p : ℝ} (hp0 : 0 < p) (hp1 : p ≤ 1 / 2) :
    0 < variance effectVal (singlePMF hp0.le hp1).toMeasure := by
  rw [variance_effectVal hp0.le hp1]; linarith

/-! ## The joint iid configuration law and the method-of-moments estimator

The configuration of all `G` group effects is a point of `Fin G → Fin 3`. Under the iid law
`jointPMF`, the masses multiply: `P(c) = ∏_g singleMass p (c g)`. The one-frame-per-group
method-of-moments estimator of `σb²` is the between-group SAMPLE variance of the observed
group means `val(c g)` minus the within-group estimate `w`. -/

variable {G : ℕ}

/-- The joint iid law over the `G` group-effect states: a `PMF (Fin G → Fin 3)` whose mass
    at a configuration `c` is the product `∏_g singleMass p (c g)`. -/
noncomputable def jointPMF (G : ℕ) {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1 / 2) :
    PMF (Fin G → Fin 3) :=
  PMF.ofFintype (fun c => ∏ g, singleMass p (c g)) <| by
    have hkey : ∏ _g : Fin G, ∑ i, singleMass p i
        = ∑ c : Fin G → Fin 3, ∏ g, singleMass p (c g) := by
      rw [Finset.prod_univ_sum (fun _ => Finset.univ) (fun _ i => singleMass p i)]
      rw [Fintype.piFinset_univ]
    rw [← hkey]
    simp [singleMass_sum hp0 hp1]

/-- The sample mean of the observed group effects `val(c g)` over the `G` groups. -/
noncomputable def groupMean (c : Fin G → Fin 3) : ℝ :=
  (G : ℝ)⁻¹ * ∑ g, effectVal (c g)

/-- The between-group SAMPLE variance of the observed group means: `(1/G) Σ_g (val(c g) − x̄)²`.
    This is the between-group mean square that the method-of-moments estimator uses. -/
noncomputable def sampleBetweenVar (c : Fin G → Fin 3) : ℝ :=
  (G : ℝ)⁻¹ * ∑ g, (effectVal (c g) - groupMean c) ^ 2

/-- **The method-of-moments / ANOVA estimator of the between-group variance component.** It
    is the between-group sample variance of the observed group means MINUS the within-group
    variance estimate `w`. This is the genuine moment estimator: subtracting the within
    estimate is exactly what allows the estimate to go negative even though `σb² > 0`. -/
noncomputable def sigmaHatB (w : ℝ) (c : Fin G → Fin 3) : ℝ :=
  sampleBetweenVar c - w

/-- The sample between-group variance is always nonnegative (it is an average of squares). -/
lemma sampleBetweenVar_nonneg (c : Fin G → Fin 3) : 0 ≤ sampleBetweenVar c := by
  unfold sampleBetweenVar
  rcases Nat.eq_zero_or_pos G with hG | hG
  · subst hG; simp
  · have : (0 : ℝ) ≤ (G : ℝ)⁻¹ := by positivity
    apply mul_nonneg this
    apply Finset.sum_nonneg
    intro g _; positivity

/-- The constant configuration in which every group lands in the middle state (`b_g = 0`). -/
def constConfig (G : ℕ) : Fin G → Fin 3 := fun _ => 1

/-- Every observed group effect is `0` on the constant (all-middle) configuration. -/
lemma effectVal_constConfig (g : Fin G) : effectVal (constConfig G g) = 0 := by
  simp [constConfig, effectVal]

/-- The sample mean of the observed group effects is `0` on the constant configuration. -/
lemma groupMean_constConfig : groupMean (constConfig G) = 0 := by
  unfold groupMean
  simp [effectVal_constConfig]

/-- On a configuration in which all groups share one state, every observed group mean equals
    the common value, so the sample between-group variance is `0`. -/
lemma sampleBetweenVar_constConfig : sampleBetweenVar (constConfig G) = 0 := by
  unfold sampleBetweenVar
  rw [groupMean_constConfig]
  simp [effectVal_constConfig]

/-- **The estimator is strictly negative on the constant configuration.** When all groups
    land in the same state the between-group sample variance vanishes, so the
    method-of-moments estimate is `0 − w = −w < 0`: the estimate undershoots zero even though
    the true between-group variance is `2p > 0`. -/
lemma sigmaHatB_constConfig_neg {w : ℝ} (hw : 0 < w) :
    sigmaHatB w (constConfig G) < 0 := by
  unfold sigmaHatB
  rw [sampleBetweenVar_constConfig]
  linarith

/-! ## The probability floor

The negative-estimate event `{c | σ̂b² c < 0}` contains the constant configuration, whose
mass under the iid law is `∏_g (1 - 2p) = (1 - 2p)^G`. Monotonicity of the measure then
gives the floor `P(σ̂b² < 0) ≥ (1 - 2p)^G = δ > 0`. -/

/-- The mass the iid law assigns to the constant (all-middle) configuration is `(1 - 2p)^G`. -/
lemma jointPMF_constConfig {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1 / 2) :
    jointPMF G hp0 hp1 (constConfig G) = ENNReal.ofReal ((1 - 2 * p) ^ G) := by
  unfold jointPMF constConfig
  rw [PMF.ofFintype_apply]
  have hone : ∀ g ∈ (Finset.univ : Finset (Fin G)),
      singleMass p ((fun _ : Fin G => (1 : Fin 3)) g) = ENNReal.ofReal (1 - 2 * p) := by
    intro g _; simp [singleMass]
  rw [Finset.prod_congr rfl hone, Finset.prod_const, Finset.card_univ, Fintype.card_fin,
    ← ENNReal.ofReal_pow (by linarith : (0:ℝ) ≤ 1 - 2 * p)]

/-- **`negative_r2_prob_floor` — strongest form (anti-concentration).** For every
    group count `G`, every effect spread `0 < p < 1/2`, and every positive within-group
    estimate `w`, the method-of-moments estimate of the between-group variance component is
    NEGATIVE with probability at least `δ := (1 - 2p)^G > 0`, while the TRUE between-group
    variance `σb² = 2p` is strictly positive.

    The floor `δ = (1 - 2p)^G` depends on `G` alone (one frame per group is fixed): more
    frames per group can never remove this lower-tail mass — only increasing `G` shrinks `δ`,
    and at small `G` it is large (e.g. `δ = 1/9` at `G = 2`, `p = 1/3`; see
    `negative_r2_witness`). This is the anti-concentration counterpart of `variance_floor`
    (`LewmValidity.variance_floor`), which bounds the estimator's variance from below by
    `σb² / G`; here we bound from below the probability that the estimate falls below zero. -/
theorem negative_r2_prob_floor {p : ℝ} (hp0 : 0 < p) (hp1 : p < 1 / 2)
    {w : ℝ} (hw : 0 < w) :
    0 < variance effectVal (singlePMF hp0.le hp1.le).toMeasure ∧
      ENNReal.ofReal ((1 - 2 * p) ^ G)
        ≤ (jointPMF G hp0.le hp1.le).toMeasure {c | sigmaHatB w c < 0} ∧
      0 < (1 - 2 * p) ^ G := by
  refine ⟨variance_effectVal_pos hp0 hp1.le, ?_, ?_⟩
  · -- the negative-estimate event contains the constant configuration
    have hsub : ({constConfig G} : Set (Fin G → Fin 3)) ⊆ {c | sigmaHatB w c < 0} := by
      intro c hc
      rw [Set.mem_singleton_iff] at hc
      subst hc
      exact sigmaHatB_constConfig_neg hw
    calc ENNReal.ofReal ((1 - 2 * p) ^ G)
        = (jointPMF G hp0.le hp1.le).toMeasure {constConfig G} := by
          rw [PMF.toMeasure_apply_singleton _ _ (MeasurableSet.singleton _),
            jointPMF_constConfig hp0.le hp1.le]
      _ ≤ (jointPMF G hp0.le hp1.le).toMeasure {c | sigmaHatB w c < 0} :=
          measure_mono hsub
  · have : 0 < 1 - 2 * p := by linarith
    positivity

/-- **Real-valued probability form.** The same floor with the probability written as a
    real number via `Measure.real`: the method-of-moments estimate is negative with
    probability at least the positive constant `(1 - 2p)^G`, while the true between-group
    variance `2p` is positive. -/
theorem negative_r2_prob_floor_real {p : ℝ} (hp0 : 0 < p) (hp1 : p < 1 / 2)
    {w : ℝ} (hw : 0 < w) :
    0 < variance effectVal (singlePMF hp0.le hp1.le).toMeasure ∧
      (1 - 2 * p) ^ G
        ≤ (jointPMF G hp0.le hp1.le).toMeasure.real {c | sigmaHatB w c < 0} ∧
      0 < (1 - 2 * p) ^ G := by
  obtain ⟨hvar, hmeas, hδ⟩ := negative_r2_prob_floor hp0 hp1 hw
  refine ⟨hvar, ?_, hδ⟩
  rw [Measure.real]
  have hle := ENNReal.toReal_le_toReal (by simp) (measure_ne_top _ _) |>.mpr hmeas
  rwa [ENNReal.toReal_ofReal hδ.le] at hle

/-- **Fully closed existential witness.** A concrete model — `G = 2` groups, effect
    spread `p = 1/3` (so `σb² = 2/3 > 0`), within-group estimate `w = 1/2` — on which the
    method-of-moments estimate of the between-group variance is negative with probability at
    least `δ = 1/9`. This is a single self-contained counterexample: a strictly positive true
    between-group variance, yet a fixed positive probability of a negative estimate, at small
    `G`. -/
theorem negative_r2_witness :
    let hp0 : (0:ℝ) < 1 / 3 := by norm_num
    let hp1 : (1:ℝ) / 3 < 1 / 2 := by norm_num
    0 < variance effectVal (singlePMF hp0.le hp1.le).toMeasure ∧
      (1 / 9 : ℝ) ≤ (jointPMF 2 hp0.le hp1.le).toMeasure.real
        {c | sigmaHatB (1 / 2) c < 0} := by
  intro hp0 hp1
  obtain ⟨hvar, hmeas, _⟩ := negative_r2_prob_floor_real (G := 2)
    (p := 1 / 3) hp0 hp1 (w := 1 / 2) (by norm_num)
  refine ⟨hvar, ?_⟩
  have hδ : ((1 : ℝ) - 2 * (1 / 3)) ^ 2 = 1 / 9 := by norm_num
  rwa [hδ] at hmeas

end NegativeRSquared

/-! ## References (negative variance-component estimates)

Covers `effectVal`, `singleMass`, `singlePMF`, `variance_effectVal`, `variance_effectVal_pos`,
`jointPMF`, `groupMean`, `sampleBetweenVar`, `sigmaHatB`, `sampleBetweenVar_constConfig`,
`sigmaHatB_constConfig_neg`, `jointPMF_constConfig`, `negative_r2_prob_floor`,
`negative_r2_prob_floor_real`, `negative_r2_witness`.

The phenomenon (negative ANOVA variance-component estimates).
  • S. R. Searle, G. Casella & C. E. McCulloch, *Variance Components* (Wiley, 1992), Ch. 3.
    The classic statement that the one-way ANOVA / method-of-moments estimator of the
    between-group variance component can be negative with positive probability even when the
    true component is strictly positive — exactly the anti-concentration this module witnesses.
  • H. O. Hartley & J. N. K. Rao, "Maximum-likelihood estimation for the mixed analysis of
    variance model," *Biometrika* **54** (1967) 93–108. Negative estimates and the contrast
    with constrained ML.
  • L. R. LaMotte, "On non-negative quadratic unbiased estimation of variance components,"
    *JASA* **68** (1973) 728–730. Why no quadratic unbiased estimator avoids negative values.

Few-clusters inference (why small `G` is the operative regime).
  • A. C. Cameron & D. L. Miller, "A Practitioner's Guide to Cluster-Robust Inference,"
    *J. Human Resources* **50**(2) (2015) 317–372. The small-number-of-clusters failure mode;
    the effective sample size is the cluster count, so few clusters give unstable, sign-
    indefinite component estimates.

The design-effect floor it complements (the upper-tail companion in `LewmValidity.lean`).
  • B. R. Moulton, "Random group effects and the precision of regression estimates,"
    *J. Econometrics* **32**(3) (1986) 385–397.
  • L. Kish, *Survey Sampling* (Wiley, 1965), §5.4, §8.2. Effective sample size
    `n_eff = n / Deff`. `variance_floor` formalizes this floor from above; the present module
    is the matching lower-tail (negative-estimate) statement that more frames cannot cure.

Foundation (the load-bearing mathematics).
  • Variance of a discrete random variable as `𝔼[(X − 𝔼X)²]` over its point masses, here via
    `ProbabilityTheory.variance_eq_integral` and `PMF.integral_eq_sum`.
  • Independence-by-product of finite laws (`PMF.ofFintype` with `Finset.prod_univ_sum`):
    the iid configuration law whose point masses factor, underwriting the `(1−2p)^G` floor. -/

end LewmValidity

/-
Copyright (c) 2026 Dhruv Gupta. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Dhruv Gupta
-/
/-
  lewm-validity-lean — machine-checked structural witnesses for the four-check
  representation-probing protocol.

  This file formalizes two of the failure modes the protocol checks for: the
  frame/episode split-leakage gap (Check 3) and the one-way random-effects variance
  identity governing clustered effective sample size (Check 4). Each result shows the
  failure is structural — fixed by the σ-algebra or the group count — and cannot be
  closed by collecting more frames per group.
-/
import Mathlib

open MeasureTheory ProbabilityTheory Filter Topology

namespace LewmValidity

/-! ## Shared model — `RandomEffectsModel` -/

/-- Random-effects latent model `z = h s + offset e + noise`: a bundle of laws.
    An alternative encoding swaps `offset : E → Z` for a random variable
    `b : Ω → Z` with `IndepFun` laws, for variance decompositions that need joint
    measurability of offset and state. -/
structure RandomEffectsModel (S Z E : Type*)
    [MeasurableSpace S] [MeasurableSpace Z] [MeasurableSpace E] where
  μs     : Measure S
  h      : S → Z
  offset : E → Z
  bμ     : Measure E
  noise  : Measure Z

/-! `bayesRisk` is parametric over the conditioning σ-algebra `G`: the σ-algebra
    carries the frame-vs-group distinction. `bayesRisk G := ∫ (s - μ[s | G])^2`, with
    `frame := σ(z,e)` and `group := σ(z)`, built on `MeasureTheory.condExp` and
    `ProbabilityTheory.variance`. -/

/-! ## Frame-split inconsistency (witnesses Check 3) -/

section FrameSplit
variable {Ω : Type*} {m₀ m₁ m₂ : MeasurableSpace Ω} {μ : Measure[m₀] Ω} {X : Ω → ℝ}

/-- Bayes (L²) risk of decoding `X` from a sub-σ-algebra `m`: `Var X − Var μ[X|m]`
    (= `∫ (X − μ[X|m])²` by the law of total variance). The frame split conditions on the
    finer `σ(z,e)`; the episode/group split on the coarser `σ(z)`. -/
noncomputable def bayesRisk (m : MeasurableSpace Ω) (X : Ω → ℝ) (μ : Measure[m₀] Ω) : ℝ :=
  variance X μ - variance (μ[X | m]) μ

/-- More information cannot raise the variance of the conditional expectation: for nested
    σ-algebras the coarser conditional expectation has the smaller variance. Law of total
    variance applied to `μ[X|m₂]` over `m₁`, plus the tower property. -/
theorem variance_condExp_mono [IsProbabilityMeasure μ]
    (hm₁₂ : m₁ ≤ m₂) (hm₂ : m₂ ≤ m₀) (hX : MemLp X 2 μ) :
    variance (μ[X | m₁]) μ ≤ variance (μ[X | m₂]) μ := by
  have hltv := integral_condVar_add_variance_condExp (μ := μ) (m := m₁)
    (hm₁₂.trans hm₂) (hX.condExp (m := m₂))
  have htower : μ[μ[X | m₂] | m₁] =ᵐ[μ] μ[X | m₁] := condExp_condExp_of_le hm₁₂ hm₂
  rw [variance_congr htower] at hltv
  have hcv : 0 ≤ μ[Var[μ[X | m₂]; μ | m₁]] := by
    refine integral_nonneg_of_ae ?_
    have hnn : (0 : Ω → ℝ) ≤ᵐ[μ] (μ[X | m₂] - μ[μ[X | m₂] | m₁]) ^ 2 := by
      filter_upwards with ω
      simp only [Pi.zero_apply, Pi.pow_apply, Pi.sub_apply]
      positivity
    simpa [condVar] using condExp_nonneg hnn
  linarith [hltv, hcv]

/-- **Check 3.** Bayes risk is antitone in the σ-algebra: a frame split (finer `m₂`)
    cannot report higher risk than an episode split (coarser `m₁`). Leakage through the finer
    σ-algebra can only lower the apparent risk, never raise it. -/
theorem bayesRisk_antitone [IsProbabilityMeasure μ]
    (hm₁₂ : m₁ ≤ m₂) (hm₂ : m₂ ≤ m₀) (hX : MemLp X 2 μ) :
    bayesRisk m₂ X μ ≤ bayesRisk m₁ X μ := by
  simp only [bayesRisk]
  linarith [variance_condExp_mono hm₁₂ hm₂ hX]

end FrameSplit

section FrameSplitGap
variable {Ω : Type*} {m₀ m₁ m₂ : MeasurableSpace Ω} {μ : Measure[m₀] Ω} {X : Ω → ℝ}

/-- **Check 3.** The episode-vs-frame Bayes-risk gap equals the explained variance
    of the finer conditional expectation given the coarser σ-algebra — an intrinsic quantity
    of the two σ-algebras alone. It does not reference the within-group frame count, so more
    frames per episode cannot close it: only changing the split (the σ-algebra) can. -/
theorem bayesRisk_gap [IsProbabilityMeasure μ]
    (hm₁₂ : m₁ ≤ m₂) (hm₂ : m₂ ≤ m₀) (hX : MemLp X 2 μ) :
    bayesRisk m₁ X μ - bayesRisk m₂ X μ = μ[Var[μ[X | m₂]; μ | m₁]] := by
  simp only [bayesRisk]
  have hltv := integral_condVar_add_variance_condExp (μ := μ) (m := m₁)
    (hm₁₂.trans hm₂) (hX.condExp (m := m₂))
  have htower : μ[μ[X | m₂] | m₁] =ᵐ[μ] μ[X | m₁] := condExp_condExp_of_le hm₁₂ hm₂
  rw [variance_congr htower] at hltv
  linarith [hltv]

end FrameSplitGap

/-! ## Strongest-form explained-variance functional (Check 3)

`bayesRisk_gap` above identifies the episode-vs-frame Bayes-risk gap with a single integral. The
following names that integral as the `explainedVariance` functional and states the witness in its
strongest tractable form: the apparent risk drop from refining the split `m₁ → m₂` is not merely
`≤ 0` (antitone) but equals the integrated conditional variance of the finer predictor, exactly,
sign-definite and quantitatively pinned. The named identity then yields (i) the strict drop with
constant-explicit magnitude whenever any frame information leaks, and (ii) the fraction of total
state variance that the split manufactures — the decoding-R² inflation. These formalize "a frame
split lowers apparent risk by precisely the variance the frame σ-algebra explains beyond the
episode σ-algebra," with no appeal to the within-group frame count.

References. Law of total variance (`Var X = 𝔼[Var[X | m]] + Var 𝔼[X | m]`), here in its conditional
form over nested σ-algebras; Mathlib `MeasureTheory`/`ProbabilityTheory.condVar`
(`Var[X; μ | m] := μ[(X − μ[X | m])² | m]`) and `integral_condVar_add_variance_condExp`. -/

section FrameSplitExplained
variable {Ω : Type*} {m₀ m₁ m₂ : MeasurableSpace Ω} {μ : Measure[m₀] Ω} {X : Ω → ℝ}

/-- The variance of the finer-frame predictor `μ[X | m₂]` explained by refining the split from the
    coarser episode σ-algebra `m₁` to the finer frame σ-algebra `m₂`, integrated against `μ`. This
    is the named functional sitting on the right of `bayesRisk_gap`: the conditional variance of the
    `m₂`-predictor given `m₁`, averaged over `Ω`. It is an intrinsic quantity of the pair of
    σ-algebras and `X` alone — independent of the within-group frame count. -/
noncomputable def explainedVariance (m₁ m₂ : MeasurableSpace Ω) (X : Ω → ℝ) (μ : Measure[m₀] Ω) :
    ℝ :=
  μ[Var[μ[X | m₂]; μ | m₁]]

/-- The explained variance is nonnegative: it is the integral of a conditional variance, which is a
    conditional expectation of a square and hence `≥ 0` almost everywhere. A refinement of the split
    can only *add* apparent explanatory power, never subtract it. -/
theorem explainedVariance_nonneg : 0 ≤ explainedVariance m₁ m₂ X μ := by
  refine integral_nonneg_of_ae ?_
  have hnn : (0 : Ω → ℝ) ≤ᵐ[μ] (μ[X | m₂] - μ[μ[X | m₂] | m₁]) ^ 2 := by
    filter_upwards with ω
    simp only [Pi.zero_apply, Pi.pow_apply, Pi.sub_apply]
    positivity
  simpa [condVar] using condExp_nonneg hnn

/-- **Check 3, named identity.** For nested splits `m₁ ≤ m₂ ≤ m₀` the apparent Bayes-risk drop
    from refining the episode split to the frame split equals the explained variance functional —
    the restatement of `bayesRisk_gap` in terms of `explainedVariance`. The leakage gap is the
    integrated conditional variance the frame σ-algebra adds over the episode σ-algebra. -/
theorem bayesRisk_sub_eq_explainedVariance [IsProbabilityMeasure μ]
    (hm₁₂ : m₁ ≤ m₂) (hm₂ : m₂ ≤ m₀) (hX : MemLp X 2 μ) :
    bayesRisk m₁ X μ - bayesRisk m₂ X μ = explainedVariance m₁ m₂ X μ :=
  bayesRisk_gap hm₁₂ hm₂ hX

/-- **Check 3, quantitative strict gap.** Whenever the frame σ-algebra leaks any information about
    `X` beyond the episode σ-algebra (`0 < explainedVariance`), the frame split reports strictly
    lower Bayes risk than the episode split, and the drop is pinned to the explained variance
    exactly: the magnitude is constant-explicit, not merely bounded. This strengthens
    `bayesRisk_antitone` (`≤`), upgrading the weak inequality to a strict one with a named, computed
    gap. -/
theorem bayesRisk_lt_of_explainedVariance_pos [IsProbabilityMeasure μ]
    (hm₁₂ : m₁ ≤ m₂) (hm₂ : m₂ ≤ m₀) (hX : MemLp X 2 μ)
    (hpos : 0 < explainedVariance m₁ m₂ X μ) :
    bayesRisk m₁ X μ - bayesRisk m₂ X μ = explainedVariance m₁ m₂ X μ ∧
      bayesRisk m₂ X μ < bayesRisk m₁ X μ := by
  have heq := bayesRisk_sub_eq_explainedVariance hm₁₂ hm₂ hX
  exact ⟨heq, by linarith [heq, hpos]⟩

/-- The apparent decoding-R² inflation manufactured by a frame split: the fraction of total state
    variance `explainedVariance / Var X` that the refinement `m₁ → m₂` adds. Under non-degenerate
    `X` (`Var X ≠ 0`), the Bayes-risk gap is exactly this fraction of the total variance. This names
    "the fraction of total variance that frame-leakage manufactures": a probe can inflate apparent
    decoding-R² by precisely `explainedVariance / Var X` purely by conditioning on the finer split. -/
theorem bayesRisk_sub_eq_variance_mul_inflation [IsProbabilityMeasure μ]
    (hm₁₂ : m₁ ≤ m₂) (hm₂ : m₂ ≤ m₀) (hX : MemLp X 2 μ) (hVar : variance X μ ≠ 0) :
    bayesRisk m₁ X μ - bayesRisk m₂ X μ =
      variance X μ * (explainedVariance m₁ m₂ X μ / variance X μ) := by
  rw [bayesRisk_sub_eq_explainedVariance hm₁₂ hm₂ hX]
  field_simp

end FrameSplitExplained

section FrameSplitSep
variable {Ω : Type*} {m₀ : MeasurableSpace Ω} {μ : Measure[m₀] Ω} {X : Ω → ℝ}

private lemma variance_const [IsProbabilityMeasure μ] (c : ℝ) :
    variance (fun _ : Ω => c) μ = 0 := by
  have h := variance_const_add (μ := μ) (X := (0 : Ω → ℝ)) aestronglyMeasurable_const c
  simpa using h.trans (variance_zero μ)

/-- **Check 3: maximal separation.** With the full σ-algebra (perfect within-episode
    information) the Bayes risk is `0`; with the trivial σ-algebra (no cross-episode
    information) it equals the entire state variance. A frame split can report risk `0` on the
    very `X` whose honest cross-episode risk is `Var X` — i.e. group `R² ≤ 0`. -/
theorem frame_group_separation [IsProbabilityMeasure μ]
    (hX : MemLp X 2 μ) (hXm : StronglyMeasurable X) :
    bayesRisk m₀ X μ = 0 ∧ bayesRisk (⊥ : MeasurableSpace Ω) X μ = variance X μ := by
  refine ⟨?_, ?_⟩
  · rw [bayesRisk, condExp_of_stronglyMeasurable le_rfl hXm (hX.integrable one_le_two), sub_self]
  · rw [bayesRisk, condExp_bot, variance_const, sub_zero]

/-- **Check 3: the gap does not vanish and is not closeable by data.** The episode/frame
    risk gap equals the full state variance — a fixed positive quantity for any non-constant
    `X`. No amount of within-episode data changes it; only the split does. -/
theorem frame_split_inconsistent [IsProbabilityMeasure μ]
    (hX : MemLp X 2 μ) (hXm : StronglyMeasurable X) :
    bayesRisk (⊥ : MeasurableSpace Ω) X μ - bayesRisk m₀ X μ = variance X μ := by
  obtain ⟨h0, hbot⟩ := frame_group_separation hX hXm
  rw [h0, hbot, sub_zero]

end FrameSplitSep

section FrameSplitBlind
variable {Ω : Type*} {m₀ m : MeasurableSpace Ω} {μ : Measure[m₀] Ω} {X : Ω → ℝ}

/-- **Check 3, control case.** Conditioning a predictor on the same σ-algebra it
    already uses adds no explained variance. The leakage gap is exactly the information the
    finer (frame) σ-algebra carries beyond the coarser (episode) one, so an episode-blind
    predictor — one restricted to the coarse σ-algebra — sees frame and episode risk coincide. -/
theorem offset_blind_predictors_coincide [IsProbabilityMeasure μ]
    (hm : m ≤ m₀) (hX : MemLp X 2 μ) :
    μ[Var[μ[X | m]; μ | m]] = 0 := by
  linarith [bayesRisk_gap (m₁ := m) (m₂ := m) le_rfl hm hX]

end FrameSplitBlind

/-! ## References — Check 3 (frame-split)

Covers the frame-split block (`bayesRisk`, `variance_condExp_mono`, `bayesRisk_antitone`,
`bayesRisk_gap`, `explainedVariance`, `bayesRisk_sub_eq_explainedVariance`,
`bayesRisk_sub_eq_variance_mul_inflation`, `frame_split_inconsistent`, `frame_group_separation`,
`offset_blind_predictors_coincide`). Augments the inline note above (which records the law of total
variance and the relevant Mathlib `condVar` machinery); not repeated here.

Methodology / domain (grouped vs random splits; identity confounding; leakage).
  • D. R. Roberts, V. Bahn, S. Ciuti, M. S. Boyce, J. Elith, G. Guillera-Arroita, S. Hauenstein,
    J. J. Lahoz-Monfort, B. Schröder, W. Thuiller, D. I. Warton, B. A. Wintle, F. Hartig &
    C. F. Dormann, "Cross-validation strategies for data with temporal, spatial, hierarchical, or
    phylogenetic structure," *Ecography* **40** (2017) 913–929. Grouped/blocked CV; random splits
    underestimate error under structure — the domain anchor for the antitone direction and for the
    novel separation witnesses.
  • J. Hewitt & P. Liang, "Designing and Interpreting Probes with Control Tasks," *EMNLP* (2019).
    Control tasks for probing; the control-task limit motivating `offset_blind_predictors_coincide`
    (a predictor restricted to the coarse σ-algebra sees frame and episode risk coincide).
  • S. Saeb, L. Lonini, A. Jayaraman, D. C. Mohr & K. P. Kording, "The need to approximate the
    use-case in clinical machine learning," *GigaScience* **6** (2017) gix019. Record-wise vs
    subject-wise splitting; identity confounding — domain anchor for the separation witnesses.
  • E. Chaibub Neto, A. Pratap, T. M. Perumal, M. Tummalacherla, P. Snyder, B. M. Bot,
    A. D. Trister, S. H. Friend, L. Mangravite & L. Omberg, "Detecting the impact of subject
    characteristics on machine learning-based diagnostic applications," *npj Digital Medicine*
    **2** (2019) 99. Identity confounding under record-wise splitting.
  • A. Rabinowicz & S. Rosset, "Cross-Validation for Correlated Data," *JASA* **117** (2022)
    718–731. CV under correlation — domain anchor for the frame-split inconsistency witnesses.
  • S. Kaufman, S. Rosset, C. Perlich & O. Stitelman, "Leakage in Data Mining: Formulation,
    Detection, and Avoidance," *ACM TKDD* **6** (2012) Art. 15. Formalizes leakage, of which a
    frame split is an instance.
  • S. Kapoor & A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based
    science," *Patterns* **4** (2023). Catalogue of leakage-induced over-optimism.
  • G. Alain & Y. Bengio, "Understanding intermediate layers using linear classifier probes,"
    arXiv:1610.01644 (2016). Linear classifier probes — the probing setting these checks scrutinize.

Foundation (the load-bearing mathematics).
  • Law of total variance / Eve's law (`Var X = 𝔼[Var[X | G]] + Var 𝔼[X | G]`): J. K. Blitzstein
    & J. Hwang, *Introduction to Probability*, 2nd ed. (CRC Press, 2019), §9.5. Underwrites
    `bayesRisk`, `bayesRisk_gap`, `variance_condExp_mono`, and the floor visible in
    `bayesRisk_antitone`. NOTE: `frame_group_separation`'s quantitative core *is* the law of total
    variance — it is cited here, not claimed as novel.
  • Conditional expectation as the L²/MMSE projection: D. Williams, *Probability with Martingales*
    (Cambridge Univ. Press, 1991), §9.4. The orthogonality underlying the Bayes-(L²)-risk reading of
    `bayesRisk` and the monotonicity in `variance_condExp_mono`.
  • Coefficient of determination R²: `bayesRisk_sub_eq_variance_mul_inflation` is the plain R²
    decomposition (explained / total variance), per Blitzstein & Hwang §9.5. (No causal-path
    coefficient interpretation is invoked.)

Novel witnesses. `frame_split_inconsistent` and `offset_blind_predictors_coincide` are original
machine-checked structural witnesses (this artifact); they are not attributed to prior work. Their
domain motivation is Roberts et al. (2017), Rabinowicz & Rosset (2022), and Saeb et al. (2017) /
Chaibub Neto et al. (2019); the control-task limit for `offset_blind_predictors_coincide` is Hewitt
& Liang (2019). -/

/-! ## Estimator geometry (Check 2): scope boundary, not formalized in this file

This file does not formalize Check 2. The claim would be that an information-preserving
diagonal rescaling drives the k-NN (KSG) mutual-information estimate toward zero while the
true MI is fixed. No placeholder theorems are left in its place, to avoid misrepresenting
the state of the artifact.

The boundary is precise and falls at the estimator/truth divide — where Check 2 lives:
  • Mathlib has no mutual-information primitive (only `klDiv`). The true side (MI invariant
    under a measurable equivalence) becomes formalizable only after vendoring an MI
    definition, and that half is essentially a definitional unfold rather than the witness.
  • Mathlib has no k-nearest-neighbour estimator infrastructure. The substantive claim is
    about the KSG estimator's finite-sample neighbourhood geometry, which cannot be stated
    cleanly here, let alone have its closed form and `a → ∞` limit proven. Formalizing k-NN
    MI-estimator asymptotics is research-level, beyond this companion's scope.

This boundary is recorded rather than discharged, and the claim is not weakened to force a
formal statement. The accompanying experiments (the geometry sweep: KSG 0.55 → 0, probe flat
at 0.99) supply the empirical witness for Check 2; the formal witness remains open. -/

/-! ## Effective sample size = number of groups (witnesses Check 4) -/

/-- Variance of the group-averaged estimator: for pairwise-independent, square-integrable
    per-group terms, the average's variance is `(card)⁻² · Σ Var`. -/
theorem variance_average {Ω ι : Type*} [MeasurableSpace Ω] {Y : ι → Ω → ℝ}
    {s : Finset ι} {μ : Measure Ω} [IsProbabilityMeasure μ]
    (hY : ∀ i ∈ s, MemLp (Y i) 2 μ)
    (hindep : Set.Pairwise (s : Set ι) fun i j => IndepFun (Y i) (Y j) μ) :
    variance ((s.card : ℝ)⁻¹ • ∑ i ∈ s, Y i) μ
      = ((s.card : ℝ)⁻¹) ^ 2 * ∑ i ∈ s, variance (Y i) μ := by
  rw [variance_smul, IndepFun.variance_sum hY hindep]

/-- **Check 4: the variance floor is independent of the within-group frame count.**
    If every held-out group's variance is at least the between-group variance `σb²`, the
    group-averaged estimator has variance at least `σb² / G`, for any number of frames per
    group. The effective sample size is the number of groups `G = s.card`, not the frame
    count: a negative held-out score at small `G` is a power artifact, not absence of
    information. -/
theorem variance_floor {Ω ι : Type*} [MeasurableSpace Ω] {Y : ι → Ω → ℝ}
    {s : Finset ι} {μ : Measure Ω} [IsProbabilityMeasure μ] {σb : ℝ}
    (hY : ∀ i ∈ s, MemLp (Y i) 2 μ)
    (hindep : Set.Pairwise (s : Set ι) fun i j => IndepFun (Y i) (Y j) μ)
    (hfloor : ∀ i ∈ s, σb ^ 2 ≤ variance (Y i) μ) :
    σb ^ 2 / s.card ≤ variance ((s.card : ℝ)⁻¹ • ∑ i ∈ s, Y i) μ := by
  rw [variance_average hY hindep]
  rcases Nat.eq_zero_or_pos s.card with hc | hc
  · rw [Finset.card_eq_zero] at hc; subst hc; simp
  · have hne : (s.card : ℝ) ≠ 0 := by positivity
    have hsum : (s.card : ℝ) * σb ^ 2 ≤ ∑ i ∈ s, variance (Y i) μ := by
      calc (s.card : ℝ) * σb ^ 2 = ∑ _i ∈ s, σb ^ 2 := by
              rw [Finset.sum_const, nsmul_eq_mul]
        _ ≤ ∑ i ∈ s, variance (Y i) μ := Finset.sum_le_sum hfloor
    calc σb ^ 2 / s.card
        = ((s.card : ℝ)⁻¹) ^ 2 * ((s.card : ℝ) * σb ^ 2) := by field_simp
      _ ≤ ((s.card : ℝ)⁻¹) ^ 2 * ∑ i ∈ s, variance (Y i) μ :=
          mul_le_mul_of_nonneg_left hsum (sq_nonneg _)

/-! ## Exact clustered-variance identity (strengthens Check 4 to an equality)

`variance_floor` above gives the one-sided witness: the group-averaged estimator's variance is
bounded below by `σb² / G`, independent of the within-group frame count. That floor is all the
antitone half (`bayesRisk_antitone` for Check 3, `variance_floor` here for Check 4) can see. The
following upgrades the Check-4 floor to the exact one-way random-effects variance decomposition.

The model is the textbook one-way ANOVA mean estimator: `G` groups, `m` frames per group, responses
`Y_{g,j} = θ + b_g + e_{g,j}` with between-group effects `b_g` (mean `0`, variance `σb²`) and
within-group noise `e_{g,j}` (mean `0`, variance `σw²`), *all mutually independent*. Averaging,
`Ȳ = (1/G) Σ_g (1/m) Σ_j Y_{g,j} = θ + (1/G) Σ_g b_g + (1/(G·m)) Σ_{g,j} e_{g,j}`: the grand mean is
the constant `θ` plus a between-group average over `G` terms and a within-group grand average over
`G·m` terms. The exact identity is

    variance Ȳ μ = σb² / G + σw² / (G·m),

proven by composing `variance_average` at *both* levels (with the exact cardinalities `G` and `G·m`)
through variance-of-independent-sums (`IndepFun.variance_add`) and `variance_smul`. The constant `θ`
drops by `variance_const_add`; the cross term between the two block-averages vanishes because the two
disjoint index blocks are independent (`iIndepFun.indepFun_finset`, pushed through the summation map
by `IndepFun.comp`). From the equality the floor `σb² / G` is recovered as the `σw² ≥ 0` corollary —
the identity strictly refines `variance_floor`, pinning the gap to the floor at exactly `σw² / (G·m)`,
which decays to `0` in `m`: frames shrink only the within-group term and never beat the group-count
floor `σb² / G`. This is the precise Check-4 statement, now an equality rather than a bound.

The equality uses a dedicated two-level structure, `OneWayRandomEffects`, rather than the flat
`RandomEffectsModel`: the latter carries no explicit within-group variance `σw²` nor the two-level
mutual independence that an equality (as opposed to the one-sided floor) requires. The decomposition
itself is `variance_grandMean_eq`, with corollaries `variance_grandMean_ge_between` (floor),
`variance_grandMean_sub_between_eq` (exact gap), and `variance_grandMean_antitone_in_m` (decay in `m`).

References. Law of total variance / variance of a sum of independent variables
(`Var[X + Y] = Var X + Var Y` for `X ⟂ Y`, Mathlib `ProbabilityTheory.IndepFun.variance_add`); one-way
random-effects ANOVA variance decomposition of the grand mean (Searle, Casella & McCulloch,
*Variance Components*, 1992, §3; Scheffé, *The Analysis of Variance*, 1959, ch. 7). -/

/-- **Two-level one-way random-effects model.** A bundle of laws for the
    grand-mean estimator over `G` groups × `m` frames per group, carrying the explicit within-group
    variance `σw²` and the two-level mutual independence that the equality (not merely the floor)
    requires — neither of which the flat `RandomEffectsModel` exposes.

    The between-group effects and within-group noises are presented as a single mutually-independent
    family `f : ι → Ω → ℝ` indexed by `ι`, partitioned by two disjoint finsets: `Sb` (the `G` group
    effects `b_g`, each of variance `σb²`) and `Se` (the `G·m` frame noises `e_{g,j}`, each of
    variance `σw²`). A single `iIndepFun` hypothesis is *more* structure than positing the per-block
    pairwise independence and the cross-block independence separately: it generates all three. -/
structure OneWayRandomEffects (Ω ι : Type*) [MeasurableSpace Ω] where
  /-- The ambient probability measure on the sample space `Ω`. -/
  μ        : Measure Ω
  /-- `μ` is a probability measure (registered as an instance below). -/
  isProb   : IsProbabilityMeasure μ
  /-- The joint family of centered random effects and noises, indexed by `ι`. -/
  f        : ι → Ω → ℝ
  /-- The between-group index block: the `G` group effects `b_g`. -/
  Sb       : Finset ι
  /-- The within-group index block: the `G·m` frame noises `e_{g,j}`. -/
  Se       : Finset ι
  /-- The number of groups. -/
  G        : ℕ
  /-- The number of frames per group. -/
  m        : ℕ
  /-- The (unknown) grand mean `θ` — a constant offset, irrelevant to the variance. -/
  θ        : ℝ
  /-- The between-group standard deviation: each `b_g` has variance `σb²`. -/
  σb       : ℝ
  /-- The within-group standard deviation: each `e_{g,j}` has variance `σw²`. -/
  σw       : ℝ
  /-- The between- and within-group index blocks are disjoint. -/
  hdisj    : Disjoint Sb Se
  /-- The between-group block has exactly `G` elements (one effect per group). -/
  hcardSb  : Sb.card = G
  /-- The within-group block has exactly `G·m` elements (one noise per frame). -/
  hcardSe  : Se.card = G * m
  /-- Each component is measurable. -/
  hmeas    : ∀ i, Measurable (f i)
  /-- Each component is square-integrable (L²), so variances are finite. -/
  hL2      : ∀ i, MemLp (f i) 2 μ
  /-- The components are mutually independent (the full two-level independence). -/
  hindep   : iIndepFun f μ
  /-- Every between-group effect has variance exactly `σb²`. -/
  hvarSb   : ∀ i ∈ Sb, variance (f i) μ = σb ^ 2
  /-- Every within-group noise has variance exactly `σw²`. -/
  hvarSe   : ∀ i ∈ Se, variance (f i) μ = σw ^ 2

namespace OneWayRandomEffects

variable {Ω ι : Type*} [MeasurableSpace Ω] (M : OneWayRandomEffects Ω ι)

attribute [instance] OneWayRandomEffects.isProb

/-- The grand-mean estimator `Ȳ = θ + (1/G) Σ_{g} b_g + (1/(G·m)) Σ_{g,j} e_{g,j}`: the constant
    `θ` plus the between-group average over the `G` effects and the within-group grand average over
    the `G·m` noises. This is the algebraic rearrangement of `(1/G) Σ_g (1/m) Σ_j (θ + b_g + e_{g,j})`. -/
noncomputable def grandMean : Ω → ℝ :=
  fun ω => M.θ + ((M.G : ℝ)⁻¹ • ∑ i ∈ M.Sb, M.f i) ω
                + ((M.G * M.m : ℝ)⁻¹ • ∑ i ∈ M.Se, M.f i) ω

/-- Variance of a block-average with a common per-term variance `v`: averaging `S.card` pairwise
    independent L² terms each of variance `v` gives variance `(S.card)⁻¹ · v`. This is
    `variance_average` specialized to constant per-term variance, the engine applied at one level. -/
private lemma variance_blockAverage (S : Finset ι) (v : ℝ)
    (hvar : ∀ i ∈ S, variance (M.f i) M.μ = v) :
    variance ((S.card : ℝ)⁻¹ • ∑ i ∈ S, M.f i) M.μ = (S.card : ℝ)⁻¹ * v := by
  have hpair : Set.Pairwise (S : Set ι) fun i j => IndepFun (M.f i) (M.f j) M.μ :=
    fun i _ j _ hij => M.hindep.indepFun hij
  rw [variance_smul, IndepFun.variance_sum (fun i _ => M.hL2 i) hpair]
  rw [Finset.sum_congr rfl hvar, Finset.sum_const, nsmul_eq_mul]
  rcases Nat.eq_zero_or_pos S.card with hc | hc
  · simp [hc]
  · have hne : (S.card : ℝ) ≠ 0 := by positivity
    field_simp

/-- The between-group total `Σ_{g} b_g` and the within-group total `Σ_{g,j} e_{g,j}` are
    independent: the two index blocks `Sb`, `Se` are disjoint, so the tuples they index are
    independent (`iIndepFun.indepFun_finset`), and summing each tuple (a measurable map) preserves
    independence (`IndepFun.comp`). This is what makes the cross term in the decomposition vanish. -/
private lemma indepFun_blockSums :
    IndepFun (∑ i ∈ M.Sb, M.f i) (∑ i ∈ M.Se, M.f i) M.μ := by
  have key := M.hindep.indepFun_finset M.Sb M.Se M.hdisj M.hmeas
  have hcomp := key.comp (φ := fun v : M.Sb → ℝ => ∑ i, v i)
    (ψ := fun v : M.Se → ℝ => ∑ i, v i) (by fun_prop) (by fun_prop)
  have e1 : ((fun v : M.Sb → ℝ => ∑ i, v i) ∘ (fun a (i : M.Sb) => M.f i a))
      = ∑ i ∈ M.Sb, M.f i := by
    funext a; simp only [Function.comp_apply, Finset.sum_apply]
    exact Finset.sum_attach M.Sb (fun i => M.f i a)
  have e2 : ((fun v : M.Se → ℝ => ∑ i, v i) ∘ (fun a (i : M.Se) => M.f i a))
      = ∑ i ∈ M.Se, M.f i := by
    funext a; simp only [Function.comp_apply, Finset.sum_apply]
    exact Finset.sum_attach M.Se (fun i => M.f i a)
  rw [e1, e2] at hcomp
  exact hcomp

/-- The between-group average `b̄` and the within-group grand average `ē` are independent: scaling
    each independent block-total by a constant (a measurable map) preserves independence. -/
private lemma indepFun_blockAverages :
    IndepFun ((M.G : ℝ)⁻¹ • ∑ i ∈ M.Sb, M.f i)
             ((M.G * M.m : ℝ)⁻¹ • ∑ i ∈ M.Se, M.f i) M.μ := by
  have h := M.indepFun_blockSums.comp
    (φ := fun x : ℝ => (M.G : ℝ)⁻¹ * x) (ψ := fun x : ℝ => (M.G * M.m : ℝ)⁻¹ * x)
    (by fun_prop) (by fun_prop)
  have e1 : ((fun x : ℝ => (M.G : ℝ)⁻¹ * x) ∘ ∑ i ∈ M.Sb, M.f i)
      = (M.G : ℝ)⁻¹ • ∑ i ∈ M.Sb, M.f i := by
    funext a; simp [Pi.smul_apply, smul_eq_mul, Finset.sum_apply]
  have e2 : ((fun x : ℝ => (M.G * M.m : ℝ)⁻¹ * x) ∘ ∑ i ∈ M.Se, M.f i)
      = (M.G * M.m : ℝ)⁻¹ • ∑ i ∈ M.Se, M.f i := by
    funext a; simp [Pi.smul_apply, smul_eq_mul, Finset.sum_apply]
  rw [e1, e2] at h
  exact h

/-- **Check 4: exact clustered-variance identity.** The grand-mean estimator's
    variance decomposes exactly into a between-group term and a within-group term:

        variance Ȳ μ = σb² / G + σw² / (G·m).

    The constant `θ` drops; the two block-averages are independent so their variances add; and
    `variance_average` evaluates each block average to its closed form (`σb²/G` over the `G` effects,
    `σw²/(G·m)` over the `G·m` noises). This is strictly stronger than the one-sided `variance_floor`:
    it pins the variance to a single value, not a half-line. -/
theorem variance_grandMean_eq :
    variance M.grandMean M.μ = M.σb ^ 2 / M.G + M.σw ^ 2 / (M.G * M.m) := by
  have hb : variance ((M.G : ℝ)⁻¹ • ∑ i ∈ M.Sb, M.f i) M.μ = M.σb ^ 2 / M.G := by
    have := M.variance_blockAverage M.Sb (M.σb ^ 2) M.hvarSb
    rw [M.hcardSb] at this
    rw [this]; ring
  have he : variance ((M.G * M.m : ℝ)⁻¹ • ∑ i ∈ M.Se, M.f i) M.μ
      = M.σw ^ 2 / (M.G * M.m) := by
    have := M.variance_blockAverage M.Se (M.σw ^ 2) M.hvarSe
    rw [M.hcardSe] at this
    push_cast at this ⊢
    rw [this]; ring
  have hBL2 : MemLp ((M.G : ℝ)⁻¹ • ∑ i ∈ M.Sb, M.f i) 2 M.μ :=
    (memLp_finset_sum' M.Sb (fun i _ => M.hL2 i)).const_smul _
  have hEL2 : MemLp ((M.G * M.m : ℝ)⁻¹ • ∑ i ∈ M.Se, M.f i) 2 M.μ :=
    (memLp_finset_sum' M.Se (fun i _ => M.hL2 i)).const_smul _
  have hgm : M.grandMean
      = fun ω => M.θ + (((M.G : ℝ)⁻¹ • ∑ i ∈ M.Sb, M.f i)
                          + ((M.G * M.m : ℝ)⁻¹ • ∑ i ∈ M.Se, M.f i)) ω := by
    funext ω; simp only [grandMean, Pi.add_apply]; ring
  calc variance M.grandMean M.μ
      = variance (((M.G : ℝ)⁻¹ • ∑ i ∈ M.Sb, M.f i)
                    + ((M.G * M.m : ℝ)⁻¹ • ∑ i ∈ M.Se, M.f i)) M.μ := by
        rw [hgm, variance_const_add (hBL2.aestronglyMeasurable.add hEL2.aestronglyMeasurable)]
    _ = variance ((M.G : ℝ)⁻¹ • ∑ i ∈ M.Sb, M.f i) M.μ
          + variance ((M.G * M.m : ℝ)⁻¹ • ∑ i ∈ M.Se, M.f i) M.μ :=
        M.indepFun_blockAverages.variance_add hBL2 hEL2
    _ = M.σb ^ 2 / M.G + M.σw ^ 2 / (M.G * M.m) := by rw [hb, he]

/-- **Corollary — the identity refines `variance_floor`.** Since the within-group term
    `σw² / (G·m)` is nonnegative, the exact decomposition gives the floor `σb² / G ≤ variance Ȳ μ`
    immediately. The group-count floor of `variance_floor` is the `σw² = 0` (or `m → ∞`) shadow of the
    equality: more frames per group can never push the variance below `σb² / G`. -/
theorem variance_grandMean_ge_between :
    M.σb ^ 2 / M.G ≤ variance M.grandMean M.μ := by
  rw [M.variance_grandMean_eq]
  have : 0 ≤ M.σw ^ 2 / (M.G * M.m) := by positivity
  linarith

/-- **Corollary — the exact gap to the floor.** The amount by which the grand-mean variance
    exceeds the irreducible between-group floor `σb² / G` is exactly the within-group term
    `σw² / (G·m)` — the only part of the variance that frames can touch. -/
theorem variance_grandMean_sub_between_eq :
    variance M.grandMean M.μ - M.σb ^ 2 / M.G = M.σw ^ 2 / (M.G * M.m) := by
  rw [M.variance_grandMean_eq]; ring

end OneWayRandomEffects

/-- **Corollary — frames only shrink the within-group term.** For two one-way random-effects
    models sharing the group count `G`, the between-group variance `σb²` and the within-group
    variance `σw²` but differing in the frame count (`m₁ ≤ m₂`, with `G, m₁ ≥ 1`), the
    larger-`m` model has the smaller grand-mean variance. Increasing `m` lowers *only* the
    `σw² / (G·m)` term toward the irreducible floor `σb² / G`: frames cannot beat the group-count
    floor, the quantitative Check-4 point. -/
theorem variance_grandMean_antitone_in_m {Ω ι : Type*} [MeasurableSpace Ω]
    (M₁ M₂ : OneWayRandomEffects Ω ι)
    (hG : M₁.G = M₂.G) (hσb : M₁.σb = M₂.σb) (hσw : M₁.σw = M₂.σw)
    (hGpos : 0 < M₁.G) (hm₁ : 0 < M₁.m) (hm : M₁.m ≤ M₂.m) :
    variance M₂.grandMean M₂.μ ≤ variance M₁.grandMean M₁.μ := by
  rw [M₁.variance_grandMean_eq, M₂.variance_grandMean_eq, ← hG, ← hσb, ← hσw]
  have hGr : (0 : ℝ) < M₁.G := by exact_mod_cast hGpos
  have hm1r : (0 : ℝ) < M₁.m := by exact_mod_cast hm₁
  have hmr : (M₁.m : ℝ) ≤ M₂.m := by exact_mod_cast hm
  have hnum : (0 : ℝ) ≤ M₁.σw ^ 2 := sq_nonneg _
  gcongr

/-! ## References — Check 4 (one-way random-effects model)

Covers `variance_average`, `OneWayRandomEffects`, `grandMean`, `betweenMean`,
`variance_grandMean_eq`, `variance_floor`, `variance_grandMean_ge_between`,
`variance_grandMean_antitone_in_m`. Augments the inline note above (which records the Mathlib
independent-sum-variance machinery); not repeated here.

Variance of an uncorrelated sum (the engine of `variance_average`).
  • I.-J. Bienaymé, "Considérations à l'appui de la découverte de Laplace sur la loi de probabilité
    dans la méthode des moindres carrés," *C. R. Acad. Sci. Paris* **37** (1853) 309–324. The
    variance of a sum of uncorrelated variables is the sum of variances — exactly what
    `variance_average` instantiates for pairwise-independent terms.

Model and decomposition (the one-way classification and its variance components).
  • S. R. Searle, G. Casella & C. E. McCulloch, *Variance Components* (Wiley, 1992), Ch. 3 (the
    1-way classification). The reference treatment of the model `OneWayRandomEffects` encodes and of
    the `variance_grandMean_eq` decomposition.
  • H. Scheffé, *The Analysis of Variance* (Wiley, 1959), Part II / Ch. 7 (random models).
  • R. A. Fisher, *Statistical Methods for Research Workers* (Oliver & Boyd, 1925). The origin of the
    ANOVA variance decomposition.

Floor / design effect (the load-bearing domain cites for `variance_floor`,
`variance_grandMean_ge_between`, `variance_grandMean_antitone_in_m`).
  • B. R. Moulton, "Random group effects and the precision of regression estimates,"
    *J. Econometrics* **32**(3) (1986) 385–397.
  • B. R. Moulton, "An Illustration of a Pitfall in Estimating the Effects of Aggregate Variables on
    Micro Units," *Rev. Econ. Stat.* **72**(2) (1990) 334–338. The Moulton factor √(1+(m−1)ρ): why
    the group-count floor `σb² / G`, not the frame count, governs precision.
  • L. Kish, *Survey Sampling* (Wiley, 1965), §5.4 (pp. 161–164) and §8.2 (pp. 257–259). The design
    effect and effective sample size n_eff = n / Deff — the survey-sampling form of "effective sample
    size = number of groups."
  • A. C. Cameron & D. L. Miller, "A Practitioner's Guide to Cluster-Robust Inference,"
    *J. Human Resources* **50**(2) (2015) 317–372. Modern synthesis of clustered-variance inference.

Intraclass correlation ρ = σb² / (σb² + σw²) (the quantity the floor isolates).
  • P. E. Shrout & J. L. Fleiss, "Intraclass correlations: Uses in assessing rater reliability,"
    *Psychological Bulletin* **86** (1979) 420–428.
  • J. J. Bartko, "The intraclass correlation coefficient as a measure of reliability,"
    *Psychological Reports* **19** (1966) 3–11. -/

end LewmValidity

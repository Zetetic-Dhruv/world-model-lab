/-
Copyright (c) 2026 Dhruv Gupta. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Dhruv Gupta
-/
import LewmValidity.Vendor.InformationTheory.MutualInformation
import LewmValidity.Vendor.FLT.Chaining
import Mathlib.MeasureTheory.Function.ConditionalExpectation.RadonNikodym

/-!
# Reparametrisation invariance of true mutual information

The ℝ-valued true mutual information `InformationTheory.mutualInformationReal`
(vendored in `LewmValidity.Vendor.InformationTheory.MutualInformation` as the genuine
Kullback–Leibler divergence between the joint law and the product of its
marginals) is invariant under a *measurable reparametrisation of each
coordinate*. Concretely, given measurable equivalences `e₁ : α ≃ᵐ α'`,
`e₂ : β ≃ᵐ β'` and a joint probability measure `P` on `α × β` whose joint law is
absolutely continuous with respect to the product of its marginals (so that the
mutual information is a genuine KL divergence rather than the `0` convention),

    mutualInformationReal (P.map (e₁.prodCongr e₂)) = mutualInformationReal P.

This is the measure-theoretic form of the *data-processing / reparametrisation
invariance* of mutual information specialised to invertible (bijective
measurable) channels: pushing a joint distribution through a measurable
isomorphism of each marginal space leaves the information shared between the two
coordinates unchanged.

## Main statement

* `InformationTheory.mutualInformationReal_map_prodCongr`: the invariance theorem
  above.

## Implementation notes

The proof is assembled from three reusable facts, the first of which is proved
here as a private lemma because Mathlib exposes pushforward of
Radon–Nikodym derivatives only for general measurable *embeddings* / via
conditional expectations, not packaged as a KL-invariance statement:

* `klDivReal_map_measurableEquiv` (private): the ℝ-valued KL divergence is
  invariant under pushing both arguments through a measurable equivalence. The
  change of variables is `MeasureTheory.integral_map`; the equality of
  integrands is `MeasurableEmbedding.rnDeriv_map` (transported from `=ᵐ[ν]` to
  `=ᵐ[μ]` along `μ ≪ ν`).
* Marginals commute with `MeasurableEquiv.prodCongr`, via `Measure.map_map` and
  the (definitional) identities `Prod.fst ∘ Prod.map e₁ e₂ = e₁ ∘ Prod.fst` and
  the `snd` analogue.
* The product of pushforwards is the pushforward of the product, via
  `MeasureTheory.Measure.map_prod_map`.

## References

* Cover & Thomas, *Elements of Information Theory*, 2nd ed. (Wiley, 2006),
  Thm. 2.8.1 (invariance of mutual information under invertible transformations)
  and §8.6 (differential entropy / continuous MI and the data-processing
  inequality).
* Kinney & Atwal, *Equitability, mutual information, and the maximal information
  coefficient*, PNAS 111 (2014) 3354–3359 (MI is invariant under invertible
  reparametrisation of each variable — the equitability/self-equitability
  property).
* Czyż, Grabowski, Vogt, Beerenwinkel & Marx, *Beyond Normal: On the Evaluation
  of Mutual Information Estimators*, NeurIPS 36 (2023), **Theorem 2**
  (`I(X;Y) = I(f(X); g(Y))` for continuous injective `f`, `g` — the population
  statement that this lemma proves in measure-theoretic form for measurable
  isomorphisms).
* Liese & Vajda, *On divergences and informations in statistics and information
  theory*, IEEE Trans. Inform. Theory 52 (2006) 4394–4412 (invariance of
  `f`-divergences, hence KL, under measurable bijections — the engine behind the
  private `klDivReal_map_measurableEquiv` lemma).
* Polyanskiy & Wu, *Information Theory: From Coding to Learning* (Cambridge,
  2024), §3 (KL divergence and its invariance under measurable bijections).
-/

noncomputable section

open MeasureTheory InformationTheory

namespace InformationTheory

variable {α α' β β' : Type*}
  [MeasurableSpace α] [MeasurableSpace α']
  [MeasurableSpace β] [MeasurableSpace β']

/-- **ℝ-valued KL divergence is invariant under a measurable equivalence.**
Pushing both measures through a measurable isomorphism `e : γ ≃ᵐ γ'` leaves the
real-valued Kullback–Leibler divergence unchanged. The absolute-continuity
hypothesis `μ ≪ ν` is what makes both sides genuine KL integrals (rather than
the `0` convention) and is what transports the Radon–Nikodym derivative equality
from `ν`-a.e. to `μ`-a.e.

## References
Liese & Vajda, *On divergences and informations in statistics and information
theory*, IEEE Trans. Inform. Theory 52 (2006) 4394–4412 (invariance of
`f`-divergences, and KL in particular, under measurable bijections); Cover &
Thomas, *Elements of Information Theory*, 2nd ed. (Wiley, 2006), §8.6 (relative
entropy / KL in the continuous setting). -/
private lemma klDivReal_map_measurableEquiv
    {γ γ' : Type*} [MeasurableSpace γ] [MeasurableSpace γ']
    (e : γ ≃ᵐ γ') (μ ν : Measure γ) [SigmaFinite μ] [SigmaFinite ν]
    (hμν : μ ≪ ν) :
    klDivReal (μ.map e) (ν.map e) = klDivReal μ ν := by
  have hμν' : (μ.map e) ≪ (ν.map e) := hμν.map e.measurable
  -- Both branches take the genuine-KL form.
  rw [klDivReal, if_pos hμν', klDivReal, if_pos hμν]
  -- Change of variables on the pushforward: `∫ g ∂(μ.map e) = ∫ g ∘ e ∂μ`.
  rw [integral_map e.measurable.aemeasurable]
  · -- Now compare the two integrals over `μ`, integrand by integrand.
    refine integral_congr_ae ?_
    -- `rnDeriv_map` for the measurable embedding `e`, transported `ν`-a.e. → `μ`-a.e.
    have h_rn : (fun x ↦ (μ.map e).rnDeriv (ν.map e) (e x)) =ᵐ[μ] μ.rnDeriv ν :=
      hμν.ae_eq (e.measurableEmbedding.rnDeriv_map μ ν)
    filter_upwards [h_rn] with x hx
    rw [hx]
  · -- AEStronglyMeasurable side condition for `integral_map`: the integrand
    -- `fun y ↦ Real.log ((μ.map e).rnDeriv (ν.map e) y).toReal` over `μ.map e`.
    exact (Real.measurable_log.comp
      (Measure.measurable_rnDeriv _ _).ennreal_toReal).aestronglyMeasurable

/-- The first marginal commutes with `MeasurableEquiv.prodCongr`:
`(P.map (e₁.prodCongr e₂)).map Prod.fst = (P.map Prod.fst).map e₁`. -/
private lemma map_fst_map_prodCongr
    (e₁ : α ≃ᵐ α') (e₂ : β ≃ᵐ β') (P : Measure (α × β)) :
    (P.map (e₁.prodCongr e₂)).map Prod.fst = (P.map Prod.fst).map e₁ := by
  rw [Measure.map_map measurable_fst (e₁.prodCongr e₂).measurable,
    Measure.map_map e₁.measurable measurable_fst]
  rfl

/-- The second marginal commutes with `MeasurableEquiv.prodCongr`:
`(P.map (e₁.prodCongr e₂)).map Prod.snd = (P.map Prod.snd).map e₂`. -/
private lemma map_snd_map_prodCongr
    (e₁ : α ≃ᵐ α') (e₂ : β ≃ᵐ β') (P : Measure (α × β)) :
    (P.map (e₁.prodCongr e₂)).map Prod.snd = (P.map Prod.snd).map e₂ := by
  rw [Measure.map_map measurable_snd (e₁.prodCongr e₂).measurable,
    Measure.map_map e₂.measurable measurable_snd]
  rfl

/-- **Reparametrisation invariance of true mutual information.**
For measurable equivalences `e₁ : α ≃ᵐ α'`, `e₂ : β ≃ᵐ β'` and a joint
probability measure `P` on `α × β` whose joint law is absolutely continuous with
respect to the product of its marginals, pushing `P` through the coordinatewise
measurable isomorphism `e₁.prodCongr e₂` leaves the mutual information unchanged.

The hypothesis `hP` is exactly the condition under which `mutualInformationReal`
is the genuine KL divergence (otherwise both sides collapse to the `0`
convention and the statement is vacuous); it also feeds the Radon–Nikodym
transport inside `klDivReal_map_measurableEquiv`.

## References
Cover & Thomas, *Elements of Information Theory*, 2nd ed. (Wiley, 2006),
Thm. 2.8.1 (MI invariance under invertible transformations) and §8.6; the
data-processing inequality for KL divergence under measurable bijections. Kinney
& Atwal, *Equitability, mutual information, and the maximal information
coefficient*, PNAS 111 (2014) 3354–3359 (invariance of MI under invertible
reparametrisation of each coordinate). Czyż, Grabowski, Vogt, Beerenwinkel &
Marx, *Beyond Normal: On the Evaluation of Mutual Information Estimators*,
NeurIPS 36 (2023), **Theorem 2** (`I(X;Y) = I(f(X); g(Y))` for continuous
injective `f`, `g` — the population statement this lemma establishes measure-
theoretically for measurable isomorphisms). -/
theorem mutualInformationReal_map_prodCongr
    (e₁ : α ≃ᵐ α') (e₂ : β ≃ᵐ β') (P : Measure (α × β))
    [IsProbabilityMeasure P]
    (hP : P ≪ (P.map Prod.fst).prod (P.map Prod.snd)) :
    mutualInformationReal (P.map (e₁.prodCongr e₂)) = mutualInformationReal P := by
  -- Abbreviate the coordinatewise reparametrisation.
  set e := e₁.prodCongr e₂ with he
  -- Unfold MI on the pushforward to KL of (pushforward) vs (product of its marginals).
  rw [mutualInformationReal]
  -- Rewrite the two marginals of `P.map e` as pushforwards of `P`'s marginals.
  rw [he, map_fst_map_prodCongr, map_snd_map_prodCongr]
  -- Product of pushforwards = pushforward of the product, through `Prod.map e₁ e₂ = e`.
  rw [Measure.map_prod_map (P.map Prod.fst) (P.map Prod.snd) e₁.measurable e₂.measurable]
  -- The two pushforward maps `Prod.map e₁ e₂` and `⇑(e₁.prodCongr e₂)` are defeq.
  show klDivReal (P.map e) (((P.map Prod.fst).prod (P.map Prod.snd)).map e) = _
  -- Invariance of KL under the measurable equivalence `e`, with AC hypothesis `hP`.
  rw [klDivReal_map_measurableEquiv e P ((P.map Prod.fst).prod (P.map Prod.snd)) hP]
  rfl

end InformationTheory

end

/-!
# Reparametrisation non-invariance of the k-NN MI-estimator core

The invariance theorem above shows the *true* mutual information `mutualInformationReal`
is invariant under a coordinatewise measurable reparametrisation `e₁.prodCongr e₂`. This
block is the estimator-side counterpart: the **k-nearest-neighbour neighbourhood-radius
statistic** that the Kozachenko–Leonenko / Kraskov–Stögbauer–Grassberger (KSG)
mutual-information *estimator* is built from is **not** invariant under a similarity
reparametrisation — it scales linearly with the similarity ratio `λ` and is therefore
unbounded as `λ → ∞`. Taken together, these say the k-NN information-estimator core is
not a function of the information content: the true MI is pinned while the estimator drifts.

## Scope

The statistic formalised here, `ksgRadiusStat`, is the k = 1 **neighbourhood-radius
core** `∑ᵢ dist xᵢ (nearest neighbour of xᵢ)` — the `Σ log εᵢ` / radius term out of which
the Kozachenko–Leonenko entropy estimate and hence the KSG MI estimate are assembled. It
is **not** the full count-cancelling KSG number. Under a *uniform* (isotropic) scaling the
full KSG count term is in fact scale-stable — the digamma-of-marginal-counts corrections
cancel — so the complete KSG MI estimate is *not* moved by uniform scaling; that failure
requires an *anisotropic* reparametrisation and remains an empirical phenomenon outside
the scope formalised here. What is formalised, exactly and generally, is that the
**geometric radius core** — the part that carries the `⟨log ε⟩` entropy contribution — is
metric-dependent. That is the honest, provable kernel of "the k-NN estimator is
reparametrisation-dependent".

## The reparametrisation

On `ℝ × ℝ` (carrying Mathlib's **sup** product metric
`dist (a,b) (c,d) = max (dist a c) (dist b d)`) the map is the **uniform scaling**
`e_λ : (x, y) ↦ (λ • x, λ • y) = λ • (x, y)` for `λ > 0`, built as `e₁.prodCongr e₁` with
`e₁ : ℝ ≃ᵐ ℝ` left-multiplication by `λ` (`Homeomorph.mulLeft₀ λ _ |>.toMeasurableEquiv`).
Because the sup metric is absolutely homogeneous, `e_λ` is a genuine **λ-similarity**:
`dist (e_λ a) (e_λ b) = λ * dist a b` (this is `dist_smul₀` with `‖λ‖ = λ`). And because
`e_λ` is *literally* `e₁.prodCongr e₁`, the true-MI invariance theorem applies to it
verbatim — the same transformation that fixes the true MI moves the estimator core.

## Main statements

* `ksgRadiusStat`: the k = 1 neighbourhood-radius statistic.
* `dist_nearestInFinset_map_of_similarity`: the crux radius-equality lemma — under an
  injective `λ`-similarity embedding, the nearest-neighbour radius of the image point in
  the image set is `λ` times the original radius.
* `ksgRadiusStat_map_smul`: the scaling law
  `ksgRadiusStat (S.map e_λ) _ = λ * ksgRadiusStat S _`.
* `ksgRadiusStat_not_invariant`: non-invariance and unboundedness on a concrete witness set.
* `ksg_estimator_not_information_invariant`: the combined witness — true MI fixed (by the
  invariance theorem) while the estimator core provably drifts.

## References

* Kozachenko & Leonenko, *Sample estimate of the entropy of a random vector*,
  Probl. Inf. Transm. 23 (1987) 95–101 (the `⟨log ε⟩` k-NN entropy estimator out
  of which the radius core here is taken).
* Kraskov, Stögbauer & Grassberger, *Estimating mutual information*, Phys. Rev. E
  69 (2004) 066138, **erratum Phys. Rev. E 83 (2011) 019903** (KSG estimator
  built on k-NN radii; §II for the count-cancellation under uniform scaling).
* Gao, Oh & Viswanath, *Demystifying fixed k-nearest neighbor information
  estimators*, IEEE Trans. Inform. Theory 64 (2018) 5629–5661 (rigorous
  consistency/bias analysis of fixed-k k-NN estimators).
* Cover & Thomas, *Elements of Information Theory*, 2nd ed. (Wiley, 2006),
  **Thm. 8.6.4** (`h(AX) = h(X) + log|det A|` — the differential-entropy scaling
  law that the radius core's `λ`-scaling mirrors).
* On the reparametrisation/metric dependence of k-NN information estimators:
  Gao, Ver Steeg & Galstyan, *Efficient estimation of mutual information for
  strongly dependent variables*, AISTATS 2015 (PMLR 38:277–286);
  Marin-Franch & Foster, *Estimating information from image colors*, IEEE TPAMI
  35 (2013) 78–91.
-/

noncomputable section

open MeasureTheory Metric

namespace LewmValidity.EstimatorGeometry

/-! ### The k = 1 neighbourhood-radius statistic -/

/-- `2 ≤ S.card` makes every leave-one-out neighbourhood nonempty: deleting one point from a
set with at least two points leaves a nonempty set. (Used to feed `nearestInFinset` on
`S.erase x` inside `ksgRadiusStat`.) The membership hypothesis `_hx : x ∈ S` is not needed
for the conclusion (`Nontrivial.erase_nonempty` holds for any `x`) but is kept so the lemma
applies directly to attached elements `p` with `p.2 : ↑p ∈ S`. -/
lemma erase_nonempty_of_two_le_card {A : Type*} [DecidableEq A] {S : Finset A}
    (hS : 2 ≤ S.card) {x : A} (_hx : x ∈ S) : (S.erase x).Nonempty :=
  (Finset.one_lt_card_iff_nontrivial.mp hS).erase_nonempty

/-- `nearestInFinset` depends only on the set and the query point, not on the nonemptiness
proof: for equal finsets the nearest-neighbour choice agrees. (Used to align leave-one-out
neighbourhoods after `Finset.map_erase` without tripping the dependent-proof motive check.) -/
lemma nearestInFinset_congr {A : Type*} [PseudoMetricSpace A] {t₁ t₂ : Finset A}
    (h : t₁ = t₂) (ht₁ : t₁.Nonempty) (ht₂ : t₂.Nonempty) (x : A) :
    nearestInFinset t₁ ht₁ x = nearestInFinset t₂ ht₂ x := by
  subst h; rfl

/-- **The k = 1 KSG neighbourhood-radius statistic.**
For a finite point cloud `S` with at least two points, this is the sum over points of the
distance from each point to its nearest *other* point (its 1-nearest neighbour in the
leave-one-out set `S.erase x`):

    ksgRadiusStat S hS = ∑ p ∈ S.attach, dist p (nearestInFinset (S.erase p) _ p).

This is the **Kozachenko–Leonenko / KSG k-NN entropy-estimator core** — the radius term
`Σᵢ εᵢ` (with `εᵢ` the 1-NN distance of point `i`) whose logarithm `⟨log ε⟩` supplies the
entropy contribution of the KSG mutual-information estimate. It is a purely *geometric*
functional of the sample and the metric; it carries no probability content of its own.
The witness below shows this geometric core is metric-dependent (it is not a function of
the information content), even though the true MI it is meant to estimate is not.

## References
Kozachenko & Leonenko, *Sample estimate of the entropy of a random vector*, Probl. Inf.
Transm. 23 (1987) 95–101 (the `⟨log ε⟩` k-NN entropy estimator); Kraskov, Stögbauer &
Grassberger, *Estimating mutual information*, Phys. Rev. E 69 (2004) 066138, erratum Phys.
Rev. E 83 (2011) 019903 (the KSG MI estimator built on these radii); Gao, Oh & Viswanath,
*Demystifying fixed k-nearest neighbor information estimators*, IEEE Trans. Inform. Theory
64 (2018) 5629–5661; Gao, Ver Steeg & Galstyan, AISTATS 2015 (PMLR 38:277–286). -/
def ksgRadiusStat {A : Type*} [PseudoMetricSpace A] [DecidableEq A]
    (S : Finset A) (hS : 2 ≤ S.card) : ℝ :=
  ∑ p ∈ S.attach, dist (p : A)
    (nearestInFinset (S.erase (p : A)) (erase_nonempty_of_two_le_card hS p.2) (p : A))

/-! ### The crux radius-equality lemma -/

/-- **Crux radius-equality lemma.**
Let `embE : A ↪ A` be an injective **λ-similarity** of a pseudometric space (its carrier
satisfies `dist (embE a) (embE b) = λ * dist a b` for `λ > 0`). Then for any finite `S` and
any point `x`, the nearest-neighbour radius of the *image* point `embE x` inside the *image*
set `S.map embE` is exactly `λ` times the nearest-neighbour radius of `x` in `S`:

    dist (embE x) (nearestInFinset (S.map embE) _ (embE x))
      = λ * dist x (nearestInFinset S hS x).

This is the general step (not a hand-computed single configuration): a similarity scales
*every* pairwise distance by `λ`, hence scales the minimum over the cloud by `λ`. The two
inequalities:
* `≤`: `embE (NN_S x) ∈ S.map embE`, so the LHS `≤ dist (embE x) (embE (NN_S x))
  = λ * dist x (NN_S x)`.
* `≥`: every member of `S.map embE` is `embE z` for some `z ∈ S`; the realised nearest one
  gives LHS `= λ * dist x z ≥ λ * dist x (NN_S x)` since `dist x (NN_S x)` is the minimum.
Antisymmetry closes it.

## References
Cover & Thomas, *Elements of Information Theory*, 2nd ed. (Wiley, 2006), Thm. 8.6.4
(`h(AX) = h(X) + log|det A|`): the population analogue of this radius `λ`-scaling at the
entropy level. Kraskov, Stögbauer & Grassberger, Phys. Rev. E 69 (2004) 066138 (erratum
83 (2011) 019903) and Kozachenko & Leonenko, Probl. Inf. Transm. 23 (1987) 95–101, for the
k-NN radii whose minimum this lemma rescales; Gao, Oh & Viswanath, IEEE Trans. Inform.
Theory 64 (2018) 5629–5661. -/
lemma dist_nearestInFinset_map_of_similarity {A : Type*} [PseudoMetricSpace A]
    {c : ℝ} (hc : 0 < c) (embE : A ↪ A)
    (hsim : ∀ a b, dist (embE a) (embE b) = c * dist a b)
    (S : Finset A) (hS : S.Nonempty) (hSe : (S.map embE).Nonempty) (x : A) :
    dist (embE x) (nearestInFinset (S.map embE) hSe (embE x))
      = c * dist x (nearestInFinset S hS x) := by
  -- Abbreviate the two nearest neighbours.
  set nnS := nearestInFinset S hS x with hnnS
  set nnSe := nearestInFinset (S.map embE) hSe (embE x) with hnnSe
  refine le_antisymm ?_ ?_
  · -- `≤`: compare against the image of the original nearest neighbour, which lies in `S.map embE`.
    have hmem : embE nnS ∈ S.map embE :=
      Finset.mem_map_of_mem embE (by rw [hnnS]; exact nearestInFinset_mem S hS x)
    calc dist (embE x) nnSe
        ≤ dist (embE x) (embE nnS) :=
          dist_nearestInFinset_le (S.map embE) hSe (embE x) (embE nnS) hmem
      _ = c * dist x nnS := hsim x nnS
  · -- `≥`: the realised nearest neighbour `nnSe ∈ S.map embE` is `embE z` for some `z ∈ S`.
    have hnnSe_mem : nnSe ∈ S.map embE := by rw [hnnSe]; exact nearestInFinset_mem _ hSe _
    rw [Finset.mem_map] at hnnSe_mem
    obtain ⟨z, hzS, hz⟩ := hnnSe_mem
    -- Rewrite the LHS distance via the similarity, then bound `dist x z` below by the minimum.
    rw [← hz, hsim x z]
    exact mul_le_mul_of_nonneg_left
      (dist_nearestInFinset_le S hS x z hzS) hc.le

/-! ### The scaling law -/

/-- The uniform-scaling measurable equivalence `e_λ : ℝ × ℝ ≃ᵐ ℝ × ℝ`,
`(x, y) ↦ (λ • x, λ • y) = λ • (x, y)`, for `λ ≠ 0`. Built as `e₁.prodCongr e₁` with `e₁`
left-multiplication by `λ`, so that **the true-MI invariance theorem applies to it
verbatim**. -/
def scaleEquiv {c : ℝ} (hc : c ≠ 0) : (ℝ × ℝ) ≃ᵐ (ℝ × ℝ) :=
  ((Homeomorph.mulLeft₀ c hc).toMeasurableEquiv).prodCongr
    ((Homeomorph.mulLeft₀ c hc).toMeasurableEquiv)

/-- The scaling equivalence acts as the diagonal scalar multiplication `c • ·` on `ℝ × ℝ`. -/
@[simp] lemma scaleEquiv_apply {c : ℝ} (hc : c ≠ 0) (p : ℝ × ℝ) :
    scaleEquiv hc p = c • p := rfl

/-- The measurable embedding underlying the scaling equivalence (carrier `c • ·`), used to
push a `Finset (ℝ × ℝ)` forward. Its coercion is defeq to `⇑(scaleEquiv hc)`. -/
def scaleEmb {c : ℝ} (hc : c ≠ 0) : (ℝ × ℝ) ↪ (ℝ × ℝ) :=
  (scaleEquiv hc).toEquiv.toEmbedding

@[simp] lemma scaleEmb_apply {c : ℝ} (hc : c ≠ 0) (p : ℝ × ℝ) :
    scaleEmb hc p = c • p := rfl

/-- The scaling equivalence is a genuine **λ-similarity** under the sup product metric: for
`c > 0`, `dist (e_c a) (e_c b) = c * dist a b`. This is `dist_smul₀` (absolute homogeneity
of the norm) together with `‖c‖ = |c| = c`. (The scaling factor is the mathematical `λ`;
the Lean identifier is `c` because `λ` is reserved syntax.) -/
lemma dist_scaleEmb {c : ℝ} (hcpos : 0 < c) (a b : ℝ × ℝ) :
    dist (scaleEmb hcpos.ne' a) (scaleEmb hcpos.ne' b) = c * dist a b := by
  simp only [scaleEmb_apply, dist_smul₀, Real.norm_eq_abs, abs_of_pos hcpos]

/-- A scaled point cloud `S.map (scaleEmb _)` still has at least two points (the scaling
embedding is injective, so `Finset.card_map` preserves cardinality). -/
lemma two_le_card_map_scaleEmb {c : ℝ} (hc : c ≠ 0) {S : Finset (ℝ × ℝ)}
    (hS : 2 ≤ S.card) : 2 ≤ (S.map (scaleEmb hc)).card := by
  rwa [Finset.card_map]

/-- **The scaling law.** The k-NN radius statistic of a uniformly scaled point cloud is `λ`
times the original:

    ksgRadiusStat (S.map (scaleEmb _)) _ = λ * ksgRadiusStat S hS    (λ > 0).

This is the general consequence of the crux similarity lemma, applied term by term. The
leave-one-out neighbourhoods are aligned by `Finset.map_erase`
(`(S.erase x).map f = (S.map f).erase (f x)`), each term is scaled by the crux lemma, and
the sum is pulled out by `Finset.mul_sum`. Hence the estimator core is **not** invariant:
the same data, re-expressed in a rescaled coordinate, reports a different value.

## References
Kraskov, Stögbauer & Grassberger, *Estimating mutual information*, Phys. Rev. E 69 (2004)
066138, erratum Phys. Rev. E 83 (2011) 019903; Kozachenko & Leonenko, Probl. Inf. Transm.
23 (1987) 95–101 (the k-NN radii this law rescales); Gao, Oh & Viswanath, IEEE Trans.
Inform. Theory 64 (2018) 5629–5661. Cover & Thomas, *Elements of Information Theory*, 2nd
ed. (Wiley, 2006), Thm. 8.6.4 (`h(AX) = h(X) + log|det A|`): for the uniform scaling
`A = c·I`, the `log|det A| = 2 log c` offset is the entropy-level shadow of the radius core's
multiplication by `c` here.

## Scope note (folklore, partial)
A *uniform* (isotropic) scaling leaves the *complete* KSG MI number stable because the
digamma-of-marginal-count corrections cancel (Kraskov et al. 2004, §II); only an
*anisotropic* reparametrisation moves it, and that remains an empirical phenomenon rather
than a settled theorem. The "max-norm makes KSG rescaling-robust" claim is folklore of this
partial kind — see *Towards Robust Scale-Invariant Mutual Information Estimators*, TMLR
(2024), together with Kraskov et al. 2004 §II. What is *proven* here is only that the
geometric radius core (the `⟨log ε⟩` term), not the full count-cancelling estimate, scales
by `c`. -/
theorem ksgRadiusStat_map_smul {c : ℝ} (hcpos : 0 < c) {S : Finset (ℝ × ℝ)}
    (hS : 2 ≤ S.card) :
    ksgRadiusStat (S.map (scaleEmb hcpos.ne')) (two_le_card_map_scaleEmb hcpos.ne' hS)
      = c * ksgRadiusStat S hS := by
  classical
  -- `e` is the underlying measurable equivalence; `emb` its embedding (defeq carriers).
  set e := scaleEquiv hcpos.ne' with he
  set emb := scaleEmb hcpos.ne' with hemb
  -- `emb x = e x` definitionally, so `e.symm` inverts `emb`.
  have hembe : ∀ x, emb x = e x := fun _ => rfl
  rw [ksgRadiusStat, ksgRadiusStat, Finset.mul_sum]
  -- Reindex the scaled attach-sum by `S.attach` through the bijection `p ↦ emb p` (inverse `e.symm`).
  refine Finset.sum_bij'
    (i := fun (q : {x // x ∈ S.map emb}) _ =>
      (⟨e.symm (q : ℝ × ℝ), by
        have hq : (q : ℝ × ℝ) ∈ S.map emb := q.2
        rw [Finset.mem_map] at hq
        obtain ⟨a, ha, hae⟩ := hq
        have : e.symm (q : ℝ × ℝ) = a := by
          rw [← hae, hembe]; exact e.symm_apply_apply a
        rw [this]; exact ha⟩ : {x // x ∈ S}))
    (j := fun (p : {x // x ∈ S}) _ =>
      (⟨emb (p : ℝ × ℝ), Finset.mem_map_of_mem emb p.2⟩ : {x // x ∈ S.map emb}))
    ?_ ?_ ?_ ?_ ?_
  · -- `i` lands in `S.attach`
    intro q _; exact Finset.mem_attach _ _
  · -- `j` lands in `(S.map emb).attach`
    intro p _; exact Finset.mem_attach _ _
  · -- left inverse: `j (i q) = q`
    intro q _
    apply Subtype.ext
    show emb (e.symm (q : ℝ × ℝ)) = (q : ℝ × ℝ)
    rw [hembe]; exact e.apply_symm_apply (q : ℝ × ℝ)
  · -- right inverse: `i (j p) = p`
    intro p _
    apply Subtype.ext
    show e.symm (emb (p : ℝ × ℝ)) = (p : ℝ × ℝ)
    rw [hembe]; exact e.symm_apply_apply (p : ℝ × ℝ)
  · -- the per-term scaling identity, via the crux similarity lemma.
    -- `q` ranges over the scaled cloud; its preimage `x := e.symm q.1 ∈ S` satisfies `emb x = q.1`.
    rintro q -
    set x : ℝ × ℝ := e.symm (q : ℝ × ℝ) with hx
    -- `q.1 = emb x` (apply-symm), the key identification of the scaled point with `emb` of preimage.
    have hqx : (q : ℝ × ℝ) = emb x := by rw [hx, hembe]; exact (e.apply_symm_apply _).symm
    -- `x ∈ S` (the membership proof carried by `i`).
    have hxS : x ∈ S := by
      have hq : (q : ℝ × ℝ) ∈ S.map emb := q.2
      rw [Finset.mem_map] at hq
      obtain ⟨a, ha, hae⟩ := hq
      have : x = a := by rw [hx, ← hae, hembe]; exact e.symm_apply_apply a
      rw [this]; exact ha
    -- Nonemptiness of the original leave-one-out neighbourhood and (hence) its image.
    have hSn : (S.erase x).Nonempty := erase_nonempty_of_two_le_card hS hxS
    have hSen : ((S.erase x).map emb).Nonempty := hSn.map
    -- The erased neighbourhoods coincide: `(S.map emb).erase q.1 = (S.erase x).map emb`.
    have hset : (S.map emb).erase (q : ℝ × ℝ) = (S.erase x).map emb := by
      rw [hqx, Finset.map_erase]
    -- The goal: scaled-cloud summand at `q.1` equals `c *` original summand at `x = i q`.
    show dist (q : ℝ × ℝ)
        (nearestInFinset ((S.map emb).erase (q : ℝ × ℝ)) _ (q : ℝ × ℝ))
      = c * dist x (nearestInFinset (S.erase x) _ x)
    -- Bridge to the crux lemma's LHS by `congr` (the two nearest-neighbour calls are over the
    -- same set and same point `emb x = q.1`; the nonemptiness proofs are irrelevant).
    have hbridge : dist (q : ℝ × ℝ)
          (nearestInFinset ((S.map emb).erase (q : ℝ × ℝ))
            (erase_nonempty_of_two_le_card (two_le_card_map_scaleEmb hcpos.ne' hS) q.2)
            (q : ℝ × ℝ))
        = dist (emb x) (nearestInFinset ((S.erase x).map emb) hSen (emb x)) := by
      rw [← hqx]
      exact congrArg _ (nearestInFinset_congr hset _ hSen (q : ℝ × ℝ))
    rw [hbridge]
    exact dist_nearestInFinset_map_of_similarity hcpos emb (dist_scaleEmb hcpos)
      (S.erase x) hSn hSen x

/-! ### Non-invariance, strict positivity, and unboundedness -/

/-- **Strict positivity of the radius core.** In a genuine metric space, the k-NN
neighbourhood-radius statistic of any cloud with at least two
points is strictly positive: every point's nearest *other* point is a distinct point, hence
at strictly positive distance (`dist_pos`), and a sum of positives over the nonempty
`S.attach` is positive (`Finset.sum_pos`). This is the engine of non-invariance: a strictly
positive quantity that scales by `c` genuinely *moves* when `c ≠ 1`. -/
theorem ksgRadiusStat_pos {A : Type*} [MetricSpace A] [DecidableEq A]
    (S : Finset A) (hS : 2 ≤ S.card) : 0 < ksgRadiusStat S hS := by
  rw [ksgRadiusStat]
  apply Finset.sum_pos
  · intro p _
    rw [dist_pos]
    -- the nearest neighbour lives in `S.erase p`, so it differs from `p`.
    have hmem := nearestInFinset_mem (S.erase (p : A))
      (erase_nonempty_of_two_le_card hS p.2) (p : A)
    exact fun heq => (Finset.ne_of_mem_erase hmem) heq.symm
  · rw [Finset.attach_nonempty_iff]
    exact Finset.card_pos.mp (by omega)

/-- The concrete two-point witness cloud `{(0,0), (1,0)} ⊆ ℝ × ℝ`. The two points are at sup
distance `1`, and `2 ≤ card`. -/
def witnessCloud : Finset (ℝ × ℝ) := {((0 : ℝ), (0 : ℝ)), ((1 : ℝ), (0 : ℝ))}

lemma witnessCloud_card : witnessCloud.card = 2 := by
  rw [witnessCloud, Finset.card_pair]
  simp

lemma two_le_witnessCloud_card : 2 ≤ witnessCloud.card := by
  rw [witnessCloud_card]

/-- **Non-invariance and unboundedness of the radius core.** On the concrete two-point cloud
`witnessCloud`:

* **(strict positivity)** `0 < ksgRadiusStat witnessCloud _`;
* **(non-invariance)** for every similarity ratio `c > 0` with `c ≠ 1`, the rescaled cloud's
  statistic differs from the original — the same data in a rescaled coordinate reports a
  different value, so the estimator core is *not* a reparametrisation invariant;
* **(unboundedness)** the rescaled statistic `c * (…)` exceeds *any* bound `C` for suitable
  `c`, so as `c → ∞` the estimator core diverges while the underlying point configuration is
  the same up to similarity.

These follow from the scaling law `ksgRadiusStat_map_smul` (the statistic scales by `c`) and
the strict positivity `ksgRadiusStat_pos`.

## References (motivation)
The concrete two-point witness and its non-invariance/unboundedness conclusion are **original
to this formalisation**; the following are cited as **motivation only**, not as the source of
the result. McAllester & Stratos, *Formal Limitations on the Measurement of Mutual Information*,
AISTATS 2020 (PMLR 108:875–884); Song & Ermon, *Understanding the Limitations of Variational
Mutual Information Estimators*, ICLR 2020 (estimators violate the data-processing /
self-consistency invariance the true MI obeys); Czyż, Grabowski, Vogt, Beerenwinkel & Marx,
*Beyond Normal*, NeurIPS 36 (2023) (MI invariant under reparametrisation while estimates are
not); Poole, Ozair, van den Oord, Alemi & Tucker, *On Variational Bounds of Mutual
Information*, ICML 2019 (PMLR 97:5171–5180); Paninski, *Estimation of Entropy and Mutual
Information*, Neural Comput. 15 (2003) 1191–1253. -/
theorem ksgRadiusStat_not_invariant :
    (0 < ksgRadiusStat witnessCloud two_le_witnessCloud_card) ∧
    (∀ c : ℝ, (hc : 0 < c) → c ≠ 1 →
      ksgRadiusStat (witnessCloud.map (scaleEmb hc.ne'))
          (two_le_card_map_scaleEmb hc.ne' two_le_witnessCloud_card)
        ≠ ksgRadiusStat witnessCloud two_le_witnessCloud_card) ∧
    (∀ C : ℝ, ∃ c : ℝ, ∃ hc : 0 < c,
      C < ksgRadiusStat (witnessCloud.map (scaleEmb hc.ne'))
          (two_le_card_map_scaleEmb hc.ne' two_le_witnessCloud_card)) := by
  -- The base value `r := ksgRadiusStat witnessCloud _` is strictly positive.
  have hpos : 0 < ksgRadiusStat witnessCloud two_le_witnessCloud_card :=
    ksgRadiusStat_pos witnessCloud two_le_witnessCloud_card
  set r := ksgRadiusStat witnessCloud two_le_witnessCloud_card with hr
  refine ⟨hpos, ?_, ?_⟩
  · -- Non-invariance: scaling by `c ≠ 1` changes a strictly positive value.
    intro c hc hc1
    rw [ksgRadiusStat_map_smul hc two_le_witnessCloud_card, ← hr]
    -- `c * r = r ↔ c = 1` (since `r ≠ 0`); contrapositive gives `c * r ≠ r`.
    intro hcontra
    apply hc1
    have hcr : c * r = 1 * r := by rw [one_mul]; exact hcontra
    exact mul_right_cancel₀ hpos.ne' hcr
  · -- Unboundedness: choose `c = (max C 0 + 1) / r > 0`; then `c * r = max C 0 + 1 > C`.
    intro C
    refine ⟨(max C 0 + 1) / r, by positivity, ?_⟩
    rw [ksgRadiusStat_map_smul (by positivity) two_le_witnessCloud_card, ← hr,
      div_mul_cancel₀ _ hpos.ne']
    have h1 : C ≤ max C 0 := le_max_left _ _
    linarith

/-! ### The witness: truth fixed, estimator drifts -/

/-- **The k-NN estimator core is not a function of the information content.**

The single uniform-scaling reparametrisation `scaleEquiv c` — a `MeasurableEquiv` of the form
`e₁.prodCongr e₁`, hence *information-preserving* — does two incompatible things at once:

* **truth is fixed.** By the invariance theorem (`mutualInformationReal_map_prodCongr`) the
  *true* mutual information of **every** absolutely-continuous joint law on `ℝ × ℝ` is left
  unchanged by the reparametrisation.
* **the estimator drifts.** By `ksgRadiusStat_not_invariant` the KSG k-NN radius core strictly
  changes on the witness cloud whenever `c ≠ 1` (and is unbounded as `c → ∞`).

Therefore no function of the true mutual information can agree with the KSG radius core: the
estimator is sensitive to a coordinate change the information itself cannot see. This is the
estimator/truth divide, now machine-checked — a boundary that elsewhere is merely recorded is
here *proven*.

## References (motivation)
The combined truth-fixed/estimator-drifts witness is **original to this formalisation**. The
following are cited as **motivation only** (they frame the estimator/truth divide; they are not
the source of this theorem): McAllester & Stratos, *Formal Limitations on the Measurement of
Mutual Information*, AISTATS 2020 (PMLR 108:875–884); Song & Ermon, *Understanding the
Limitations of Variational Mutual Information Estimators*, ICLR 2020 (the estimator-side
analogue: estimators violate the data-processing / self-consistency invariance that the
true MI obeys); Czyż, Grabowski, Vogt, Beerenwinkel & Marx, *Beyond Normal*, NeurIPS 36 (2023)
(MI invariant under reparametrisation yet estimates are not); Poole, Ozair, van den Oord, Alemi
& Tucker, *On Variational Bounds of Mutual Information*, ICML 2019 (PMLR 97:5171–5180);
Paninski, *Estimation of Entropy and Mutual Information*, Neural Comput. 15 (2003) 1191–1253.
The estimator-core mechanics rest on Kraskov, Stögbauer & Grassberger, Phys. Rev. E 69 (2004)
066138 (erratum Phys. Rev. E 83 (2011) 019903) and Kozachenko & Leonenko, Probl. Inf. Transm.
23 (1987) 95–101; the true-MI invariance instantiates the invariance theorem above. -/
theorem ksg_estimator_not_information_invariant {c : ℝ} (hc : 0 < c) (hc1 : c ≠ 1) :
    (∀ (P : Measure (ℝ × ℝ)) [IsProbabilityMeasure P],
        P ≪ (P.map Prod.fst).prod (P.map Prod.snd) →
        InformationTheory.mutualInformationReal (P.map (scaleEquiv hc.ne'))
          = InformationTheory.mutualInformationReal P)
      ∧ ksgRadiusStat (witnessCloud.map (scaleEmb hc.ne'))
            (two_le_card_map_scaleEmb hc.ne' two_le_witnessCloud_card)
          ≠ ksgRadiusStat witnessCloud two_le_witnessCloud_card := by
  refine ⟨fun P _ hP => ?_, ksgRadiusStat_not_invariant.2.1 c hc hc1⟩
  exact InformationTheory.mutualInformationReal_map_prodCongr
    ((Homeomorph.mulLeft₀ c hc.ne').toMeasurableEquiv)
    ((Homeomorph.mulLeft₀ c hc.ne').toMeasurableEquiv) P hP

/-!
# Scale-invariance of the *standardised* radius statistic (strongest form)

The block above is the **non**-invariance side: the raw KSG radius core
`ksgRadiusStat` scales by `c` under a uniform coordinate rescaling `scaleEmb c`
(`ksgRadiusStat_map_smul`), so the same data re-expressed in a rescaled coordinate
reports a different value — the estimator core is not a reparametrisation invariant.

This block is the **fix**, and the formal reason the *standardised* probe is the
trustworthy estimator. We **standardise the sample to unit per-axis spread before the
neighbour search** and prove the resulting `standardizedRadiusStat` is *invariant* under
diagonal coordinate rescaling. Concretely, dividing each coordinate by its own range
`spreadX`/`spreadY` makes the point cloud — and hence every nearest-neighbour radius, and
hence the whole statistic — depend only on the *shape* of the configuration, not on the
units of the axes. This is the standard preprocessing prescription for k-NN information
estimators (whitening / per-axis normalisation), here made exact: the standardisation
*cancels* the rescaling.

## The mechanism

Write `Tᵤ := S.map (scaleEmb c)` for the uniformly rescaled cloud (and `T := S.map
(diagEmb c₁ c₂)` for the anisotropic one). The ranges scale with the coordinates,
`spreadX Tᵤ = c · spreadX S` and `spreadY Tᵤ = c · spreadY S` (each from `max'`/`min'`
scaling by the positive factor, `Finset.max'_image`/`min'_image` with
`monotone_mul_left_of_nonneg`). Therefore the standardising map sends the rescaled point
`c • p` to `(c·p.1 / (c·spreadX S), c·p.2 / (c·spreadY S)) = (p.1 / spreadX S, p.2 /
spreadY S)` — the `c` cancels coordinate-by-coordinate — so **the standardised finsets are
literally equal**, `standardize Tᵤ = standardize S`. The two `standardizedRadiusStat`s are
then equal because `ksgRadiusStat` is being evaluated on *the same finset* (its cardinality
hypothesis is proof-irrelevant: `ksgRadiusStat_congr`). The radius geometry is never
recomputed; the rescaling has been quotiented out before the estimator ever sees the cloud.

The same cancellation works **per axis** for the anisotropic diagonal scaling `(x,y) ↦
(c₁ x, c₂ y)`, which is the strongest form: it covers exactly the anisotropic case that the
raw-estimator scope note (`ksgRadiusStat_map_smul`) left to the empirical study. This
anisotropic case is *proven* here, not recorded as a boundary — see
`standardizedRadiusStat_diagEmb_invariant`.

## Non-degeneracy

Standardisation by the range is only defined when each range is positive: a cloud collapsed
onto a vertical or horizontal line has a zero spread and cannot be normalised on that axis.
`NonDegenerate S` is exactly `0 < spreadX S ∧ 0 < spreadY S`; under it the standardising map
is globally injective (division by a nonzero constant per coordinate), so the standardised
cloud keeps `2 ≤ card` and feeds `ksgRadiusStat`. Positivity of the spreads is preserved by
positive scaling, so the rescaled cloud is non-degenerate whenever the original is.

## Main statements

* `spreadX`, `spreadY`: per-axis range of the sample (`max' − min'` of the coordinate
  projections).
* `NonDegenerate`: both spreads strictly positive.
* `standardize` / `standardizedRadiusStat`: the per-axis unit-spread normalisation of the
  cloud, and the radius core evaluated on it.
* `standardizedRadiusStat_scaleEmb_invariant`: invariance under uniform scaling `scaleEmb c`
  — the direct counterpart to the raw non-invariance `ksgRadiusStat_map_smul`.
* `standardizedRadiusStat_diagEmb_invariant` (strongest form): invariance under the
  anisotropic diagonal scaling `diagEmb c₁ c₂`.

## References

* Kraskov, Stögbauer & Grassberger, *Estimating mutual information*, Phys. Rev. E 69 (2004)
  066138, erratum Phys. Rev. E 83 (2011) 019903 — the KSG estimator built on k-NN radii and
  its sensitivity to the coordinate metric (§II); the present standardisation is the probe
  cross-check that removes that sensitivity for the radius core.
* Towards Robust Scale-Invariant Mutual Information Estimators, TMLR (2024) — scale-invariant
  k-NN MI estimation by per-axis normalisation / whitening before the neighbour search (the
  estimator-design literature this theorem formalises a kernel of).
* Gao, Ver Steeg & Galstyan, *Efficient estimation of mutual information for strongly
  dependent variables*, AISTATS 2015 (PMLR 38:277–286) — local geometry / rescaling
  dependence of k-NN information estimators motivating normalisation.
* Czyż, Grabowski, Vogt, Beerenwinkel & Marx, *Beyond Normal: On the Evaluation of Mutual
  Information Estimators*, NeurIPS 36 (2023), Theorem 2 — invariance of the *true* MI under
  injective reparametrisation of each coordinate; standardisation is the estimator-side
  attempt to inherit (a fragment of) that invariance, which this block makes exact for the
  radius core.
-/

/-! ### Per-axis spread and the non-degeneracy predicate -/

/-- The X-projection `S.image Prod.fst` of a cloud with `2 ≤ card` is nonempty (image of a
nonempty finset). Feeds `Finset.max'`/`min'` in `spreadX`. -/
lemma imageFst_nonempty {S : Finset (ℝ × ℝ)} (hS : 2 ≤ S.card) :
    (S.image Prod.fst).Nonempty :=
  (Finset.card_pos.mp (by omega)).image _

/-- The Y-projection `S.image Prod.snd` of a cloud with `2 ≤ card` is nonempty. Feeds
`Finset.max'`/`min'` in `spreadY`. -/
lemma imageSnd_nonempty {S : Finset (ℝ × ℝ)} (hS : 2 ≤ S.card) :
    (S.image Prod.snd).Nonempty :=
  (Finset.card_pos.mp (by omega)).image _

/-- **Per-axis spread on the X-coordinate.** The range of the first coordinate over the
sample: `max' − min'` of `S.image Prod.fst`. This is the natural
scale of the X-axis as read off the data; standardisation divides the X-coordinate by it so
that the normalised sample has unit X-spread, killing the X-units. -/
def spreadX (S : Finset (ℝ × ℝ)) (hS : 2 ≤ S.card) : ℝ :=
  (S.image Prod.fst).max' (imageFst_nonempty hS)
    - (S.image Prod.fst).min' (imageFst_nonempty hS)

/-- **Per-axis spread on the Y-coordinate.** The range of the second coordinate over the
sample: `max' − min'` of `S.image Prod.snd`. -/
def spreadY (S : Finset (ℝ × ℝ)) (hS : 2 ≤ S.card) : ℝ :=
  (S.image Prod.snd).max' (imageSnd_nonempty hS)
    - (S.image Prod.snd).min' (imageSnd_nonempty hS)

/-- The X-spread is always nonnegative (`min' ≤ max'`). It is a genuine dispersion; the
non-degeneracy predicate below asks for *strict* positivity. -/
lemma spreadX_nonneg {S : Finset (ℝ × ℝ)} (hS : 2 ≤ S.card) : 0 ≤ spreadX S hS := by
  rw [spreadX, sub_nonneg]; exact Finset.min'_le_max' _ _

/-- The Y-spread is always nonnegative (`min' ≤ max'`). -/
lemma spreadY_nonneg {S : Finset (ℝ × ℝ)} (hS : 2 ≤ S.card) : 0 ≤ spreadY S hS := by
  rw [spreadY, sub_nonneg]; exact Finset.min'_le_max' _ _

/-- **Non-degeneracy.** A cloud is non-degenerate when it has strictly positive spread on
*both* axes — i.e. it is not collapsed onto a horizontal or
vertical line — so that per-axis standardisation (division by the spread) is well defined and
injective. This is the precise admissibility condition under which the standardised radius
statistic is defined and scale-invariant. -/
def NonDegenerate (S : Finset (ℝ × ℝ)) (hS : 2 ≤ S.card) : Prop :=
  0 < spreadX S hS ∧ 0 < spreadY S hS

/-! ### Standardisation to unit per-axis spread -/

/-- The standardising map of a cloud: divide each coordinate by its own per-axis spread,
`(x, y) ↦ (x / spreadX S, y / spreadY S)`. On a non-degenerate cloud this is a globally
injective affine-diagonal rescaling that normalises the sample to unit X- and Y-spread. -/
def standardizeMap (S : Finset (ℝ × ℝ)) (hS : 2 ≤ S.card) : (ℝ × ℝ) → (ℝ × ℝ) :=
  fun p => (p.1 / spreadX S hS, p.2 / spreadY S hS)

/-- **Standardisation of a cloud:** the image of `S` under `standardizeMap`, i.e. the sample
rescaled to unit per-axis spread. The neighbour search of
the standardised radius statistic runs on *this* cloud, after the units have been removed. -/
def standardize (S : Finset (ℝ × ℝ)) (hS : 2 ≤ S.card) : Finset (ℝ × ℝ) :=
  S.image (standardizeMap S hS)

/-- On a non-degenerate cloud the standardising map is **globally injective**: dividing each
coordinate by a nonzero constant is injective per axis, hence injective on `ℝ × ℝ`. This is
what keeps `standardize` from collapsing the cardinality. -/
lemma standardizeMap_injective {S : Finset (ℝ × ℝ)} (hS : 2 ≤ S.card)
    (hnd : NonDegenerate S hS) : Function.Injective (standardizeMap S hS) := by
  intro p q h
  simp only [standardizeMap, Prod.mk.injEq] at h
  obtain ⟨h1, h2⟩ := h
  have hx : spreadX S hS ≠ 0 := hnd.1.ne'
  have hy : spreadY S hS ≠ 0 := hnd.2.ne'
  exact Prod.ext (by field_simp at h1; exact h1) (by field_simp at h2; exact h2)

/-- A standardised non-degenerate cloud still has at least two points: the standardising map
is injective (`standardizeMap_injective`), so `Finset.card_image_of_injective` preserves the
cardinality. This is the cardinality hypothesis `standardizedRadiusStat` needs to call
`ksgRadiusStat` on the normalised cloud. -/
lemma two_le_card_standardize {S : Finset (ℝ × ℝ)} (hS : 2 ≤ S.card)
    (hnd : NonDegenerate S hS) : 2 ≤ (standardize S hS).card := by
  rw [standardize, Finset.card_image_of_injective _ (standardizeMap_injective hS hnd)]
  exact hS

/-- **The standardised KSG radius statistic.** The k = 1 neighbourhood-radius core
`ksgRadiusStat` evaluated on the *standardised* cloud —
the sample first normalised to unit per-axis spread, then fed to the nearest-neighbour radius
sum. This is the trustworthy probe: unlike the raw `ksgRadiusStat` (which scales by `c` under
rescaling, `ksgRadiusStat_map_smul`), it reads off only the shape of the configuration, and
is invariant under diagonal coordinate rescaling (the two theorems below).

## References
Kraskov, Stögbauer & Grassberger, Phys. Rev. E 69 (2004) 066138 (erratum 83 (2011) 019903)
— the radius core whose metric-sensitivity standardisation removes; *Towards Robust
Scale-Invariant Mutual Information Estimators*, TMLR (2024) — per-axis normalisation /
whitening for scale-invariant k-NN MI estimation. -/
def standardizedRadiusStat (S : Finset (ℝ × ℝ)) (hS : 2 ≤ S.card)
    (hnd : NonDegenerate S hS) : ℝ :=
  ksgRadiusStat (standardize S hS) (two_le_card_standardize hS hnd)

/-! ### Proof-irrelevance plumbing for the dependent cardinality / nonemptiness arguments -/

/-- `Finset.max'` depends only on the finset, not the nonemptiness proof, and respects
finset equality. (Used to rescale `max'` of a coordinate projection past the dependent
`Nonempty` argument without a motive-not-type-correct failure.) -/
lemma max'_congr {α : Type*} [LinearOrder α] {s t : Finset α}
    (h : s = t) (hs : s.Nonempty) (ht : t.Nonempty) : s.max' hs = t.max' ht := by
  subst h; rfl

/-- `Finset.min'` depends only on the finset, not the nonemptiness proof, and respects
finset equality. -/
lemma min'_congr {α : Type*} [LinearOrder α] {s t : Finset α}
    (h : s = t) (hs : s.Nonempty) (ht : t.Nonempty) : s.min' hs = t.min' ht := by
  subst h; rfl

/-- `ksgRadiusStat` depends only on the finset, not the `2 ≤ card` proof, and respects finset
equality. This is the proof-irrelevance fact that turns the finset equality `standardize T =
standardize S` into the statistic equality — the heart of the invariance argument: the
radius geometry is identical because it is computed on identical finsets. -/
lemma ksgRadiusStat_congr {A B : Finset (ℝ × ℝ)} (h : A = B)
    (hA : 2 ≤ A.card) (hB : 2 ≤ B.card) : ksgRadiusStat A hA = ksgRadiusStat B hB := by
  subst h; rfl

/-- Pointwise reduction of the image of a mapped finset: if `g (f p) = h p` on `S`, then
`(S.map f).image g = S.image h`. (Used to evaluate `standardize` of a pushed-forward cloud
by composing the standardising map with the scaling embedding, term by term, avoiding a
`map_eq_image` rewrite under the dependent standardising map.) -/
lemma image_map_congr {f : (ℝ × ℝ) ↪ (ℝ × ℝ)} {g h : (ℝ × ℝ) → (ℝ × ℝ)}
    {S : Finset (ℝ × ℝ)} (H : ∀ p ∈ S, g (f p) = h p) :
    (S.map f).image g = S.image h := by
  rw [Finset.map_eq_image, Finset.image_image]
  exact Finset.image_congr (fun p hp => H p hp)

/-- The standardised statistic is determined by the standardised cloud: equal standardised
finsets give equal `standardizedRadiusStat` (`ksgRadiusStat_congr`). This is the bridge from
"standardisation cancels the rescaling" (a finset equality) to "the statistic is invariant"
(the theorems below). -/
lemma standardizedRadiusStat_congr_of_standardize_eq {S T : Finset (ℝ × ℝ)}
    (hT : 2 ≤ T.card) (hndT : NonDegenerate T hT)
    (hS : 2 ≤ S.card) (hndS : NonDegenerate S hS)
    (heq : standardize T hT = standardize S hS) :
    standardizedRadiusStat T hT hndT = standardizedRadiusStat S hS hndS := by
  rw [standardizedRadiusStat, standardizedRadiusStat]
  exact ksgRadiusStat_congr heq _ _

/-! ### The anisotropic diagonal scaling (strongest-form reparametrisation) -/

/-- The **anisotropic diagonal** measurable equivalence `diagEquiv c₁ c₂ : ℝ × ℝ ≃ᵐ ℝ × ℝ`,
`(x, y) ↦ (c₁ • x, c₂ • y)`, for `c₁, c₂ ≠ 0`. Built as `e₁.prodCongr e₂` with `eᵢ`
left-multiplication by `cᵢ`, generalising `scaleEquiv` (which is the `c₁ = c₂` case) and
staying a `MeasurableEquiv`. This is the strongest reparametrisation the standardised
statistic is shown invariant under — exactly the anisotropic rescaling the raw estimator's
scope note leaves to the empirical study. -/
def diagEquiv {c₁ c₂ : ℝ} (h₁ : c₁ ≠ 0) (h₂ : c₂ ≠ 0) : (ℝ × ℝ) ≃ᵐ (ℝ × ℝ) :=
  ((Homeomorph.mulLeft₀ c₁ h₁).toMeasurableEquiv).prodCongr
    ((Homeomorph.mulLeft₀ c₂ h₂).toMeasurableEquiv)

/-- `diagEquiv` acts as the anisotropic diagonal scaling `(c₁ * p.1, c₂ * p.2)`. -/
@[simp] lemma diagEquiv_apply {c₁ c₂ : ℝ} (h₁ : c₁ ≠ 0) (h₂ : c₂ ≠ 0) (p : ℝ × ℝ) :
    diagEquiv h₁ h₂ p = (c₁ * p.1, c₂ * p.2) := rfl

/-- The measurable embedding underlying `diagEquiv`, used to push a `Finset (ℝ × ℝ)`
forward. Its coercion is defeq to `⇑(diagEquiv h₁ h₂)`. Note `scaleEmb hc = diagEmb hc hc`
definitionally, so the uniform case is a literal specialisation. -/
def diagEmb {c₁ c₂ : ℝ} (h₁ : c₁ ≠ 0) (h₂ : c₂ ≠ 0) : (ℝ × ℝ) ↪ (ℝ × ℝ) :=
  (diagEquiv h₁ h₂).toEquiv.toEmbedding

@[simp] lemma diagEmb_apply {c₁ c₂ : ℝ} (h₁ : c₁ ≠ 0) (h₂ : c₂ ≠ 0) (p : ℝ × ℝ) :
    diagEmb h₁ h₂ p = (c₁ * p.1, c₂ * p.2) := rfl

/-- An anisotropically scaled cloud still has at least two points (the embedding is
injective, so `Finset.card_map` preserves cardinality). -/
lemma two_le_card_map_diagEmb {c₁ c₂ : ℝ} (h₁ : c₁ ≠ 0) (h₂ : c₂ ≠ 0)
    {S : Finset (ℝ × ℝ)} (hS : 2 ≤ S.card) : 2 ≤ (S.map (diagEmb h₁ h₂)).card := by
  rwa [Finset.card_map]

/-- The X-projection of an anisotropically scaled cloud is the X-projection scaled by `c₁`:
`(S.map (diagEmb c₁ c₂)).image Prod.fst = (S.image Prod.fst).image (c₁ * ·)`. -/
lemma imageFst_map_diagEmb {c₁ c₂ : ℝ} (h₁ : c₁ ≠ 0) (h₂ : c₂ ≠ 0) (S : Finset (ℝ × ℝ)) :
    (S.map (diagEmb h₁ h₂)).image Prod.fst
      = (S.image Prod.fst).image (fun x : ℝ => c₁ * x) := by
  rw [Finset.map_eq_image, Finset.image_image, Finset.image_image]
  apply Finset.image_congr; intro p _; rfl

/-- The Y-projection of an anisotropically scaled cloud is the Y-projection scaled by `c₂`. -/
lemma imageSnd_map_diagEmb {c₁ c₂ : ℝ} (h₁ : c₁ ≠ 0) (h₂ : c₂ ≠ 0) (S : Finset (ℝ × ℝ)) :
    (S.map (diagEmb h₁ h₂)).image Prod.snd
      = (S.image Prod.snd).image (fun x : ℝ => c₂ * x) := by
  rw [Finset.map_eq_image, Finset.image_image, Finset.image_image]
  apply Finset.image_congr; intro p _; rfl

/-- **X-spread scales by `c₁`** under anisotropic scaling: `spreadX (S.map (diagEmb c₁ c₂)) =
c₁ * spreadX S` for `c₁ > 0`. Both `max'` and `min'` of the X-projection scale by the
positive factor (`Finset.max'_image`/`min'_image` with `monotone_mul_left_of_nonneg`), and
the difference scales likewise. -/
lemma spreadX_map_diagEmb {c₁ c₂ : ℝ} (h₁pos : 0 < c₁) (h₂ : c₂ ≠ 0)
    {S : Finset (ℝ × ℝ)} (hS : 2 ≤ S.card) :
    spreadX (S.map (diagEmb h₁pos.ne' h₂)) (two_le_card_map_diagEmb h₁pos.ne' h₂ hS)
      = c₁ * spreadX S hS := by
  have hmono := monotone_mul_left_of_nonneg (a := c₁) h₁pos.le
  rw [spreadX, max'_congr (imageFst_map_diagEmb h₁pos.ne' h₂ S) _
        ((imageFst_nonempty hS).image _),
      min'_congr (imageFst_map_diagEmb h₁pos.ne' h₂ S) _
        ((imageFst_nonempty hS).image _),
      Finset.max'_image hmono, Finset.min'_image hmono, spreadX]
  ring

/-- **Y-spread scales by `c₂`** under anisotropic scaling. -/
lemma spreadY_map_diagEmb {c₁ c₂ : ℝ} (h₁ : c₁ ≠ 0) (h₂pos : 0 < c₂)
    {S : Finset (ℝ × ℝ)} (hS : 2 ≤ S.card) :
    spreadY (S.map (diagEmb h₁ h₂pos.ne')) (two_le_card_map_diagEmb h₁ h₂pos.ne' hS)
      = c₂ * spreadY S hS := by
  have hmono := monotone_mul_left_of_nonneg (a := c₂) h₂pos.le
  rw [spreadY, max'_congr (imageSnd_map_diagEmb h₁ h₂pos.ne' S) _
        ((imageSnd_nonempty hS).image _),
      min'_congr (imageSnd_map_diagEmb h₁ h₂pos.ne' S) _
        ((imageSnd_nonempty hS).image _),
      Finset.max'_image hmono, Finset.min'_image hmono, spreadY]
  ring

/-- Anisotropic positive scaling **preserves non-degeneracy**: positive spreads stay positive
after multiplication by `c₁, c₂ > 0`. So the standardised statistic is defined on the scaled
cloud whenever it is defined on the original. -/
lemma nonDegenerate_map_diagEmb {c₁ c₂ : ℝ} (h₁pos : 0 < c₁) (h₂pos : 0 < c₂)
    {S : Finset (ℝ × ℝ)} (hS : 2 ≤ S.card) (hnd : NonDegenerate S hS) :
    NonDegenerate (S.map (diagEmb h₁pos.ne' h₂pos.ne'))
      (two_le_card_map_diagEmb h₁pos.ne' h₂pos.ne' hS) := by
  refine ⟨?_, ?_⟩
  · rw [spreadX_map_diagEmb h₁pos h₂pos.ne' hS]; exact mul_pos h₁pos hnd.1
  · rw [spreadY_map_diagEmb h₁pos.ne' h₂pos hS]; exact mul_pos h₂pos hnd.2

/-- **The cancellation lemma** (anisotropic): standardising an anisotropically scaled cloud
gives the *same finset* as standardising the original, `standardize (S.map (diagEmb c₁ c₂)) =
standardize S`. The standardising map sends the scaled point `(c₁ x, c₂ y)` to `(c₁ x / (c₁
spreadX S), c₂ y / (c₂ spreadY S)) = (x / spreadX S, y / spreadY S)` — the scale factors
cancel per axis (`mul_div_mul_left`) against the rescaled spreads. This is where the
rescaling is quotiented out. -/
lemma standardize_map_diagEmb {c₁ c₂ : ℝ} (h₁pos : 0 < c₁) (h₂pos : 0 < c₂)
    {S : Finset (ℝ × ℝ)} (hS : 2 ≤ S.card) :
    standardize (S.map (diagEmb h₁pos.ne' h₂pos.ne'))
        (two_le_card_map_diagEmb h₁pos.ne' h₂pos.ne' hS)
      = standardize S hS := by
  rw [standardize, standardize]
  apply image_map_congr
  intro p _
  show standardizeMap (S.map (diagEmb h₁pos.ne' h₂pos.ne')) _ (diagEmb h₁pos.ne' h₂pos.ne' p)
        = standardizeMap S hS p
  simp only [standardizeMap, diagEmb_apply]
  rw [spreadX_map_diagEmb h₁pos h₂pos.ne' hS, spreadY_map_diagEmb h₁pos.ne' h₂pos hS,
      mul_div_mul_left _ _ h₁pos.ne', mul_div_mul_left _ _ h₂pos.ne']

/-! ### Invariance under anisotropic diagonal scaling (strongest form) -/

/-- **The standardised radius statistic is invariant under anisotropic diagonal coordinate
rescaling (strongest form).** For `c₁, c₂ > 0` and a non-degenerate cloud `S`,

    standardizedRadiusStat (S.map (diagEmb c₁ c₂)) _ _ = standardizedRadiusStat S hS hnd.

This is the strongest form of the standardisation result: it covers the *anisotropic*
reparametrisation `(x, y) ↦ (c₁ x, c₂ y)` — precisely the case that moves the raw KSG
estimate and that the raw-estimator scope note (`ksgRadiusStat_map_smul`) leaves to the
empirical study. Here it is *proven*: standardising to unit per-axis spread cancels the
per-axis scale factors (`standardize_map_diagEmb`), so the normalised clouds coincide and the
statistic is literally unchanged (`standardizedRadiusStat_congr_of_standardize_eq`). The
anisotropic case is proven, not recorded as a boundary.

## References
*Towards Robust Scale-Invariant Mutual Information Estimators*, TMLR (2024) (per-axis
normalisation / whitening for scale-invariant k-NN MI estimation — the design principle this
theorem formalises for the radius core); Kraskov, Stögbauer & Grassberger, Phys. Rev. E 69
(2004) 066138, erratum 83 (2011) 019903 (the rescaling-sensitivity of the raw k-NN radii
this normalisation removes); Czyż, Grabowski, Vogt, Beerenwinkel & Marx, *Beyond Normal*,
NeurIPS 36 (2023), Theorem 2 (invariance of the true MI under injective per-coordinate
reparametrisation — the population invariance the standardised estimator inherits for the
radius core). -/
theorem standardizedRadiusStat_diagEmb_invariant {c₁ c₂ : ℝ} (h₁pos : 0 < c₁) (h₂pos : 0 < c₂)
    {S : Finset (ℝ × ℝ)} (hS : 2 ≤ S.card) (hnd : NonDegenerate S hS) :
    standardizedRadiusStat (S.map (diagEmb h₁pos.ne' h₂pos.ne'))
        (two_le_card_map_diagEmb h₁pos.ne' h₂pos.ne' hS)
        (nonDegenerate_map_diagEmb h₁pos h₂pos hS hnd)
      = standardizedRadiusStat S hS hnd :=
  standardizedRadiusStat_congr_of_standardize_eq _ _ hS hnd
    (standardize_map_diagEmb h₁pos h₂pos hS)

/-! ### Invariance under uniform scaling (counterpart to the non-invariance) -/

/-- Uniform positive scaling preserves non-degeneracy. (Specialisation of
`nonDegenerate_map_diagEmb` to `c₁ = c₂ = c`, since `scaleEmb hc = diagEmb hc hc`
definitionally.) -/
lemma nonDegenerate_map_scaleEmb {c : ℝ} (hcpos : 0 < c) {S : Finset (ℝ × ℝ)}
    (hS : 2 ≤ S.card) (hnd : NonDegenerate S hS) :
    NonDegenerate (S.map (scaleEmb hcpos.ne')) (two_le_card_map_scaleEmb hcpos.ne' hS) :=
  nonDegenerate_map_diagEmb hcpos hcpos hS hnd

/-- **The cancellation lemma** (uniform): standardising a uniformly scaled cloud gives the
same finset as standardising the original. (Specialisation of `standardize_map_diagEmb`.) -/
lemma standardize_map_scaleEmb {c : ℝ} (hcpos : 0 < c) {S : Finset (ℝ × ℝ)}
    (hS : 2 ≤ S.card) :
    standardize (S.map (scaleEmb hcpos.ne')) (two_le_card_map_scaleEmb hcpos.ne' hS)
      = standardize S hS :=
  standardize_map_diagEmb hcpos hcpos hS

/-- **The standardised radius statistic is invariant under uniform coordinate rescaling.**
For `c > 0` and a non-degenerate cloud `S`,

    standardizedRadiusStat (S.map (scaleEmb c)) _ _ = standardizedRadiusStat S hS hnd.

This is the direct counterpart to the raw non-invariance `ksgRadiusStat_map_smul`
(`ksgRadiusStat (S.map (scaleEmb c)) _ = c * ksgRadiusStat S hS`): the *same* uniform
reparametrisation `scaleEmb c` that multiplies the raw radius core by `c` leaves the
*standardised* statistic exactly fixed. Standardising to unit per-axis spread before the
neighbour search is therefore the formal reason the standardised probe is the trustworthy
estimator — it reports the shape of the configuration, not the units of the coordinates.

## References
Kraskov, Stögbauer & Grassberger, *Estimating mutual information*, Phys. Rev. E 69 (2004)
066138, erratum 83 (2011) 019903 (the raw radius core's metric-sensitivity, §II); *Towards
Robust Scale-Invariant Mutual Information Estimators*, TMLR (2024) (scale-invariant k-NN MI
estimation by per-axis normalisation); Czyż, Grabowski, Vogt, Beerenwinkel & Marx, *Beyond
Normal*, NeurIPS 36 (2023), Theorem 2 (the population reparametrisation-invariance the
standardised estimator inherits for the radius core). -/
theorem standardizedRadiusStat_scaleEmb_invariant {c : ℝ} (hcpos : 0 < c)
    {S : Finset (ℝ × ℝ)} (hS : 2 ≤ S.card) (hnd : NonDegenerate S hS) :
    standardizedRadiusStat (S.map (scaleEmb hcpos.ne'))
        (two_le_card_map_scaleEmb hcpos.ne' hS)
        (nonDegenerate_map_scaleEmb hcpos hS hnd)
      = standardizedRadiusStat S hS hnd :=
  standardizedRadiusStat_congr_of_standardize_eq _ _ hS hnd
    (standardize_map_scaleEmb hcpos hS)

end LewmValidity.EstimatorGeometry

end

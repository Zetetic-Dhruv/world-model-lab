# A validity protocol for representation probing, in URS form

**Object.** A four-gate protocol that decides whether a measured property of a learned
representation (mutual-information estimate, decoding-probe R², effective rank) may be
admitted as **knowledge** or must be returned to **open inquiry**.

**Kind of URS.** This is a *problem* URS on the **discovery (γ) axis** — the object is the
protocol, ⟨A, M, R, T⟩ describes the protocol, not the reasoner. It was produced the hard
way: a multi-day study in which four confident, plottable, mechanistically-narratable
results were each a distinct artifact. The protocol is what stabilized.

---

## 0. Critical argument — KK-resolution

A measurement artifact is not an honest unknown. It is a **false-KK**: an articulated
commitment ("effective rank ≈ 12", "MI saturates at 3.4–3.5") *treated as stabilized
knowledge* while not, in fact, invariant. A false-KK is **more dangerous than a KU**,
because a KU drives inquiry (you keep asking) whereas a false-KK **halts** it (you publish,
you build on it, you stop looking).

The protocol's job is therefore not to *measure* but to **gate admission to KK**. Each gate
applies a perturbation under which a *genuine* property must be invariant; an apparent-KK is
admitted to true-KK only after surviving **all four**. A gate-failure is a **Pl-kill** — a
precise boundary, and per A4 **content, not failure**: the false-KK collapses to an honest
KU ("we do not yet know the converged rank") or to a UU-boundary ("this estimator cannot
see this").

This raises the bar of Pl-admissibility for what counts as *known* about a representation.
That re-grammaring of "known" is the move: the same number is admissible-as-KK before the
protocol and inadmissible after. The quadrant transitions the protocol enacts:

```
apparent effect ──(survives all 4 gates)──▶ true-KK
false-KK        ──(fails a gate: Pl-kill)──▶ KU (honest open question)  or  UU-boundary
```

---

## A — Axioms / identity (the Will-snapshot the protocol holds constant)

- **A-i (gated KK).** A measured quantity is **not knowledge until it survives every gate.**
  "Plottable", "monotone", "low-variance-across-seeds" are not admission tickets.
- **A-ii (Pl-kill is content, A4).** A failed gate is a discovered boundary, reported as such,
  never hidden or softened.
- **A-iii (anti-simplification, A5).** When a gate fails, *escalate* (add the control, raise the
  power, change the estimator) — never downgrade the **question** to the weaker thing the broken
  measurement could support.
- **A-iv (baseline-relative reading).** A representation quantity is read **against a provable
  ceiling/floor**, never as an absolute. "The latent encodes 0.33" is meaningless without the
  ceiling it is 0.33 *of*.
- **A-v (reflexivity, Axiom 1.16).** The protocol is itself a world-model under inquiry; it must
  survive its own gates (see §T-3, where it does — and kills one of its own instruments).

---

## R — grammar / what becomes askable

Before this grammar, "is my probe-R² an artifact?" is **UK/UU** — felt as unease, not
askable-as-such. The protocol installs the vocabulary that makes it a **KU** (askable,
resolvable):

| term | makes askable |
|---|---|
| **false-KK** | "is this number knowledge, or belief?" |
| **convergence-matched** | "am I comparing models, or comparing training-stages?" |
| **estimator-reparametrization-invariance** | "did the *quantity* move, or did the *estimator's geometry* move?" |
| **group/trajectory leakage** | "did the probe learn the target, or learn which trajectory it's in?" |
| **statistical power-floor** | "is this null a property, or too few groups?" |
| **decodability-ceiling (DPI)** | "less than *what* maximum could possibly be decoded?" |

R defines askability: the four gates exist because this grammar lets the question be posed.

---

## M — mechanisms (the four gates, as test-operators) + proof-witnesses

Each gate is an **Inv operator**: it perturbs the measurement along one axis and demands the
property survive. Each carries a witness, tagged by `act(·)` type (the LLM-reliability and
formality of the evidence differ — A4/2.5).

### M-0 · Convergence gate — *Inv under training-time*
**Operator.** Measure m(θ_t) along the training trajectory; admit only if m has plateaued; compare
conditions **convergence-matched** (each at its own plateau), never at a shared epoch.
**Proof-witness.** *Near-trivial (flagged, A4):* if dm/dt ≠ 0 at the measurement step then
m(θ_t) ≠ m(θ_∞) generically, so m is not a converged property — a definition-unfold, not a theorem.
Its force is **empirical**: showing the precondition (non-stationarity) actually held. `act(describe)`.
**Tooling.** `run_sweep.py --keep-ckpt-epochs`; `plot_trajectory.py`.

### M-1 · Estimator-geometry gate — *Inv under reparametrization*
**Operator.** Cross-check any k-NN MI (KSG) trend against a standardized decoding probe; if they
disagree, the KSG trend is geometry.
**Proof-witness (formal, borrowed-theorem transport).** Mutual information is **invariant under
invertible maps**: I(φ(z); s) = I(z; s) (Cover & Thomas; Kraskov et al. 2004). KSG depends on
k-NN distances in z-space, which are **not** invariant under the non-isometric rescalings that
anti-collapse regularizers (VICReg, Barlow-Twins, SIGReg) induce over training. Hence a KSG change
*attributable to geometry* is provably an estimation artifact, since the true MI is unmoved.
`act(prove)` via `act(structural-mapping)` — the theorem is textbook (McAllester–Stratos 2020 and
Gao et al. 2016 give the complementary KSG-bias formalisation); the **diagnostic application is the
new instrument**. **Caveat (A4):** training also genuinely shifts p(z,s), so not all KSG motion is
artifact — the probe-disagreement *isolates* the artifact component, it does not prove the whole drift
is artifact.
**Tooling.** `ksg_mi` and `state_decoding_probe` surfaced together so disagreement is visible.

### M-2 · Split-leakage gate — *Inv under split-structure*
**Operator.** Split train/test **by trajectory/group** (`GroupShuffleSplit` on episode ids), never
by frame; hold probe capacity and budget fixed (`early_stopping=False`) across conditions.
**Proof-witness (constructive Pl-kill).** Nuisance decomposition: if z_t = h(s_t) + b(episode) + ε,
a frame-split probe sees every episode in *both* train and test, learns b̂(episode) for all of them,
and subtracts it — so its R² reports h-decodability *with the nuisance known*. A trajectory-split
probe meets unseen episodes, has no b̂ for them, and reports the **honest cross-episode** decodability
of h. The synthetic adversarial dataset (§T-2) **constructs** a model where this gap is total —
`act(counterexample)`, the strongest own-witness in the document: a concrete world in which the naive
method is confidently wrong.
**Tooling.** `state_decoding_probe(..., groups=episode_ids)`; `extract_latent_pairs` returns ids.

### M-3 · Power-and-ceiling gate — *Comp + Pl*
**Operator.** Run the identical probe on the **raw input** as a ceiling control; scale held-out groups
until the ceiling is positive; report representation-R² **normalized to the input ceiling**.
**Proof-witness (formal + reflexive Pl-kill).** Data-Processing Inequality: the state generates the
observation generates the latent, s → x → z (Markov), so I(s; z) ≤ I(s; x); equivalently, since
z = E(x) is deterministic, every decoder of s from z composes to a decoder from x, so best-decode(z)
≤ best-decode(x). **The raw input is a provable ceiling.** `act(prove)` (Cover & Thomas). **And then
the protocol turns on itself:** our ceiling *instrument* used PCA-192 of pixels, a lossy
variance-shaped compression x → x̃ with I(s; x̃) ≤ I(s; x) — *not* the true ceiling. So the
task-shaped latent legitimately beat it (§T-3: 0.33 > 0.30) **without violating DPI**. That is a
`act(counterexample)` against our **own** Gate-3 instrument — a reflexive Pl-kill (A-v) — and it
*sharpens* the gate: the ceiling must be the full input or a capacity-matched input probe, never a
lossy projection.
**Tooling.** raw-pixel control (`PCA → state_decoding_probe`); episode-count scaling.

---

## T — traces (the empirical witnesses; the encounter material)

Each trace is tagged with the gate it drove and what it witnessed. These are the γ-content:
nothing here is conjecture, all are run outputs.

### T-0 — convergence
`effective rank`: 11.3 (ep9) → 15.96 (ep39) → 21.71 (ep69) → **28.05 (ep99), slope still > 0.**
The 10-epoch number "≈12" was a snapshot mid-climb. *Witness:* the precondition of M-0 (non-stationarity)
held — the false-KK "rank ≈ 12, resolution-invariant" was a training-stage reading.

### T-1 — estimator geometry
`KSG MI(128px; state)`: 3.64 → 3.39, a clean monotone decrease (a beautiful "compression" story).
`probe-MLP R²(128px)`: 0.81 → 0.80, **flat.** *Witness:* the quantity (probe) did not move; the
estimator (KSG) did. The narratable decrease was the SIGReg-reshaped geometry, exactly the M-1 artifact.

### T-2 — split leakage
`frame-split probe-R²`: 64px **0.886**, 128px **0.798** (the original headline gap).
`trajectory-split probe-R²`: 64px **−0.56**, 128px **−0.98** (the gap inverts to noise).
`synthetic adversarial` (state under a per-episode offset): frame **0.90** → trajectory **−1.5**.
*Witness:* the constructive counterexample shows frame-split R² measures memorized nuisance, not
decodability — the 0.886/0.798 false-KK was within-episode leakage.

### T-3 — power & ceiling
`raw-pixel decode, trajectory-split`: 50 episodes **−0.31** → 500 episodes **+0.19**.
`powered latent (64px)` linear **0.33** vs `PCA-192 pixel ceiling` linear **0.30**.
*Witness:* the negative R² that *looked* like "the representation destroyed the state" was the
power-floor — raw pixels failed identically at 50 episodes. And the latent exceeding the PCA-pixel
"ceiling" is the reflexive Pl-kill of our own instrument (M-3): PCA-192 was not the true ceiling.

### Consolidated: false-KK → reality

| false-KK (withdrawn) | gate that killed it | reality |
|---|---|---|
| effective rank ≈ 12, resolution-invariant | M-0 | under-trained; → ~28 and rising |
| MI(z; state) saturates at 3.4–3.5 | M-0 + M-1 | rises to 3.9 *and* the KSG "trend" is geometry |
| state-sufficiency gap 0.886 vs 0.798 | M-2 + M-3 | frame-leakage; then power-floor; gap currently unmeasured |

**Survives all four gates → true-KK:** predictive-accuracy convergence speed is monotone in
resolution; effective rank rises over training (it measures anti-collapse compliance, not quality).
Both estimator-robust, both probe-free.

---

## Proof-witness ledger (the explicit ask)

| Gate | claim it defends | witness | `act(·)` | formality | own / borrowed |
|---|---|---|---|---|---|
| M-0 | snapshot ≠ converged property | rank slope ≠ 0 at ep99 | describe | near-trivial (flagged) | own (empirical) |
| M-1 | KSG drift can be pure geometry | MI-invariance under invertible φ ⇒ KSG-vs-true gap | prove ∘ structural-map | **theorem** (borrowed) + new application | instrument new |
| M-2 | frame-split measures nuisance | synthetic offset model: 0.90 → −1.5 | **counterexample** | constructive Pl-kill | **own** |
| M-3 | raw input is a ceiling; null can be power | DPI: I(s;z) ≤ I(s;x); + PCA-pixel self-kill | prove + counterexample | **theorem** + reflexive Pl-kill | instrument new |

Two `act(prove)` witnesses transport textbook theorems (MI-invariance; DPI) into a **diagnostic**
role — the theorems are not new, the *instruments* are. Two `act(counterexample)` witnesses are own,
constructive, and the strongest: M-2's synthetic, and M-3 turning the DPI on the protocol's **own**
ceiling instrument. M-0's "proof" is honestly a definition-unfold; its value is the trace.

---

## Ledger (the protocol's own ignorance state)

- **KK (admitted, gated):** the four gates are *individually necessary* — each has a witness that
  a real study tripped on. The two survivors above are true-KK.
- **KU (open, drives discovery):** is the powered, ceiling-correct sufficiency gap real? (→ `V2_DESIGN.md`).
  Is M-0 ever sufficient *alone*, or only jointly?
- **UK (pressure, drives inquiry):** is the set of four **exhaustive**? A fifth artifact class
  (e.g., goal/target distribution shift between probe-train and probe-test groups) is pressing on the
  grammar (Axiom 1.16: the protocol is a world-model under inquiry).
- **UU:** artifacts unaskable in the present grammar — by construction, the ones we cannot yet see.

## η / honest boundary

Per literature reconnaissance (`act(search)`), this is **consolidation of scattered cautions plus two
novel instruments**, *not* novelty on leakage or MI-fragility:
- **owned elsewhere:** group-leakage (Kapoor & Narayanan 2023; subject-wise splits, medical/EEG ML);
  MI-estimation fragility (McAllester & Stratos 2020; Gao et al. 2016); probe capacity/control baselines
  (Hewitt & Liang 2019; Pimentel et al. 2020; Voita & Titov 2020); measurement-artifact-as-phenomenon
  in training dynamics (Saxe et al. 2018); DPI (Cover & Thomas).
- **the two new instruments:** (1) **probe-vs-KSG disagreement as a geometry detector** keyed to
  anti-collapse regularizers; (2) **DPI-ceiling normalization** of decodability — which, applied
  reflexively, caught the flaw in our own ceiling control.
- **the synthesis:** no source assembles all four as one *ordered validity gauntlet for SSL /
  world-model representation probing*, a community where M-0, M-1-as-cross-check, and M-2 are routinely
  violated. The carefulness keeps being independently rediscovered (Kapoor & Narayanan's own diagnosis).

**The strongest single validation is reflexive:** the protocol, run on itself, used Gate-3's DPI to
Pl-kill its own ceiling instrument. A validity protocol that catches its own artifact is the only kind
worth trusting.

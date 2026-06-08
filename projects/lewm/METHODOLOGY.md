# Measuring representation quality without fooling yourself

### A four-gate validity protocol for probing SSL / world-model latents

This is the concrete, durable outcome of the LeWM resolution study. The "results"
of that study are mostly in limbo or dead; **this protocol is what survived, and
it is reusable for any work that measures how much a learned representation encodes
about some quantity** (mutual information, linear/MLP probe R², effective rank,
"state sufficiency," etc.).

> A formal companion — the same protocol in URS form ⟨A,M,R,T⟩, with the
> KK-resolution argument and the **proof-witness ledger** (MI-invariance for Gate 1,
> the data-processing-inequality ceiling for Gate 3, the constructive
> counterexample for Gate 2) — is in [`METHODS_URS.md`](METHODS_URS.md).

It exists because, over a multi-day GPU study, **we were about to report four
different confident claims, and each one was a distinct measurement artifact.**
Each was caught by a cheap control. The lesson is ordered and unforgiving:

> A measured "effect" on a learned representation must clear **all four gates**
> before it is a claim. Skipping any one of them will, eventually, publish an artifact.

Order matters — later gates only make sense once earlier ones pass.

---

## Gate 0 — Convergence: never measure a snapshot

**Failure mode.** A metric read at a fixed early epoch reflects *training stage*,
not a converged property of the model. Worse: when you compare conditions (here,
input resolutions) at a *fixed epoch*, you compare models sitting at *different
convergence stages*, and every apparent cross-condition "effect" can be a
convergence-rate difference in disguise.

**The control.** Measure the metric's **trajectory vs epoch** at several
checkpoints; verify it has plateaued. Compare conditions **convergence-matched**
(each at its own plateau), never at a shared fixed epoch.

**The fix.** Train to convergence; keep checkpoints across training; report each
condition at its own plateau; use a heterogeneous epoch budget if conditions
converge at different rates.

**Worked example (what it caught).** At 10 epochs we measured "effective rank ≈ 12
of 192, resolution-invariant" and "MI saturates at 3.4–3.5." Trained to 100 epochs
with checkpoint trajectories:
- effective rank: 11.3 (ep9) → **28.0 (ep99), still rising** — not converged, not 12.
- MI(z;state): 3.37 (ep9) → **3.91 (ep99), still rising** — never saturated.

Both headline numbers were under-training snapshots. The fixed-10-epoch
cross-resolution comparison was comparing a chaotic-early 64px model against a
near-converged 128px model.

---

## Gate 1 — Estimator geometry-robustness: k-NN MI lies under a moving metric

**Failure mode.** k-nearest-neighbour mutual-information estimators (KSG) on a
high-dimensional latent are corrupted by changes in the latent's **scale and
anisotropy** — and anti-collapse regularizers (SIGReg, VICReg, Barlow-Twins) *actively
reshape* that geometry over training. So a KSG-MI "trend" over training (or across
conditions) can be **pure geometry, carrying no change in actual information.** A
drift of ~0.2 nats is exactly the size an estimator artifact produces.

**The control.** Cross-check every KSG-MI trend against a **decoding probe** (train
a regressor to predict the target from the frozen latent, report R²). The probe is
invariant to latent scale/anisotropy when inputs are standardized. If KSG and probe
disagree, **the KSG trend was geometry.**

**The fix.** Treat the decoding probe as the trustworthy estimator. Demote KSG to a
flagged cross-check, useful only to *demonstrate* the artifact.

**Worked example (what it caught).** KSG said 128px "sheds state info over training"
(MI 3.64 → 3.39, a clean monotone decrease — a beautiful slow-feature-compression
story). The probe said **flat** (R² 0.81 → 0.80). The decrease was the SIGReg
regularizer contracting the latent, not information loss. We nearly built a
mechanism narrative on a geometry artifact.

---

## Gate 2 — Split leakage: probe by trajectory, never by frame

**Failure mode.** Probing on a **frame-level** train/test split lets the probe
memorize **per-trajectory latent offsets**. Frames within one episode are highly
autocorrelated, so a random frame split puts near-duplicate frames in both train and
test. The probe learns "which trajectory is this" and reads the answer off the
offset — inflating R², and (critically) inflating it **more for whichever condition
has smoother latents**, which can *manufacture or erase a cross-condition gap.*

**The control.** Split train/test by **trajectory / episode** (`GroupShuffleSplit`
on episode IDs), never by frame. Compare frame-split vs trajectory-split R²; a large
drop *is* the leakage.

**The fix.** Always trajectory-split. Hold probe **capacity fixed** (same regressor,
same width) and **budget fixed** (fixed iterations, **early-stopping off** — early
stopping makes the training budget data-dependent, so you'd be measuring
probe-fitting effort, not encoding quality) across all conditions.

**Worked example (what it caught).** Frame-split gave 64px = 0.886, 128px = 0.798 —
the entire "predictability–sufficiency dissociation" headline. Trajectory-split gave
64px = **−0.56**, 128px = **−0.98**. The 0.88/0.80 gap was *pure within-episode
leakage*. A synthetic adversarial check (state buried under a per-episode latent
offset) made it unmistakable: **frame-split R² = 0.90, trajectory-split R² = −1.5.**

---

## Gate 3 — Power and baseline: the raw-input ceiling

**Failure mode.** Once you trajectory-split honestly (Gate 2), you can swing to the
*opposite* artifact: with **too few held-out trajectories**, the probe is
underpowered, and **even the raw input** — which provably contains the target —
fails to decode it cross-trajectory (negative R²). A negative R² then *looks* like
"the representation destroyed the information," a dramatic kill, when it is only a
sample-size artifact.

**The control.** Run the **identical probe on the raw input** (e.g. PCA of pixels)
under the same trajectory split. If raw-input R² is *also* negative, you are
underpowered, not measuring a representation property. Scale up the held-out
trajectory count and re-check.

**The fix.** Probe with **hundreds of held-out trajectories** (50 is not enough for a
group-split). Report the representation's R² **relative to the raw-input R² ceiling** —
that ratio ("how much of the input's decodable state does the encoder retain?") is
the meaningful, normalized quantity; an absolute R² is not.

**Worked example (what it caught).** Trajectory-split on 50 episodes gave raw-pixel
R² = **−0.31** — i.e. *raw pixels* couldn't decode state cross-episode either. With
500 episodes, raw-pixel R² = **+0.19**. The earlier negative latent R² was a power
artifact, **not** a clean kill of the dissociation. Against the powered pixel ceiling
(0.30 linear), the powered latent showed 64px ≈ ceiling (0.33, retains state) and
128px **below** ceiling (−0.37, sheds state) — which is the *normalized* form of the
effect, the only form worth claiming.

---

## Summary table

| Gate | Artifact it catches | Cheap control | One-line fix |
|---|---|---|---|
| 0 Convergence | snapshot ≠ converged; fixed-epoch compares different stages | trajectory vs epoch; check plateau | train to convergence; convergence-matched comparison |
| 1 Estimator geometry | KSG-MI moved by regularizer's scale/anisotropy, not info | cross-check vs decoding probe | probe is truth; KSG is a flagged cross-check |
| 2 Split leakage | frame-split memorizes per-trajectory offsets | frame-split vs trajectory-split | `GroupShuffleSplit` on episodes; fixed capacity+budget |
| 3 Power & baseline | underpowered group-split → even raw input fails | same probe on raw pixels; scale episodes | hundreds of held-out trajectories; normalize to raw-input ceiling |

## The meta-lesson

Four times in one study, a confident, plottable, mechanistically-narratable result
was an artifact — under-training, estimator geometry, frame leakage, and probe
underpowering, in that order. Each survived until a *specific* control was run, and
each control was cheap relative to the GPU-days spent producing the artifact.

In URT terms: each Pl-kill was a **discovery of a boundary**, not a failure. The
durable γ of the whole episode is not a Reacher number — it is this protocol. The
η-optimal move in any future representation-probing study is to **run all four gates
before believing the first figure**, because the cost of the controls is trivial
against the cost of retracting a published artifact.

## Implementation in this repo

- Gate 0: `tools/run_sweep.py --keep-ckpt-epochs` (checkpoint trajectories) +
  `tools/plot_trajectory.py` (metric-vs-epoch).
- Gate 1: `src/diagnostics.py::ksg_mi` (flagged) vs `state_decoding_probe` (truth);
  both surfaced together so disagreement is visible.
- Gate 2: `state_decoding_probe(..., groups=episode_ids)` → `GroupShuffleSplit`,
  `early_stopping=False`; `extract_latent_pairs` returns episode IDs.
- Gate 3: raw-pixel control is a few lines (PCA of obs → same probe); v2 mandates a
  large dedicated held-out probe set and ceiling-normalized reporting.

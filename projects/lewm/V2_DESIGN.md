# v2 experiment design — predictability vs state-sufficiency in a JEPA world model

Locked design for the next run. v1 produced one durable thing (the four-gate
validity protocol in `METHODOLOGY.md`); v2 is the clean experiment that protocol
makes possible. **Nothing here is claimable from v1** — v1's surviving signal
(64px latent retains state to the pixel ceiling, 128px falls below it) is
single-seed, contaminated, and must be re-earned cleanly.

## The single falsifiable question

> In a JEPA world model, is minimizing self-prediction loss aligned or anti-aligned
> with world-state sufficiency — and does planning return follow predictability or
> sufficiency?

**Kill condition (write it down):** if the 64-vs-128 sufficiency gap (normalized to
the raw-pixel ceiling, trajectory-split, held-out, powered) washes out across seeds,
the hypothesis is dead. Clean stop.

## Design table

| Factor | Setting | Rationale |
|---|---|---|
| Knob | resolution {48, 64, 80, 96, 128}, fixed 16×16 grid (patch = res/16) | dense **at/below the knee** (sufficiency saturates ≥128); ≥160 dropped as dead plateau |
| Seeds | 5 on anchors {64,128}, 3 on knee {48,80,96}; adaptive → 7 if seed-SD/gap > 0.15 | error bars are the finding; n=3 too noisy for the anchors |
| Budget | heterogeneous, plateau-reported: 48/64→80 ep, 80→60, 96/128→50; frontier X = converged (plateau-avg) L_pred | 64px L_pred unstable to ~ep70-90; a 60-cap mislocates its frontier point |
| **Probe set** | **dedicated held-out pool of 300–500 episodes the encoder never trained on** | 50 val episodes cannot power a trajectory-split probe (Gate 3); requires data regen |
| Headline metrics | val L_pred (predictability); **probe-R² (lin+MLP), trajectory-split, normalized to raw-pixel ceiling** (sufficiency); **planning return** (control) | all estimator-robust per the four gates |
| Secondary | effective rank (anti-collapse compliance, NOT quality) | flagged as such |
| Demoted | KSG MI → artifact cross-check only | Gate 1 |

## The four gates, baked in (see METHODOLOGY.md)

- **Gate 0:** convergence-matched, heterogeneous budget, plateau-reported.
- **Gate 1:** probe is truth; KSG only to demonstrate the artifact.
- **Gate 2:** `GroupShuffleSplit` on episodes, fixed capacity + budget (early_stopping off).
- **Gate 3:** 300–500 held-out probe episodes; report probe-R² **÷ raw-pixel-ceiling R²**.

## Pre-registered control-outcome forks (planning return)

Lock the CEM config (horizon, samples, iterations, latent cost) **identical across all
cells**, or a reviewer asks whether 64px just got a friendlier planner. Return needs
its own seed error bars (CEM is stochastic) and a numeric decision threshold a priori.
Return metric: mean −final-distance-to-goal-state (+ success-rate secondary).

1. **Return tracks SUFFICIENCY** (64 ≥ 128) → "self-prediction loss / rank are
   anti-correlated proxies for control." The provocative one.
2. **Return tracks PREDICTABILITY** (128 ≥ 64) → "predictability is what control needs;
   sufficiency matters only to a floor." Reconciles with practice.
3. **Return tracks NEITHER** (flat) → "both axes saturate above a threshold;
   resolution irrelevant to control once minimal." Null with teeth.

## Mechanism sub-test (free, high value)

Decode each state dim separately (already computed: `probe_*_r2_per_dim`). Slow-feature
account predicts JEPA keeps position, sheds velocity. If the gap concentrates in qvel
(dims 4-5) rather than spread, that's the mechanism — and pre-commits the bridge to
fork 1 (a reacher needs velocity → return tracks sufficiency).

## Compute / parallelization

- RAM-cache the dataset (kills the dataloader bottleneck that GPU-starved high-res;
  h5 files < 180 MB fit in memory). 2-3× per-cell speedup; makes the extra epochs free.
- 2 VMs split a disjoint slice (anchors seeds 1-2 on one, knee on the other).
- 2a = **hard STOP/GO gate** before 2b: if the powered held-out gap dies on the
  anchors, never run the knee (η-optimal).

## Gate order of operations

1. Regenerate data: large train set + **dedicated 300-500-episode held-out probe pool**.
2. Build: RAM-cache dataset; per-checkpoint planning-return eval (dm_control headless,
   `MUJOCO_GL=egl`); raw-pixel-ceiling baseline into the probe pipeline.
3. Stage 2a: anchors {64,128} × 5 seeds, powered held-out probe, ceiling-normalized.
   **Gate.** If gap survives → 2b; else stop, write the negative result.
4. Stage 2b: knee {48,80,96} × 3 seeds. Stage 2c: planning return on all kept ckpts.

## Centerpiece figure

Predictability–sufficiency frontier, **return-colored**: X = converged val L_pred,
Y = ceiling-normalized probe-R², color = planning return, one point per
(resolution × seed) with seed error bars, line tracing resolution. Second panel:
per-dim decode (velocity-R² vs position-R² by resolution) as the mechanism.

## Honest — not claimable even after v2

- **One task** (Reacher) + **one architecture** (tiny preset). A general claim needs
  ≥2 dm_control tasks (or PushT / OGBench). v2 = solid single-task workshop; main-conf
  needs task #2.
- **Probe-R² = decodable (lin+MLP) state**, not total information.
- **Decodable ≠ used.** Return partly addresses it, but a causal intervention
  (ablate latent dims → measure return) is the gold standard v2 still lacks.
- **Patch covaries with density** (48→patch-3, 128→patch-8): the knob is jointly
  total-pixels and pixels-per-token; the effect is reframable as patch-embedding
  capacity. Name it; watch 48px/patch-3 for low-end degeneracy faking a knee.
- **Return is a joint test** of encoder + predictor + planner: a null-return could be
  a planner-geometry failure, not sufficiency-irrelevance.

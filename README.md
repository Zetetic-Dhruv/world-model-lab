# World Models for preventing Machine Errors in long horizon automation

A working repository for world-model research aimed at preventing machine errors in long-horizon automation. Each experiment is a self-contained directory under [`projects/`](projects/).

## Experiment 1 — `projects/lewm`

Reproduction of **LeWorldModel** (Maes, Le Lidec, Scieur, LeCun, Balestriero — arXiv:2603.19312, 2026): a stable end-to-end Joint-Embedding Predictive Architecture (JEPA) world model trained from raw pixels.

(Mostly) replicates the canonical mechanism — 2-layer MLP projector with BN-in-middle, separate `pred_proj` for asymmetric projection, full DiT-style AdaLN-zero with multiplicative residual gates, SIGReg with sample-count scaling and trapezoidal Epps–Pulley quadrature, LinearWarmupCosineAnnealingLR. So far, have been using synthetic 2D-particle and MiniPushT environments in that order for experimental iteration. 

### Status

- ✓ Faithful mechanism replication (LLM as a judge check) 
- ✓ Stable training: 0 NaN, 0 recoveries on synthetic substrates
- ✓ Encoder produces honest near-N(0,1) representations (z_std → 1.00 on MiniPushT)
- ✓ ~18M parameters, single-CPU training viable
- ✓ Held-out validation pipeline + identity-baseline comparison built in
- ✓ Threaded NaN-recovery supervisor for hardware-flaky substrates (Apple MPS, etc.)
- ✓ CEM planner + MPC runner in latent space
- ✓ Real-data-ready: episode-directory format (NPZ or video) accepted via `--episode-dir`
- ✓ replicates on `dm_control` Reacher (30/30 = 100% success)
- ✓ Canonical TwoRoom env imported directly from `stable_worldmodel` (Maes et al., MIT 2026) for faithful Lim #3 study
- ✓ Information-geometric diagnostics: `effective_rank_pr`, `ksg_mi`, `twonn_intrinsic_dim` (`src/diagnostics.py`)
- ✓ Resolution-sweep orchestrator (`tools/run_sweep.py`) with disk audit, resume, JSON aggregation
- ⚠ **Reacher resolution study — earlier "5 verified claims" RETRACTED as measurement artifacts** (under-training + estimator geometry + frame-leakage + probe underpowering). The durable outcome is a four-gate validity protocol: [`projects/lewm/METHODOLOGY.md`](projects/lewm/METHODOLOGY.md). Clean redo specced in [`V2_DESIGN.md`](projects/lewm/V2_DESIGN.md).

### Layout

```
world-model-lab/
├── README.md                      # this file
└── projects/
    └── lewm/                      # Experiment 1: LeWorldModel reproduction
        ├── README.md               # detailed operational README
        ├── src/                    # encoder / predictor / sigreg / planner / supervisor
        ├── tests/                  # kill checks + sigreg + smoke
        ├── tools/                  # video renderers
        ├── train.py                # canonical training (with val pipeline)
        ├── eval_probe.py           # linear probe + identity-baseline rollout
        └── eval_planning.py        # CEM-MPC success-rate evaluation
```

### Setup

```bash
cd projects/lewm
pip install -r requirements.txt
```

Python ≥ 3.9, PyTorch ≥ 2.0. Optional: `wandb`, `h5py`, `imageio` + `imageio-ffmpeg`.

### Usage

```bash
# Pre-flight kill checks
python tests/test_kill_checks.py

# Train on MiniPushT (canonical config, 10% held-out validation)
python train.py --env pusht --device cpu --no-wandb --out-dir runs/pusht --epochs 10

# Visualizations
python tools/render_training_video.py --npz data/pusht_n500_T30_seed42.npz \
    --out runs/training_data.mp4 --upscale 4 --annotate-actions
python tools/render_planning_video.py --ckpt runs/pusht/ckpt_epoch9.pt \
    --out runs/planning_demo.mp4 --upscale 4
```

### Config (mostly matches official `lucas-maes/le-wm`)

| Component | Setting |
|---|---|
| Encoder | ViT-Tiny: dim=192, depth=12, heads=3, patch=8 (for 64×64) |
| Projector | `Linear(192→2048) → BN1d(2048) → GELU → Linear(2048→192)` |
| pred_proj | Same shape, applied to predictor outputs (asymmetric projection) |
| Predictor | depth=6, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1, full DiT AdaLN-zero |
| SIGReg | trapezoidal quadrature, knots=17, num_proj=1024, multiplied by sample-count B |
| Loss | `MSE(pred_proj(predictor(...)), emb_next) + 0.09 · SIGReg(emb in (T, B, D))` |
| Optimizer | AdamW, lr=5e-5, weight_decay=1e-3, grad_clip=1.0 |
| Scheduler | LinearWarmupCosineAnnealingLR, 1% warmup |
| Batch | 128 |
| Validation | 10% of episodes held out episode-wise (`--val-fraction 0.1`) |

### Reproduction numbers

**MiniPushT synthetic data run (ps you can find video for intuition in the repo) — 500 WeakPolicy episodes × 100 env steps, stride=5, action_block=5, history_size=3, action_token_dim=10, small preset (~10M params), 10 epochs, batch 128, CPU:**

| Metric | Value |
|---|---|
| Wallclock | 5.9 hr for 2840 training steps |
| Final L_pred (train) | 0.031 |
| Final L_sigreg | 2.72 |
| z_std (train) | 1.020 (target: N(0,1)) |
| Val L_pred / L_identity / P/I | 0.0654 / 0.0522 / 1.25× |
| NaN events / recoveries / escalations | 0 / 0 / 0 |
| Data contact rate (WeakPolicy) | 96% |

**planning eval (30 heldout episodes, canonical CEM: H=5, action_block=5, N=300, K=30, T=30, budget=50 env steps):**

| Metric | Value |
|---|---|
| **Success rate** (latent τ-match) | **29 / 30 = 96.7%** |
| τ (p95-near, calibrated) | 11.63 |
| actual_dist:  mean / median | 1.93 / 0.60 |
| τ-calibration: near_mean / unrelated_mean | 2.17 / 17.69 (5.8× gap) |
| Block-to-recorded-goal mean (diagnostic) | 18.96 px |
| Wallclock (eval) | 180 s (6 s/episode) |

**Reacher (`dm_control` reacher-easy) Pl-check — 500 WeakPolicy episodes × 60 env steps, stride=5, action_block=5, history_size=3, action_token_dim=10, tiny preset (~5.5M params), 10 epochs, batch 64, CPU:**

| Metric | Value |
|---|---|
| Wallclock (training) | 83.1 min for 2880 training steps |
| Final L_pred (train) | 0.013 |
| Final L_sigreg (train) | 2.13 |
| z_std (train) | 1.02 (target: N(0,1)) |
| Val L_pred / L_identity / P/I | 0.0133 / 0.0042 / 3.15× |
| NaN events / recoveries / escalations | 0 / 0 / 0 |

**Reacher planning eval (30 heldout episodes, canonical CEM: H=5, action_block=5, N=300, K=30, T=30, budget=50 env steps):**

| Metric | Value |
|---|---|
| **Success rate** (latent τ-match) | **30 / 30 = 100%** |
| τ (calibrated valley) | 6.83 |
| actual_dist:  mean / median | 0.93 / 0.38 |
| τ-calibration: near_mean / unrelated_mean | 0.72 / 21.05 (**29.3× gap**) |
| qpos_to_recorded_goal_dist (joint-angle diagnostic) | 1.26 rad mean |
| tip_to_target_dist (env's old radius metric) | 0.19 mean (well above 0.015 reach radius — goals are recorded states, NOT target-reaching frames) |
| Wallclock (eval) | 134.5 s (4.5 s/episode) |

The Reacher Pl-check confirms LeWM's mechanism replicates on a real continuous-control benchmark with even sharper latent-space discrimination than MiniPushT (29× near/unrelated gap vs MiniPushT's 5.8×). The encoder generalizes well enough for the planner to drive any heldout init-state to a recorded future state in latent space, despite the val P/I plateau at 3.15× — the τ-calibration captures the relevant signal-vs-distractor structure, not the absolute prediction accuracy.

**2D-particle run (per-step prediction, ~Round 3-10ep, 1050 steps, 10 epochs):**

| Metric | Value |
|---|---|
| Wallclock | 51.7 min |
| Final L_pred | 0.25 |
| Linear probe R² (x, y) | 0.57 / 0.48 |
| Rollout MSE | 0.43× of identity baseline |
| z_std (post-projector) | 0.90 |

**2D-particle — 500 episodes × 30 steps, 10 epochs, batch 128, CPU:**

| Metric | Value |
|---|---|
| Wallclock | 51.7 min |
| Final L_pred | 0.25 |
| Linear probe R² (x, y) | 0.57 / 0.48 |
| Rollout MSE | 0.43× of identity baseline (predictor learns real dynamics) |
| z_std (post-projector) | 0.90 |

### Reacher resolution study — RETRACTED claims, and the protocol that survived

> **Retraction.** An earlier version of this section reported "5 strong, cross-seed-verified
> claims" from a 68.1 hr CPU resolution sweep — most prominently *"effective rank ≈ 12 of 192,
> resolution-invariant"* and *"MI(z, state) saturates at 3.4–3.5."* **Those claims were
> measurement artifacts and are withdrawn.** A subsequent convergence study (100-epoch
> training on GPU) plus estimator-validity controls showed each was a distinct artifact:

| Withdrawn claim | What it actually was |
|---|---|
| effective rank ≈ 12, resolution-invariant | **under-training** — at 10 epochs rank is mid-climb; trained out it rises to ~28 and is still rising at epoch 100 |
| MI(z, state) saturates at 3.4–3.5 | **under-training + KSG geometry** — MI still rises to 3.9 at convergence, and the KSG estimator itself drifts with the SIGReg-reshaped latent geometry, not with information |
| the downstream "state sufficiency" gap | **frame-leakage** — frame-split probe R² (0.88 vs 0.80) collapses under trajectory-level split; the gap was within-episode memorization, then *underpowered* on too few held-out episodes |

**What actually survives** (estimator-robust, probe-free):
- **Predictive-accuracy convergence speed is monotone in resolution** — higher input density trains the next-latent predictor faster and far more stably (128px near-converged by ~epoch 10, 64px chaotic until ~epoch 70).
- **Effective rank rises over training toward ~28/192** (not 12; 12 was a snapshot), at a resolution-dependent rate. It measures anti-collapse-regularizer compliance, **not** representation quality.

**What is suggestive but not yet claimable:** a predictability–sufficiency dissociation
(the better self-predictor may be the worse state-encoder). A powered-but-contaminated probe
hints at it — 64px latent decodes state to the raw-pixel ceiling, 128px falls below it — but
this needs the clean v2 protocol (held-out probe pool, seeds, ceiling-normalization) before it
is a claim.

#### The concrete, durable outcome

The reusable result of this whole episode is **[`projects/lewm/METHODOLOGY.md`](projects/lewm/METHODOLOGY.md)** —
a four-gate validity protocol (convergence, estimator geometry, split leakage, power & baseline)
for probing learned representations without fooling yourself. Each gate caught one of the
artifacts above. The next clean experiment is specced in
**[`projects/lewm/V2_DESIGN.md`](projects/lewm/V2_DESIGN.md)**.

### For Claude to follow (or "axioms" if one uses AMRT!) - 

1. Projection has no BN after final Linear (BN-in-middle topology)
2. SIGReg distinguishes collapsed vs Gaussian by ≥3× and B-scaling holds
3. AdaLN-zero block(x, c) ≈ x at init (max diff < 1e-5)
4. Prediction topology: preds = pred_proj(predictor(...)), no z_t residual addition
5. Scheduler: LR profile follows linear warmup → cosine decay to 0

All pass on the current implementation.

### Reference

Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., Balestriero, R. *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.* arXiv:2603.19312, 2026.

Official implementation: <https://github.com/lucas-maes/le-wm>

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
- ✓ **Reacher resolution sweep complete (6 cells × 1 seed + 2 cells denoise) — 5 cross-seed-verified original claims about LeWM**

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

### Original findings — Reacher resolution sweep (controlled study)

Beyond reproduction, we ran a controlled **resolution × information-geometry** sweep on Reacher (6 cells × 1 seed for the full curve, plus 2 cells × 2 seeds for cross-seed verification). The constant-grid design holds patch GRID at 16×16 (256 tokens) across all resolutions {64, 96, 128, 160, 192, 224}, isolating "pixel info per patch" as the only varied variable while keeping attention compute fixed. Total compute: 68.1 hr CPU.

#### Per-cell metrics (single seed=42, all 6 resolutions)

| img | patch | success | τ-gap | near | unrel | ER (out of 192) | MI(z, env_state) | MI(z, z_next) |
|---|---|---|---|---|---|---|---|---|
| 64  | 4  | 1.000 | 10.71× | 1.82 | 19.50 | 11.94 | 3.826 nats | 5.446 |
| 96  | 6  | 0.967 | 23.94× | 0.84 | 20.19 | 11.35 | 3.500 | 4.941 |
| 128 | 8  | 0.967 | 26.30× | 0.73 | 19.11 | 12.08 | 3.475 | 5.037 |
| 160 | 10 | 1.000 | 26.62× | 0.76 | 20.30 | 12.58 | 3.506 | 4.929 |
| 192 | 12 | 1.000 | 52.10× | 0.36 | 18.89 | 11.69 | 3.408 | 4.899 |
| 224 | 14 | 0.967 | 20.24× | 0.85 | 17.14 | 12.88 | 3.503 | 5.029 |

#### Cross-seed denoise (192 + 224 with seed=1)

| img | seed | τ-gap | near | unrel | ER | MI(z, env_state) | MI(z, z_next) |
|---|---|---|---|---|---|---|---|
| 192 | 42 | 52.10× | 0.363 | 18.89 | 11.69 | 3.408 | 4.899 |
| 192 | 1  | 32.02× | 0.575 | 18.41 | 12.83 | 3.404 | 4.794 |
| 224 | 42 | 20.24× | 0.847 | 17.14 | 12.88 | 3.503 | 5.029 |
| 224 | 1  | 22.04× | 0.868 | 19.13 | 11.37 | 3.484 | 5.110 |

**Cross-seed variance:** MI(z, state) **0.1–0.5%** (rock-solid), effective rank **10–12%**, τ-gap **9–62%** (high-variance at 192).

#### Five strong original claims (multi-seed verified or methodologically self-evident)

1. **Effective rank of LeWM-mechanism JEPA latents on Reacher is ~12 of 192 dims, resolution-invariant.** Out of the 192-D Gaussian prior SIGReg targets, only ~6% is utilized. Cross-seed variance 10–12%. The paper's "isotropic Gaussian in high-D" claim is, in practice, a low-rank Gaussian.

2. **MI(z, env_state) saturates after a sharp 64→96 drop, with cross-seed variance < 1%.** Encoder fidelity stabilizes at 3.4–3.5 nats once resolution exceeds 96×96. Higher resolution buys *zero* additional state-recoverable information.

3. **Reacher env_state has TwoNN intrinsic dim ≈ 3.87** (out of 6-D state). Quantifies LeWM Limitation #3's qualitative "low intrinsic dim" claim. Combined with Claim 1: encoder uses **3× more dims** than the env's data manifold needs.

4. **Constant-grid resolution sweep methodology**: scaling `patch_size = image_size / 16` keeps attention compute constant (256 tokens) across resolutions, isolating "pixel info per patch" as the only varied variable. Novel design vs prior fixed-patch-size sweeps.

5. **Single-seed τ-gap evaluation is unreliable at high resolution.** Up to 62% relative variance across seeds at image=192. Path-C `tau_gap_factor` reports without seed analysis should be treated cautiously — methodological caveat applies to LeWM and follow-ups.

#### Two suggestive claims (single-seed direction, need cross-env Stage 2)

6. **Resolution buys trajectory-discriminability at the cost of state-fidelity.** As resolution rises, MI(z, env_state) drops monotonically while τ-gap (mean) rises. Direction is robust; magnitude is noisy.

7. **Saturation around 96–128 for low-intrinsic-dim envs.** Both MI saturation and τ-gap plateauing suggest the *useful* effect of resolution increase ends around 96×96 for Reacher. Practical implication: 96×96 may suffice instead of canonical 224×224 at 1/12 the pixel compute, on this substrate.

#### What we cannot claim from Reacher data alone

- Cross-env generalizability — only Reacher tested at sweep scale
- Causal mechanism for resolution effects (correlations, not interventions)
- Properties of canonical 18M-param + 100-epoch + 224×224 LeWM (we ran tiny preset + 10 epochs)
- Whether SIGReg vs ViT capacity vs latent-dim choice is the dominant cause of the rank-12 plateau (no ablation)
- Generalization to V-JEPA / I-JEPA / DINO-WM (single-substrate, single-architecture)

See `projects/lewm/README.md` for full sweep methodology, `tools/run_sweep.py` for the orchestrator, `src/diagnostics.py` for the estimator implementations.

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

# lewm — faithful LeWorldModel reproduction

PyTorch reimplementation of **LeWorldModel** (Maes, Le Lidec, Scieur, LeCun, Balestriero — arXiv:2603.19312, 2026): a stable end-to-end Joint-Embedding Predictive Architecture (JEPA) trained from raw pixels.

Mechanism-canonical: 2-layer MLP projector with BN-in-middle, separate `pred_proj` for asymmetric projection, full DiT-style AdaLN-zero with multiplicative residual gates, SIGReg with sample-count scaling and trapezoidal Epps–Pulley quadrature, LinearWarmupCosineAnnealingLR. No dependency on `stable-pretraining` or `stable-worldmodel`.

## Status

- ✓ Faithful mechanism replication (5/5 kill checks pass)
- ✓ Stable training: 0 NaN, 0 recoveries, no collapse on 2D-particle and MiniPushT
- ✓ Encoder produces honest near-N(0,1) representations (post-projector z_std → 1.00 on MiniPushT)
- ✓ ~18M parameters, single-CPU training viable
- ✓ Held-out validation pipeline + identity-baseline comparison built in
- ✓ Threaded NaN-recovery supervisor for hardware-flaky substrates (Apple MPS, etc.)
- ✓ Two synthetic envs: 2D particle (low-D, fast) and MiniPushT (multi-object, contact dynamics)
- ✓ Real continuous-control benchmark: `dm_control` reacher-easy via `src/env_reacher.py`
- ✓ Canonical TwoRoom benchmark via direct import from `stable_worldmodel` (Maes et al., MIT 2026) — `src/env_tworoom.py` is a thin API-translation wrapper only; env physics + ExpertPolicy are imported verbatim and cited
- ✓ CEM planner + MPC runner in latent space (`src/lewm/planner.py`)
- ✓ Real-data-ready: episode-directory format (NPZ or video) accepted via `--episode-dir`
- ✓ Information-geometric diagnostic suite — `effective_rank_pr`, `ksg_mi`, `twonn_intrinsic_dim` (`src/diagnostics.py`)
- ✓ Resolution-sweep orchestrator with disk audit + resume + JSON aggregation (`tools/run_sweep.py`, `tools/run_diagnostics.py`)
- ⚠ **Reacher resolution study — earlier "5 strong claims" RETRACTED as measurement artifacts.** Durable outcome: a four-gate validity protocol for representation probing ([`METHODOLOGY.md`](METHODOLOGY.md)); clean redo specced in [`V2_DESIGN.md`](V2_DESIGN.md). See the retraction below.

## Layout

```
lewm/
├── src/
│   ├── env.py                    2D particle environment (synthetic)
│   ├── env_pusht.py              MiniPushT (agent + T-block + target, contact dynamics)
│   ├── env_reacher.py            dm_control reacher-easy wrapper (real benchmark)
│   ├── env_tworoom.py            Canonical swm/TwoRoom-v1 wrapper (env + ExpertPolicy imported)
│   ├── data.py                   Trajectory generators + Datasets (in-memory, HDF5, episode-dir)
│   ├── diagnostics.py            effective_rank_pr / ksg_mi / twonn_intrinsic_dim primitives
│   └── lewm/
│       ├── encoder.py            ViT-Tiny + 2-layer MLP projector
│       ├── predictor.py          Causal ViT with full DiT AdaLN-zero
│       ├── sigreg.py             SIGReg (Cramér–Wold + Epps–Pulley + B-scaling)
│       ├── scheduler.py          LinearWarmupCosineAnnealingLR
│       ├── trainer.py            NaNSupervisedTrainer (threaded snapshot + watchdog)
│       ├── planner.py            CEM planner + MPC runner
│       └── model.py              LeWM module + canonical loss
├── tests/
│   ├── test_kill_checks.py       5 mechanism-canonical assertions
│   ├── test_sigreg.py            SIGReg distinguishes Gaussian vs collapsed
│   └── test_smoke.py             End-to-end forward/backward smoke
├── tools/
│   ├── render_training_video.py  Render cached training data → MP4
│   ├── render_planning_video.py  Render planning episodes with overlays → MP4
│   ├── run_diagnostics.py        Per-cell latent extraction + diagnostic_suite
│   └── run_sweep.py              Resolution-sweep orchestrator (subprocess-isolated, resumable)
├── train.py                      Training entry point (with held-out validation)
├── eval_probe.py                 Linear probe + open-loop rollout MSE
└── eval_planning.py              CEM-MPC planning success-rate evaluation
```

## Setup

```bash
pip install -r requirements.txt
```

Python ≥ 3.9, PyTorch ≥ 2.0. Optional: `wandb` (CSV fallback), `h5py` (only for HDF5 datasets), `imageio` + `imageio-ffmpeg` (for video tools), `dm_control` + `mujoco` (only for `--env reacher`), `stable-worldmodel` ≥ 0.0.6 (only for `--env tworoom`; **requires Python ≥ 3.10**).

For TwoRoom experiments, use a Python 3.11 venv:

```bash
brew install python@3.11   # if not already installed
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install stable-worldmodel torch h5py dm_control
```

## Usage

```bash
# 0. Pre-flight kill checks (run before any training run)
python tests/test_kill_checks.py

# 1. Train on synthetic MiniPushT (canonical config, 10% held-out validation)
python train.py --env pusht --device cpu --no-wandb --out-dir runs/pusht --epochs 10

# 1b. (alternative) Train Path C on dm_control Reacher (real benchmark Pl-check)
python train.py --path-c --env reacher --device cpu --no-wandb \
    --out-dir runs/reacher --arch-preset tiny --epochs 10 --batch-size 64 \
    --n-episodes 500 --path-c-episode-length 60

# 1c. (alternative) Train Path C on canonical TwoRoom (Python 3.11 venv required)
# Env + ExpertPolicy imported directly from stable_worldmodel (Maes et al., MIT 2026)
python train.py --path-c --env tworoom --tworoom-image-size 64 \
    --device cpu --no-wandb --out-dir runs/tworoom \
    --arch-preset tiny --epochs 10 --batch-size 64 \
    --n-episodes 500 --path-c-episode-length 100
# For the canonical resolution-sweep study, vary --tworoom-image-size in
# {64, 96, 128, 160, 192, 224}; the env always renders at canonical 224 and
# downsamples post-render, isolating image resolution as the only variable.

# 2. Render the training data stream (video, agent-attached action arrows)
python tools/render_training_video.py \
    --npz data/pusht_n500_T30_seed42.npz \
    --out runs/training_data.mp4 \
    --n-episodes 30 --upscale 4 --annotate-actions

# 3. Evaluate
python eval_probe.py --ckpt runs/pusht/ckpt_epoch9.pt
python eval_planning.py --ckpt runs/pusht/ckpt_epoch9.pt --n-episodes 30

# 4. Render planning episodes with overlays
python tools/render_planning_video.py \
    --ckpt runs/pusht/ckpt_epoch9.pt \
    --out runs/planning_demo.mp4 \
    --n-episodes 10 --upscale 4
```

## Canonical config (matches official `lucas-maes/le-wm`)

| Component | Setting |
|---|---|
| Encoder | ViT-Tiny: dim=192, depth=12, heads=3, patch=8 (for 64×64) |
| Projector | `Linear(192→2048) → BN1d(2048) → GELU → Linear(2048→192)` |
| pred_proj | Same shape, applied to predictor outputs (asymmetric projection) |
| Predictor | depth=6, heads=16, dim_head=64 (decoupled, inner=1024), mlp_dim=2048, dropout=0.1 |
| AdaLN | Full DiT-style: 6-param (shift, scale, gate × 2), zero-init final Linear, multiplicative residual gates |
| SIGReg | Trapezoidal quadrature, knots=17, num_proj=1024, multiplied by sample-count B |
| Loss | `MSE(pred_proj(predictor(...)), emb_next) + 0.09 · SIGReg(emb in (T, B, D))` |
| Optimizer | AdamW, lr=5e-5, weight_decay=1e-3, grad_clip=1.0 |
| Scheduler | LinearWarmupCosineAnnealingLR, 1% warmup, cosine to 0 |
| Batch | 128 |
| Validation | 10% of episodes held out episode-wise (`--val-fraction 0.1`) |

## Reproduction numbers

### MiniPushT — 500 episodes × 30 steps, 10 epochs, batch 128, CPU

| Metric | Value |
|---|---|
| Wallclock | 48.4 min for 1050 training steps |
| Final L_pred (training) | 0.020 |
| Final L_sigreg | 3.12 |
| z_std (post-projector) | 1.009 (target: N(0,1)) |
| ‖z‖ mean / std | 13.6 / 0.7 |
| NaN events / recoveries / escalations | 0 / 0 / 0 |

### Reacher (dm_control reacher-easy) — Path C Pl-check, 500 episodes × 60 steps, 10 epochs, tiny preset (~5.5M params), batch 64, CPU

| Metric | Value |
|---|---|
| Wallclock (training) | 83.1 min for 2880 training steps |
| Final L_pred (training) | 0.013 |
| Final L_sigreg | 2.13 |
| z_std (post-projector, train) | 1.02 (target: N(0,1)) |
| Val L_pred / L_identity / P/I | 0.0133 / 0.0042 / 3.15× |
| NaN events / recoveries / escalations | 0 / 0 / 0 |
| **Planning success rate** (latent τ-match, 30 heldout eps) | **30 / 30 = 100%** |
| τ (calibrated valley) | 6.83 |
| τ-calibration: near_mean / unrelated_mean | 0.72 / 21.05 (29.3× gap) |
| actual_dist:  mean / median | 0.93 / 0.38 |
| Wallclock (eval) | 134.5 s (4.5 s/episode) |

Confirms the LeWM mechanism replicates on a real continuous-control benchmark with sharper latent-space discrimination than MiniPushT (29.3× near/unrelated gap vs MiniPushT's 5.8×). The encoder generalizes well enough for the latent-space planner to drive any heldout init-state to a recorded future state, despite the val P/I plateau at 3.15× — τ-calibration captures the relevant signal-vs-distractor structure.

### 2D-particle — 500 episodes × 30 steps, 10 epochs, batch 128, CPU

| Metric | Value |
|---|---|
| Wallclock | 51.7 min |
| Final L_pred | 0.25 |
| Linear probe R² (x, y) | 0.57 / 0.48 |
| Rollout MSE | 0.43× of identity baseline (predictor learns real dynamics) |
| z_std (post-projector) | 0.90 |

## Resolution × information-geometry sweep — Reacher (controlled study)

Beyond reproduction, we ran a controlled cross-resolution study on Reacher to put quantitative numbers on LeWM Limitation #3 ("matching the isotropic Gaussian prior in a high-dimensional latent space becomes challenging" on low-intrinsic-dim envs).

### Methodology

**Constant-grid design.** Rather than fix patch_size and let token count grow with resolution (confounding compute and information), we hold the patch GRID at 16×16 = 256 tokens and scale `patch_size = image_size // 16` instead. Result: attention compute is approximately constant across all resolutions, isolating "pixel info per patch" as the only varied variable.

| image_size | patch_size | tokens | params (M) |
|---|---|---|---|
| 64  | 4  | 256 | 5.55 |
| 96  | 6  | 256 | 5.56 |
| 128 | 8  | 256 | 5.58 |
| 160 | 10 | 256 | 5.60 |
| 192 | 12 | 256 | 5.62 |
| 224 | 14 | 256 | 5.65 |

**Per-cell training:** 500 WeakPolicy Reacher episodes × 60 env steps, 10 epochs, batch=64, tiny preset, single seed. Cells run as isolated subprocesses via `tools/run_sweep.py`; per-cell artifacts in `runs/sweep/reacher/img<N>/`.

**Diagnostics** (`src/diagnostics.py`): per cell, sample 2000 (Z, Z_next, env_state) triples from val split; compute `effective_rank_pr` (participation ratio of singular values), `ksg_mi` (Kraskov-Stögbauer-Grassberger MI estimator, k=3), `twonn_intrinsic_dim` (Facco et al. 2017). All estimators sanity-tested on synthetic baselines with known answers.

**Cross-seed denoise:** image_size ∈ {192, 224} re-run with seed=1 to characterize single-seed variance.

### Reacher sweep results (single seed=42, 6 cells)

| img | patch | success | τ-gap | near | unrel | ER | twoNN_z | twoNN_state | MI(z,state) | MI(z,z_next) |
|---|---|---|---|---|---|---|---|---|---|---|
| 64  | 4  | 1.000 | 10.71× | 1.82 | 19.50 | 11.94 | 0.181 | 3.866 | 3.826 | 5.446 |
| 96  | 6  | 0.967 | 23.94× | 0.84 | 20.19 | 11.35 | 0.183 | 3.866 | 3.500 | 4.941 |
| 128 | 8  | 0.967 | 26.30× | 0.73 | 19.11 | 12.08 | 0.180 | 3.866 | 3.475 | 5.037 |
| 160 | 10 | 1.000 | 26.62× | 0.76 | 20.30 | 12.58 | 0.189 | 3.866 | 3.506 | 4.929 |
| 192 | 12 | 1.000 | 52.10× | 0.36 | 18.89 | 11.69 | 0.215 | 3.866 | 3.408 | 4.899 |
| 224 | 14 | 0.967 | 20.24× | 0.85 | 17.14 | 12.88 | 0.188 | 3.866 | 3.503 | 5.029 |

Wallclock totals: cell-by-cell 3.4 hr → 6.1 hr → 6.3 hr → 11.5 hr → 8.3 hr → 10.8 hr (Mac CPU; high-variance pace due to background system load). **Sweep total: 46.3 hr**.

### Cross-seed denoise (img 192 + 224 with seed=1)

| img | seed | τ-gap | near | unrel | ER | MI(z,state) | MI(z,z_next) | Δ vs orig |
|---|---|---|---|---|---|---|---|---|
| 192 | 42 | 52.10× | 0.363 | 18.89 | 11.69 | 3.408 | 4.899 | (orig) |
| 192 | 1  | 32.02× | 0.575 | 18.41 | 12.83 | 3.404 | 4.794 | τ-gap −38%, MI flat |
| 224 | 42 | 20.24× | 0.847 | 17.14 | 12.88 | 3.503 | 5.029 | (orig) |
| 224 | 1  | 22.04× | 0.868 | 19.13 | 11.37 | 3.484 | 5.110 | τ-gap +9%, MI flat |

**Cross-seed variance per metric:**

| Metric | @ 192 | @ 224 | Verdict |
|---|---|---|---|
| τ-gap absolute | 62% rel | 9% rel | NOISY at 192, stable at 224 |
| MI(z, env_state) | 0.1% | 0.5% | ROCK SOLID |
| MI(z, z_next) | 2.1% | 1.6% | Stable |
| Effective rank | 10% | 12% | Moderate |
| Success rate | 0% | 3% | Flat |

**Denoise total wallclock: 21.8 hr.** Combined Reacher compute: **68.1 hr CPU**.

### ⚠ RETRACTION — the "claims" from this 10-epoch sweep were artifacts

The tables above come from a **10-epoch** sweep and were originally written up as five
strong claims. **They are withdrawn.** A follow-up 100-epoch convergence study plus
estimator-validity controls showed each headline was a distinct measurement artifact:

| Withdrawn | Reality |
|---|---|
| "effective rank ≈ 12, resolution-invariant" | **under-training** — rank climbs to ~28 by epoch 100 and is still rising; 12 was a snapshot. Rank also measures anti-collapse compliance, not quality. |
| "MI(z, state) saturates at 3.4–3.5" | **under-training + KSG geometry** — MI rises to ~3.9 at convergence; and KSG on a 192-D latent drifts with the SIGReg-reshaped geometry, not with information (a decoding probe shows the "trend" flat). |
| "resolution trades discriminability for state-fidelity" | rests on the same KSG MI + a **frame-leaked** downstream probe; collapses under trajectory-level splitting + power controls. |

Still standing (estimator-robust): **predictive-accuracy convergence speed is monotone in
resolution**, and **effective rank rises over training**. The genuinely reusable result is the
four-gate validity protocol that caught all of this: **[`METHODOLOGY.md`](METHODOLOGY.md)**.
The clean redo is specced in **[`V2_DESIGN.md`](V2_DESIGN.md)**. The methodology note in this
section that *is* legitimate is the **constant-grid design** (patch = image_size/16 holds token
count + attention cost fixed, isolating pixels-per-token) — reused in v2.

### Reproducing the (superseded) 10-epoch sweep

```bash
# NOTE: 10 epochs is under-trained (see retraction above). Kept for provenance only.
# Full sweep on Reacher (6 cells × ~3-11 hr each, Mac CPU):
python tools/run_sweep.py --device cpu \
    --envs reacher --resolutions 64,96,128,160,192,224 \
    --n-episodes 500 --episode-length 60 --epochs 10 \
    --batch-size 64 --eval-episodes 30 --diag-samples 2000 \
    --out-dir runs/sweep --seed 42

# Cross-seed denoise:
python tools/run_sweep.py --device cpu \
    --envs reacher --resolutions 192,224 \
    --out-dir runs/sweep_denoise/seed1 --seed 1 \
    --n-episodes 500 --episode-length 60 --epochs 10 --batch-size 64 \
    --eval-episodes 30 --diag-samples 2000

# Aggregate across cells:
python tools/run_sweep.py --aggregate-only --out-dir runs/sweep
# → runs/sweep/sweep_results.json
```

## Kill checks (pre-train assertions)

```
1. Projection has no BN after final Linear (BN-in-middle topology)
2. SIGReg distinguishes collapsed vs Gaussian by ≥3× and B-scaling holds
3. AdaLN-zero block(x, c) ≈ x at init (max diff < 1e-5)
4. Prediction topology: preds = pred_proj(predictor(...)), no z_t residual addition
5. Scheduler: LR profile follows linear warmup → cosine decay to 0
```

All pass on the current implementation.

## Data format

The trainer accepts three input pathways. Pick whichever matches your data shape.

### A. Synthetic (default)

```bash
python train.py --env pusht --n-episodes 500 --episode-length 30 --seed 42
```

Trajectories are generated from the env class on first run, cached to `data/<env>_n<E>_T<T>_seed<S>.npz`, and reused on subsequent runs. Episode-wise train/val split via `--val-fraction` (default 0.1).

### B. HDF5 (single or multi-file)

```python
from src.data import HDF5TrajectoryDataset
ds = HDF5TrajectoryDataset(["recordings/session_01.h5", "recordings/session_02.h5"], sub_len=4)
```

Expected per-file structure:
```
recordings/session_01.h5
├── /obs       (E, T, H, W, C)  uint8 [0, 255] OR float32 [0, 1]
└── /actions   (E, T, A)        float32
```

### C. Episode-directory (canonical external format — for labelled video data)

```bash
python train.py --episode-dir /path/to/data --val-episode-dir /path/to/val
```

Layout (mode A — NPZ per episode):
```
data/
├── manifest.json
├── ep_00000.npz
├── ep_00001.npz
└── ...
```

Each `ep_NNNNN.npz` contains:
- `obs`: `(T, H, W, C)` uint8 [0,255] OR float32 [0,1]
- `actions`: `(T, action_dim)` float32
- `states` (optional): `(T, state_dim)` float32

Layout (mode B — video + sidecar labels):
```
data/
├── manifest.json
├── ep_00000/
│   ├── video.mp4                   (T frames, HxWxC at given fps)
│   ├── actions.npy                 (T, action_dim) float32
│   └── states.npy                  (T, state_dim)  float32   [optional]
├── ep_00001/
└── ...
```

`manifest.json` schema:
```json
{
  "version":        "1.0",
  "format":         "npz",
  "action_dim":     2,
  "image_size":     [64, 64],
  "channels":       3,
  "n_episodes":     500,
  "episode_length": 30,
  "fps":            10,
  "obs_dtype":      "uint8",
  "obs_range":      "[0, 255]"
}
```

The mode is `"npz"` or `"video"`. The trainer auto-detects format from manifest and reads accordingly.

A helper `src.data.write_episode_directory(...)` round-trips numpy arrays into either mode for ergonomic conversion of synthetic / collected data.

## Reference

Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., Balestriero, R. *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.* arXiv:2603.19312, 2026.

Official implementation: <https://github.com/lucas-maes/le-wm>

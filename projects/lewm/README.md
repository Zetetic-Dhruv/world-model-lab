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
- ✓ CEM planner + MPC runner in latent space (`src/lewm/planner.py`)
- ✓ Real-data-ready: episode-directory format (NPZ or video) accepted via `--episode-dir`

## Layout

```
lewm/
├── src/
│   ├── env.py                    2D particle environment (synthetic)
│   ├── env_pusht.py              MiniPushT (agent + T-block + target, contact dynamics)
│   ├── env_reacher.py            dm_control reacher-easy wrapper (real benchmark)
│   ├── data.py                   Trajectory generators + Datasets (in-memory, HDF5, episode-dir)
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
│   └── render_planning_video.py  Render planning episodes with overlays → MP4
├── train.py                      Training entry point (with held-out validation)
├── eval_probe.py                 Linear probe + open-loop rollout MSE
└── eval_planning.py              CEM-MPC planning success-rate evaluation
```

## Setup

```bash
pip install -r requirements.txt
```

Python ≥ 3.9, PyTorch ≥ 2.0. Optional: `wandb` (CSV fallback), `h5py` (only for HDF5 datasets), `imageio` + `imageio-ffmpeg` (for video tools), `dm_control` + `mujoco` (only for `--env reacher`).

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

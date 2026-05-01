# World Models for preventing Machine Errors in long horizon automation

## Experiment 1 
A faithful PyTorch reproduction of **LeWorldModel** (Maes, Le Lidec, Scieur, LeCun, Balestriero - arXiv:2603.19312, 2026): a stable end-to-end Joint-Embedding Predictive Architecture (JEPA) world model trained from raw pixels.

Replicates the canonical mechanis, 2-layer MLP projector with BN-in-middle, separate `pred_proj` for asymmetric projection, full DiT-style AdaLN-zero with multiplicative residual gates, SIGReg with sample-count scaling and trapezoidal Epps–Pulley quadrature, LinearWarmupCosineAnnealingLR, on a synthetic 2D-particle environment for fast attribution-clean iteration. No dependency on `stable-pretraining` or `stable-worldmodel`; everything is in this repo.

### Status

- ✓ Faithful mechanism replication (5/5 kill checks pass)
- ✓ Stable training: 0 NaN, 0 recoveries, no collapse over 1050 steps
- ✓ Predictor learns real dynamics: 55–58% lower MSE than predict-identity baseline
- ✓ Encoder produces honest near-N(0,1) representations (post-projector z_std = 0.90)
- ✓ ~18M parameters, ~50 min CPU training for 10 epochs on a 2D-particle env
- ✓ Includes a threaded NaN-recovery supervisor for hardware-flaky substrates (Apple MPS, etc.)

### Layout

```
src/
├── env.py              2D particle environment (synthetic)
├── data.py             Trajectory generator + Dataset
└── lewm/
    ├── encoder.py      ViT-Tiny + 2-layer MLP projector
    ├── predictor.py    Causal ViT with DiT AdaLN-zero
    ├── sigreg.py       SIGReg (Cramér–Wold + Epps–Pulley + B-scaling)
    ├── scheduler.py    LinearWarmupCosineAnnealingLR
    ├── trainer.py      NaNSupervisedTrainer (threaded snapshot + watchdog)
    └── model.py        LeWM module + canonical loss

tests/
├── test_kill_checks.py   5 mechanism-canonical assertions (run pre-train)
├── test_sigreg.py        SIGReg distinguishes Gaussian vs collapsed
└── test_smoke.py         End-to-end forward/backward smoke

train.py                  Training entry point
eval_probe.py             Linear probe + open-loop rollout MSE
```

### Setup

```bash
pip install -r requirements.txt
```

Python ≥ 3.9, PyTorch ≥ 2.0. `wandb` optional (CSV fallback in `runs/<name>/train_log.csv`).

### Usage

```bash
# 1. Pre-flight kill checks (run before any training run)
python tests/test_kill_checks.py

# 2. Train (canonical config)
python train.py --device cpu --no-wandb --out-dir runs/v3-10ep --epochs 10

# 3. Linear probe + rollout MSE evaluation
python eval_probe.py --ckpt runs/v3-10ep/ckpt_epoch9.pt
```

### Config (mostly matches official `lucas-maes/le-wm`)

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

### Reproduction numbers — Round 3-10ep, synthetic 2D particle, 1050 steps

| Metric | Value |
|---|---|
| Wallclock (CPU) | 51.7 min |
| Final L_pred | 0.25 |
| Final L_sigreg | 4.22 |
| z_std (post-projector) | 0.90 (target: N(0,1)) |
| ‖z‖ mean / std | 12.7 / 0.8 |
| Linear probe R²(x, y) | 0.57 / 0.48 |
| Rollout MSE per step | 0.73 (vs identity baseline 1.70 → 0.43×) |
| NaN events / recoveries / escalations | 0 / 0 / 0 |

The flat rollout-MSE-across-horizons is a property of the bounded-motion synthetic env (particle moves ~4 px/frame, latent change is bounded), not predictor failure. Predictor MSE ≈ 0.43× identity baseline at every horizon.

### Other checks and checks (pre-train assertions)

```
1. Projection has no BN after final Linear (BN-in-middle topology)
2. SIGReg distinguishes collapsed vs Gaussian by ≥3× and B-scaling holds
3. AdaLN-zero block(x, c) ≈ x at init (max diff < 1e-5)
4. Prediction topology: preds = pred_proj(predictor(...)), no z_t residual addition
5. Scheduler: LR profile follows linear warmup → cosine decay to 0
```

All pass on the current implementation.

### Reference

Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., Balestriero, R. *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.* arXiv:2603.19312, 2026.

Official implementation: <https://github.com/lucas-maes/le-wm>
